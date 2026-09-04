import json
import logging
import os.path
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models import build_loss
from pyquaternion import Quaternion

import torch.nn.functional as F

from mmdet3d.core.bbox.coders import build_bbox_coder
from ...mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ...mmdet3d_plugin.models.utils.grid_mask import GridMask
from .attention_dert3d import inverse_sigmoid
from . import predictor_lib
from .memory_bank import build_memory_bank
from .qim import build_qim
from .radar_encoder import build_radar_encoder
from .bev_vis import visualize_bev
from .. import utils as predictor_utils
from ..structures import Instances


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=None, filter_score_thresh=None, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.link_track_id = False
        self.last_track_instances = None

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))

                if True:
                    track_instances.obj_idxes[i] = self.max_obj_id
                    self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1

    def update_fix_label(self, track_instances: Instances, old_class_scores):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] >= self.filter_score_thresh:
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]


class ImageGuidedBEVProjection(nn.Module):
    """
    Projects image features onto the BEV plane via per-camera cross-attention.
    Implements Figure 4 of TransFusion (Section 3.6):
      1. Collapse image height axis via max → column features [B, W, C] per camera
      2. Per-camera MHA: Q=BEV locations, K/V=camera columns → F_LC_i per camera
      3. Average F_LC_i across cameras → F_LC [B, C_bev, H, W]
    """
    def __init__(self, bev_channels, img_channels=256, embed_dims=256, num_heads=8, num_cams=6):
        super().__init__()
        self.num_cams = num_cams
        self.bev_proj = nn.Linear(bev_channels, embed_dims)
        self.img_proj = nn.Linear(img_channels, embed_dims)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
            for _ in range(num_cams)
        ])
        self.out_proj = nn.Linear(embed_dims, bev_channels)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, bev_feat, img_feats):
        """
        Args:
            bev_feat:  [B, C_bev, H, W]
            img_feats: list of [B, N_cams, C_img, H_img, W_img]
        Returns:
            F_LC: [B, C_bev, H, W]
        """
        B, C_bev, H, W = bev_feat.shape
        img = img_feats[0]  # highest-res FPN level
        _, N_cams, C_img, H_img, W_img = img.shape

        # Collapse height axis via max → [B, N_cams, W_img, C_img] then project
        img_cols = img.max(dim=3).values           # [B, N_cams, C_img, W_img]
        img_cols = img_cols.permute(0, 1, 3, 2)    # [B, N_cams, W_img, C_img]
        img_cols = self.img_proj(img_cols)          # [B, N_cams, W_img, D]

        # BEV as queries → [B, H*W, D]
        bev_q = self.bev_proj(bev_feat.permute(0, 2, 3, 1).reshape(B, H * W, C_bev))

        # Per-camera attention, average across cameras
        n = min(N_cams, self.num_cams)
        cam_outs = []
        for i in range(n):
            kv_i = img_cols[:, i]                          # [B, W_img, D]
            attn_i, _ = self.cross_attns[i](bev_q, kv_i, kv_i)
            cam_outs.append(attn_i)
        attn_avg = torch.stack(cam_outs, dim=0).mean(dim=0)  # [B, H*W, D]
        attn_avg = self.norm(bev_q + attn_avg)

        F_LC = self.out_proj(attn_avg).reshape(B, H, W, C_bev).permute(0, 3, 1, 2)
        return F_LC


@DETECTORS.register_module()
class ViP3D(MVXTwoStageDetector):
    def __init__(self,
                 embed_dims=256,
                 num_query=300,
                 num_classes=7,
                 # NEW 
                use_lidar=False,
                lidar_bev_channels=256,
                lidar_voxel_size=None,
                lidar_out_size_factor=4,
                heatmap_score_thresh=0.1,
                # END NEW
                 bbox_coder=None,
                 qim_args=None,
                 mem_cfg=None,
                 radar_encoder=None,
                 fix_feats=False,
                 fix_lidar=False,
                 score_thresh=None,
                 filter_score_thresh=None,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 loss_cfg=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 do_pred=False,
                 predictor=None,
                 relative_pred=False,
                 agents_layer_0=False,
                 agents_layer_0_num=2,
                 only_matched_query=False,
                 add_branch=False,
                 add_branch_2=False,
                 debug=False,
                 bev_vis=True,
                 vis_interval=20,
                 use_img_guided=False,
                 use_smca=False,
                 ):
        # Thread pc_range down into the decoder's nested config, once, before
        # super().__init__() builds pts_bbox_head from this dict.
        if use_lidar and pts_bbox_head is not None:
            _pc_range = bbox_coder.get('pc_range') if bbox_coder else None
            pts_bbox_head = dict(pts_bbox_head)
            transformer_cfg = dict(pts_bbox_head.get('transformer', {}))
            decoder_cfg = dict(transformer_cfg.get('decoder', {}))
            decoder_cfg['pc_range'] = _pc_range
            transformer_cfg['decoder'] = decoder_cfg
            pts_bbox_head['transformer'] = transformer_cfg

        super(ViP3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        import plugin.vip3d.models.bev_vis as _bv_mod
        _bv_mod.ENABLED      = bev_vis
        _bv_mod.DEBUG_PRINTS = debug
        _bv_mod.VIS_INTERVAL = vis_interval

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = embed_dims
        self.num_query = num_query
        self.fix_feats = fix_feats
        if self.fix_feats:
            self.img_backbone.eval()
            self.img_neck.eval()
        self.bbox_size_fc = nn.Linear(self.embed_dims, 3)

        self.use_smca = use_smca
        self.pts_bbox_head.transformer.decoder.use_smca = use_smca

        self.use_lidar = use_lidar
        self.heatmap_score_thresh = heatmap_score_thresh
        if fix_lidar and self.use_lidar:
            for m in [self.pts_voxel_encoder, self.pts_middle_encoder,
                      self.pts_backbone, self.pts_neck]:
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(False)
        if not self.use_lidar:
            # Original: fixed learnable query embeddings
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            self.reference_points = nn.Linear(self.embed_dims, 3)
        else:
            # LiViP3D: input-dependent init from LiDAR heatmap peaks
            self.lidar_bev_channels = lidar_bev_channels
            self.lidar_voxel_size = lidar_voxel_size 
            self.lidar_out_size_factor = lidar_out_size_factor
            self.category_embeds = nn.Embedding(num_classes, embed_dims)
            # heatmap_head: shared_conv (384→64, 3×3+BN+ReLU), pretrained from dense_head.shared_conv
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(lidar_bev_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            # Per-task heatmap heads loaded directly from CenterPoint — no merging, no adaptation.
            # task layout: 0=car(1), 1=truck+cveh(2), 2=bus+trailer(2), 4=moto+bike(2), 5=ped+cone(2)
            # Forward slices: car=t0[:,0], truck=t1[:,0], bus=t2[:,0], trailer=t2[:,1],
            #                 moto=t4[:,0], bike=t4[:,1], ped=t5[:,0]
            def _task_head(n_cls):
                return nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, n_cls, 3, padding=1),
                )
            self.hm_task0 = _task_head(1)   # car
            self.hm_task1 = _task_head(2)   # truck, cveh
            self.hm_task2 = _task_head(2)   # bus, trailer
            self.hm_task4 = _task_head(2)   # motorcycle, bicycle
            self.hm_task5 = _task_head(2)   # pedestrian, traffic_cone
            # Projects BEV channels → embed_dims for query content vector
            self.lidar_bev_proj = nn.Linear(lidar_bev_channels, embed_dims)

            self.use_img_guided = use_img_guided
            if self.use_img_guided:
                self.img_bev_proj = ImageGuidedBEVProjection(
                    bev_channels=lidar_bev_channels,
                    img_channels=256,
                    embed_dims=embed_dims,
                    num_cams=pts_bbox_head.get('num_cams', 6) if pts_bbox_head else 6,
                )
                self.img_hm_head = nn.Sequential(
                    nn.Conv2d(lidar_bev_channels, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.img_hm_task0 = _task_head(1)   # car
                self.img_hm_task1 = _task_head(2)   # truck, cveh
                self.img_hm_task2 = _task_head(2)   # bus, trailer
                self.img_hm_task4 = _task_head(2)   # motorcycle, bicycle
                self.img_hm_task5 = _task_head(2)   # pedestrian, traffic_cone

        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=5)

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.memory_bank = build_memory_bank(
            args=mem_cfg,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length

        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None

        self.radar_encoder = build_radar_encoder(radar_encoder) if radar_encoder else None

        self.do_pred = do_pred
        self.relative_pred = relative_pred

        self.add_branch = add_branch
        self.add_branch_2 = add_branch_2
        if self.add_branch:
            from mmcv.utils import build_from_cfg
            from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER)
            self.add_branch_mlp = predictor_lib.MLP(256, 256)
            self.add_branch_attention = build_from_cfg(predictor_utils.get_attention_cfg(), TRANSFORMER_LAYER)

        if True:
            self.agents_layer_0 = agents_layer_0
            if self.agents_layer_0:
                self.agents_layer_mlp_0 = nn.Sequential(*[predictor_lib.MLP(256, 256) for _ in range(agents_layer_0_num)])

        self.only_matched_query = only_matched_query

        if self.do_pred:
            from .predictor_vectornet import VectorNet
            self.predictor = VectorNet(**predictor)
            self.empty_linear = nn.Linear(embed_dims, embed_dims)

    def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta):
        '''
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        '''
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())

        if time_delta > 1. or time_delta < 0.:
            time_delta = torch.tensor(0., device=time_delta.device, dtype=time_delta.dtype)

        # class 'torch.Tensor'> tensor([0.5005], device='cuda:0', dtype=torch.float64)
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # Update first: according to query velocity
        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1.T + l2g_t1
        # fix — rotation matrix: inv(R).T = R:
        ref_pts = (ref_pts - l2g_t2) @ l2g_r2.type(torch.float)
        # ref_pts = (ref_pts - l2g_t2) @ torch.linalg.inv(l2g_r2).T.type(torch.float)

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                # img: fp16
                img = self.grid_mask(img)
                # img: fp32
            # slightly different

            # img: fp32
            img_feats = self.img_backbone(img)
            # img: fp16

        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img', 'points', 'radar'), out_fp32=True)
    def extract_feat(self, points, img, radar=None, img_metas=None):
        """Extract features from images and points."""
        if radar is not None:
            radar_feats = self.radar_encoder(radar)
        else:
            radar_feats = None
        if self.fix_feats:
            with torch.no_grad():
                img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = self.extract_img_feat(img, img_metas)

        bev_feats = None
        if self.use_lidar and points is not None:
            bev_feats = self.extract_pts_feat(points, img_feats, img_metas)

        return (img_feats, radar_feats, bev_feats)

    def _targets_to_instances(self, gt_bboxes_3d=None,
                              gt_labels_3d=None, instance_inds=None,
                              img_shape=(1, 1,)):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = gt_bboxes_3d
        gt_instances.labels = gt_labels_3d
        gt_instances.obj_ids = instance_inds
        return gt_instances

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries = self.num_query
        dim = self.embed_dims * 2
        device = self.bbox_size_fc.weight.device  # always exists, no query_embedding dependency

        if not self.use_lidar:
            query = self.query_embedding.weight          # (300, 512) learnable
            ref_pts = self.reference_points(query[..., :dim // 2])
        else:
            # Zeros — will be filled per-frame in _forward_single from heatmap
            query = torch.zeros((num_queries, dim), dtype=torch.float, device=device)
            ref_pts = torch.zeros((num_queries, 3), dtype=torch.float, device=device)

        # init box sizes from content half of query
        box_sizes = self.bbox_size_fc(query[..., :dim // 2])
        pred_boxes_init = torch.zeros((num_queries, 10), dtype=torch.float, device=device)
        pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
        pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]

        track_instances.ref_pts = ref_pts
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full(
            (num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (num_queries,), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (num_queries,), dtype=torch.long, device=device)
        track_instances.scores = torch.zeros(
            (num_queries,), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = pred_boxes_init
        track_instances.pred_logits = torch.zeros(
            (num_queries, self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (num_queries, mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones(
            (num_queries, mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            (num_queries,), dtype=torch.float32, device=device)

        return track_instances.to(device)


    def _copy_tracks_for_loss(self, tgt_instances):

        device = self.bbox_size_fc.weight.device
        
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        track_instances.save_period = deepcopy(tgt_instances.save_period)
        return track_instances.to(self.bbox_size_fc.weight.device)

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            if self.do_pred:
                self.predictor.decoder.do_eval = True
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'radar', 'points'))
    def _forward_single(self, points, img, radar, img_metas, track_instances,
                        l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None,
                        time_delta=None, is_last_frame=False,
                        gt_bboxes_3d=None,
                        gt_labels_3d=None,
                        mapping=None,
                        ):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]

        if l2g_r2 is None or l2g_t2 is None:
            no need to call velo update
        '''
        # l2g_r2 is rotation matrix of next frame

        if img is not None:  # lidar-only: img is None, B/num_cam/H/W unused
            B, num_cam, _, H, W = img.shape

        if self.use_lidar and not hasattr(self, '_heatmap_verified'):
            self._heatmap_verified = True
            print(f'[verify] heatmap_head[0].weight.sum()={self.heatmap_head[0].weight.data.sum().item():.3f} (expected -2340.769)')
            print(f'[verify] hm_task0[-1].bias={self.hm_task0[-1].bias.data.tolist()} (expected [-0.375])')

        #always run
        if True:
            # extract features - run all sensor backbones for the frame
            img_feats, radar_feats, pts_feats = self.extract_feat(
                points, img=img, radar=radar, img_metas=img_metas)

            ref_box_sizes = torch.cat(
                [track_instances.pred_boxes[:, 2:4],
                track_instances.pred_boxes[:, 5:6]], dim=1)

        # ── LiVip add: fill empty slots + heatmap loss 
        if self.use_lidar and pts_feats is not None:
            bev_feat = pts_feats[0] if isinstance(pts_feats, (list, tuple)) else pts_feats
            shared   = self.heatmap_head(bev_feat)                      # [B, 64, H, W]
            t2 = self.hm_task2(shared); t4 = self.hm_task4(shared); t5 = self.hm_task5(shared)
            heatmap  = torch.cat([                                      # [B, 7, H, W]
                self.hm_task0(shared),          # car
                self.hm_task1(shared)[:, :1],   # truck
                t2[:, :1],                       # bus
                t2[:, 1:],                       # trailer
                t4[:, :1],                       # motorcycle
                t4[:, 1:],                       # bicycle
                t5[:, :1],                       # pedestrian
            ], dim=1)

            feat_for_queries = bev_feat
            if self.use_img_guided and img_feats is not None:
                F_LC = self.img_bev_proj(bev_feat, img_feats)
                img_shared = self.img_hm_head(F_LC)
                _it2 = self.img_hm_task2(img_shared)
                _it4 = self.img_hm_task4(img_shared)
                _it5 = self.img_hm_task5(img_shared)
                img_heatmap = torch.cat([
                    self.img_hm_task0(img_shared),
                    self.img_hm_task1(img_shared)[:, :1],
                    _it2[:, :1], _it2[:, 1:],
                    _it4[:, :1], _it4[:, 1:],
                    _it5[:, :1],
                ], dim=1)
                heatmap = (heatmap + img_heatmap) * 0.5
                feat_for_queries = F_LC

            # 1. fill so far empty slots
            empty_mask = track_instances.obj_idxes < 0
            num_empty  = int(empty_mask.sum())
            if num_empty > 0:
                active_ref = track_instances.ref_pts[track_instances.obj_idxes >= 0] \
                            if (track_instances.obj_idxes >= 0).any() else None
                new_ref_pts, new_queries, _ = self.select_topk_from_heatmap(
                    heatmap, feat_for_queries, num_select=num_empty, active_ref_pts=active_ref, img_metas=img_metas)

                track_instances.ref_pts = track_instances.ref_pts.clone()
                track_instances.query   = track_instances.query.clone()
                track_instances.ref_pts[empty_mask] = new_ref_pts
                track_instances.query[empty_mask]   = new_queries

                # debug vis - store top-20 for SMCA visualisation
                from .bev_vis import set_top_queries as _set_tq
                empty_indices = empty_mask.nonzero(as_tuple=False)[:, 0]
                top20_idx = empty_indices[:20]
                _set_tq(track_instances.ref_pts[top20_idx], top20_idx)

            # 2. heatmap supervision — independent of whether any slots were filled
            gt_hm_vis = None
            if self.training and gt_bboxes_3d is not None and gt_labels_3d is not None:
                H_bev, W_bev = heatmap.shape[2], heatmap.shape[3]
                boxes_t    = gt_bboxes_3d[0].tensor.to(bev_feat.device) \
                            if hasattr(gt_bboxes_3d[0], 'tensor') else gt_bboxes_3d[0]
                labels_t   = gt_labels_3d[0].to(bev_feat.device)
                boxes_norm = normalize_bbox(boxes_t, self.pc_range)
                gt_hm      = self._generate_gt_heatmap(
                    boxes_norm, labels_t, H_bev, W_bev, bev_feat.device)   # [K, H, W]
                gt_hm_vis  = gt_hm
                pred_hm    = heatmap[0].sigmoid()

                # don't apply visibillity mask - heat map loss is only applied to visible objects
                # vis_mask = self._build_camera_visibility_mask(
                #     H_bev, W_bev, img_metas, bev_feat.device)
                # gt_hm = gt_hm * vis_mask.unsqueeze(0).float()

                # only exact peak cells get 1
                pos_mask    = gt_hm.eq(1).float()
                # the closer to exact peak -> weight close to 0
                neg_weights = torch.pow(1 - gt_hm, 4)

                # focal loss with alpha=0.25, gamma=2.0
                pos_loss = torch.log(pred_hm.clamp(min=1e-6)) * torch.pow(1 - pred_hm, 2) * pos_mask
                neg_loss = torch.log((1 - pred_hm).clamp(min=1e-6)) * torch.pow(pred_hm, 2) \
                        * neg_weights * gt_hm.lt(1).float()
                num_pos  = pos_mask.sum().clamp(min=1)
                # normalize by the number of positive cells (exact peaks)
                heatmap_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
                frame_idx = self.criterion._current_frame_idx
                self.criterion.losses_dict[f'frame_{frame_idx}_heatmap_loss'] = heatmap_loss

            visualize_bev(bev_feat, heatmap, gt_hm_vis)
        # LiVip add end

        # always runs regardless of use_lidar

        output_classes, output_coords, \
            query_feats, last_ref_pts = self.pts_bbox_head(
            img_feats, radar_feats, track_instances.query,
            track_instances.ref_pts, ref_box_sizes, img_metas,
            bev_feat=bev_feat if (self.use_lidar and pts_feats is not None) else None,
            alive_mask=track_instances.obj_idxes >= 0,
            velocity=track_instances.pred_boxes[:, -2:],
            cur_ego_r=l2g_r1, cur_ego_t=l2g_t1, time_delta=time_delta)

        # SMCA attention visualisation (runs after pts_bbox_head so attn is collected)
        if self.use_lidar and pts_feats is not None:
            from .bev_vis import visualize_train_smca as _vis_smca, visualize_smca_gauss as _vis_gauss
            _vis_smca(heatmap, img)
            _vis_gauss(img)

        if self.add_branch:
            self.update_history_img_list(img_metas, img, img_feats)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)  # nb_dec = how many decoder layers = how many "cards" to grade

        # the track id will be assigned by the mather.
        # only copy matched_gt_idxes, obj_idxes, etc.
        track_instances_list = [self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)]
        # ^ makes (nb_dec-1) disposable "practice cards" -- deep copies, only ID fields (obj_idxes etc)
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        # ^ writes onto the REAL card (still just a loose var here, not in the list yet)
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        # ^ just READS the already-predicted vx,vy from the last layer's box -- not a new prediction

        if True:
            if l2g_r2 is not None:
                ref_pts = self.velo_update(
                    last_ref_pts[0], velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta=time_delta)
            else:
                ref_pts = last_ref_pts[0]
            track_instances.ref_pts = ref_pts
            # ^ move REAL card's ref point into next frame's coordinates (skip if this is the last frame)

        # track_instances.query = torch.cat((track_instances.query[:, :self.embed_dims // 2],
        #   query_feats[0]), dim=1)
        # ^ dead/commented-out, ignore

        track_instances_list.append(track_instances)
        # ^ NOW the real card joins the list too: [practice card(s)..., REAL card] -- REAL card always LAST
        for i in range(nb_dec):
            track_instances = track_instances_list[i]
            # ^ move the "current card" pointer to card i (last iteration = the REAL card)
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            # ^ stamp THIS layer's predictions onto whichever card is currently pointed at

            # used keys in 'track_instances': 'pred_logits', 'pred_boxes', 'obj_idxes', 'matched_gt_idxes',
            # modified keys: 'matched_gt_idxes', 'obj_idxes'
            track_instances = self.criterion.match_for_single_frame(
                track_instances, i, if_step=(i == (nb_dec - 1)))
            # ^ grade this card vs GT (adds to loss), updates obj_idxes/matched_gt_idxes in place

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # ^ pointer is left on the REAL card (last loop iter) -- blend in ITS OWN past memory only

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)  # see qim.py
        # ^ refresh REAL card's query for next frame + drop dead slots -> this carries to frame+1
        return out_track_instances

    def forward_train(self,
                      points=None,
                      img=None,
                      radar=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      l2g_r_mat=None,
                      l2g_t=None,
                      gt_bboxes_ignore=None,
                      timestamp=None,
                      instance_idx_2_labels=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            radar (Torch.Tensor) of shape [B, T, num_points, radar_dim]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            l2g_r_mat (list[Tensor]). element shape [T, 3, 3]
            l2g_t (list[Tensor]). element shape [T, 3]
                points @ R_Mat + T
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        # [T, 3, 3]
        l2g_r_mat = l2g_r_mat[0]
        # change to [T, 1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)

        timestamp = timestamp

        bs = len(gt_bboxes_3d)          # batch size (always 1)
        num_frame = l2g_r_mat.size(0)   # T frames in this clip
        _device = l2g_r_mat.device      # use lidar pose tensor as device ref — img may be None
        # project GT to check camera visibility only when using a partial camera rig
        # (< 6 cams → no 360° coverage); with 6 cams or lidar-only all objects are visible
        check_cam_vis = (img is not None) and (self.pts_bbox_head.num_cams < 6)
        if check_cam_vis:
            img_h, img_w = img.shape[-2], img.shape[-1]
        track_instances = self._generate_empty_tracks()

        # init gt instances!
        # SUPERVISE VISIBLE GT INSTANCES
        # Compute per-instance camera visibility once, store as vis_mask.
        # All GT boxes stay in the count (keeps num_samples stable → stable loss
        # normalisation), but loss_labels and loss_boxes will zero out invisible ones.
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(_device)

            if check_cam_vis and len(boxes) > 0:
                lidar2img_gt = img_metas[0]['lidar2img']  # list[T] of [num_cam, 4, 4]
                centers = boxes[:, :3]
                pts_h = torch.cat([centers, torch.ones(len(centers), 1, device=_device)], dim=1)
                l2i = torch.tensor(np.array(lidar2img_gt[i]), dtype=torch.float32, device=_device)
                pts_cam = torch.einsum('cij,nj->nci', l2i, pts_h)  # [N, num_cam, 4]
                depth = pts_cam[..., 2]
                u = pts_cam[..., 0] / depth.clamp(min=1e-5)
                v = pts_cam[..., 1] / depth.clamp(min=1e-5)
                visible = (depth > 0) & (u > 0) & (u < img_w) & (v > 0) & (v < img_h)
                vis_mask = visible.any(dim=1)  # [N] True = visible in at least one camera
            elif check_cam_vis:
                vis_mask = torch.zeros(0, dtype=torch.bool, device=_device)
            else:
                vis_mask = torch.ones(len(boxes), dtype=torch.bool, device=_device)

            boxes = normalize_bbox(boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = instance_inds[0][i]
            gt_instances.vis_mask = vis_mask
            gt_instances_list.append(gt_instances)
        #     print(f'[GT vis] frame {i}: {vis_mask.sum().item()}/{len(vis_mask)} visible')
        # total = sum(len(g.vis_mask) for g in gt_instances_list)
        # kept  = sum(g.vis_mask.sum().item() for g in gt_instances_list)
        # print(f'[GT vis] clip total: {kept}/{total} visible ({100*kept//max(total,1)}%)')

        # reset call at the start of each training sample
        self.criterion.initialize_for_single_clip(gt_instances_list)

        others_dict = {}
        mapping = None
        if self.do_pred:
            mapping = kwargs['mapping'][0]
            same_scene = mapping['same_scene']
            valid_pred = mapping['valid_pred']

        if True:
            # for bs 1
            # extract lidar-image projection matrices for each frame; None if lidar-only
            lidar2img = img_metas[0]['lidar2img'] if img is not None else None  # [T, num_cam]; None for lidar-only
            for i in range(num_frame):

                # take out the i-th frame from full sequence; None if lidar-only
                points_single = [p_[i] for p_ in points] if points is not None else None
                img_single = torch.stack([img_[i] for img_ in img], dim=0) if img is not None else None  # None for lidar-only
                radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0) if radar is not None else None

                img_metas_single = deepcopy(img_metas)
                if lidar2img is not None:  # only set per-frame lidar2img when camera is active
                    #img_metas[0]['lidar2img'][i] gives the [num_cam, 4, 4]
                    img_metas_single[0]['lidar2img'] = lidar2img[i]

                if i == num_frame - 1:
                    l2g_r2 = None
                    l2g_t2 = None
                    time_delta = None
                else:
                    l2g_r2 = l2g_r_mat[i + 1]
                    l2g_t2 = l2g_t[i + 1]
                    time_delta = timestamp[i + 1] - timestamp[i]

                is_last_frame = i == num_frame - 1

                if True:
                    track_instances = self._forward_single(points_single, img_single,
                                                           radar_single, img_metas_single,
                                                           track_instances,
                                                           l2g_r_mat[i], l2g_t[i],
                                                           l2g_r2, l2g_t2, time_delta, is_last_frame,
                                                           gt_bboxes_3d=predictor_utils.tensors_tracking_to_detection(gt_bboxes_3d, i),
                                                           gt_labels_3d=predictor_utils.tensors_tracking_to_detection(gt_labels_3d, i),
                                                           mapping=mapping)

                if True:
                    track_instances = Instances.cat([track_instances, self._generate_empty_tracks()])

                if self.add_branch:
                    track_ids = []
                    decoded_boxes = []

                    all_decoded_boxes = predictor_utils.to_numpy(predictor_utils.get_decoded_boxes(track_instances.pred_boxes, self.pc_range, img_metas).tensor)
                    for j in range(len(track_instances)):
                        obj_id = track_instances.obj_idxes[j].item()
                        if obj_id != -1 and obj_id != -2:
                            if np.any(np.isnan(all_decoded_boxes[j])):
                                logging.getLogger('mmdet').warning(f'[NaN box] filtered obj_id={obj_id} frame={i}')
                                continue
                            track_ids.append(obj_id)

                            # clip
                            all_decoded_boxes[j][3:6] = np.clip(all_decoded_boxes[j][3:6], -20, 20)
                            decoded_boxes.append(all_decoded_boxes[j])

                    if i == 0:
                        self.track_idx_2_boxes = defaultdict(dict)
                        self.track_idx_2_boxes_in_lidar = defaultdict(dict)

                    # mapping must use last mapping
                    r_index_2_rotation_and_transform = mapping['r_index_2_rotation_and_transform']
                    if valid_pred:
                        predictor_utils.update_track_idx_2_boxes(self.track_idx_2_boxes, track_ids, decoded_boxes, r_index_2_rotation_and_transform[i], i)
                        predictor_utils.update_track_idx_2_boxes_in_lidar(self.track_idx_2_boxes_in_lidar, track_ids, decoded_boxes, r_index_2_rotation_and_transform[i], i)

        if self.do_pred:

            instance_idx_2_labels = instance_idx_2_labels[0]
            device = track_instances.output_embedding.device

            if not valid_pred:
                loss = None
            else:
                # valid agents which have been matched with GT
                agents_indices = []
                labels_list = []
                labels_is_valid_list = []
                if self.add_branch:
                    track_scores = []
                    track_ids = []
                    track_labels = []
                    all_track_labels = predictor_utils.get_labels_from_pred_logits(track_instances.pred_logits, self.bbox_coder.num_classes)

                for j in range(len(track_instances)):
                    # obj_idxes is obtained by matching
                    obj_id = track_instances.obj_idxes[j].item()
                    if obj_id != -1 and obj_id != -2 and obj_id in instance_idx_2_labels:
                        def run(future_traj=None,
                                future_traj_relative=None,
                                future_traj_is_valid=None,
                                past_traj=None,
                                past_traj_is_valid=None,
                                category=None,
                                past_boxes=None,
                                **kwargs):
                            if self.relative_pred:
                                labels_list.append(future_traj_relative)
                            else:
                                labels_list.append(future_traj)
                            labels_is_valid_list.append(future_traj_is_valid)

                        run(**instance_idx_2_labels[obj_id])
                        agents_indices.append(j)

                        if self.add_branch:
                            track_scores.append(track_instances.scores[j].item())
                            track_ids.append(obj_id)
                            track_labels.append(all_track_labels[j])

                if len(agents_indices) > 0:
                    if True:
                        if True:
                            output_embedding = track_instances.output_embedding

                            if self.add_branch:
                                output_embedding = output_embedding[torch.tensor(agents_indices, dtype=torch.long, device=device)]
                                _, _, past_boxes_list_in_lidar, _, _ = \
                                    predictor_utils.extract_from_track_idx_2_boxes(self.track_idx_2_boxes_in_lidar, track_scores, track_ids, track_labels, mapping, num_frame - 1)
                                query = self.add_branch_update_query(output_embedding, past_boxes_list_in_lidar[:, -1, :3], device)
                                output_embedding = output_embedding + query

                            output_embedding = self.output_embedding_forward(output_embedding)

                        if self.add_branch:
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              # agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                        elif self.only_matched_query:
                            output_embedding = output_embedding[torch.tensor(agents_indices, dtype=torch.long, device=device)]
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              # agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                        else:
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                else:
                    loss = None

            if loss is None:
                # zero loss as a placeholder
                loss = torch.abs(self.empty_linear(torch.zeros(self.embed_dims, device=device, dtype=torch.float))).mean()

            self.criterion.update_prediction_loss(loss)

        outputs = self.criterion.losses_dict
        return outputs, others_dict

    def _inference_single(self, points, img, radar, img_metas, track_instances,
                          l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None,
                          time_delta=None,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        '''

        # velo update:
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                time_delta=time_delta)
            active_inst.ref_pts = ref_pts
        track_instances = Instances.cat([other_inst, active_inst])

        if img is not None:  # lidar-only: img is None
            B, num_cam, _, H, W = img.shape

        if True:
            img_feats, radar_feats, pts_feats = self.extract_feat(
                points, img=img, radar=radar, img_metas=img_metas)
            img_feats = [a.clone() for a in img_feats] if img_feats is not None else None  # None for lidar-only

            # output_classes: [num_dec, B, num_query, num_classes]
            # query_feats: [B, num_query, embed_dim]
            ref_box_sizes = torch.cat(
                [track_instances.pred_boxes[:, 2:4],
                 track_instances.pred_boxes[:, 5:6]], dim=1)

            # ── LiViP3D: fill empty query slots from LiDAR heatmap ──────────────────
            bev_feat = None
            if self.use_lidar and pts_feats is not None:
                bev_feat = pts_feats[0] if isinstance(pts_feats, (list, tuple)) else pts_feats
                shared   = self.heatmap_head(bev_feat)
                _t2 = self.hm_task2(shared); _t4 = self.hm_task4(shared); _t5 = self.hm_task5(shared)
                heatmap  = torch.cat([
                    self.hm_task0(shared),
                    self.hm_task1(shared)[:, :1],
                    _t2[:, :1], _t2[:, 1:],
                    _t4[:, :1], _t4[:, 1:],
                    _t5[:, :1],
                ], dim=1)

                feat_for_queries = bev_feat
                if self.use_img_guided and img_feats is not None:
                    F_LC = self.img_bev_proj(bev_feat, img_feats)
                    img_shared = self.img_hm_head(F_LC)
                    _it2 = self.img_hm_task2(img_shared)
                    _it4 = self.img_hm_task4(img_shared)
                    _it5 = self.img_hm_task5(img_shared)
                    img_heatmap = torch.cat([
                        self.img_hm_task0(img_shared),
                        self.img_hm_task1(img_shared)[:, :1],
                        _it2[:, :1], _it2[:, 1:],
                        _it4[:, :1], _it4[:, 1:],
                        _it5[:, :1],
                    ], dim=1)
                    heatmap = (heatmap + img_heatmap) * 0.5
                    feat_for_queries = F_LC

                empty_mask = track_instances.obj_idxes < 0
                num_empty  = int(empty_mask.sum())
                if num_empty > 0:
                    active_ref = track_instances.ref_pts[track_instances.obj_idxes >= 0] \
                                if (track_instances.obj_idxes >= 0).any() else None
                    new_ref_pts, new_queries, hm_scores = self.select_topk_from_heatmap(
                        heatmap, feat_for_queries, num_select=num_empty, active_ref_pts=active_ref, img_metas=img_metas)

                    # Always fill all empty slots with heatmap positions so ref_pts are
                    # spread across the BEV rather than sitting at zero (vehicle origin).
                    # Track birth is still gated by TrackBase.update's score_thresh=0.4,
                    # so low-confidence peaks won't birth tracks even though they get positions.
                    n_fill = new_ref_pts.shape[0]
                    if n_fill > 0:
                        empty_indices = empty_mask.nonzero(as_tuple=False)[:, 0][:n_fill]
                        track_instances.ref_pts = track_instances.ref_pts.clone()
                        track_instances.query   = track_instances.query.clone()
                        track_instances.ref_pts[empty_indices] = new_ref_pts
                        track_instances.query[empty_indices]   = new_queries

            # ── end LiViP3D ──────────────────────────────────────────────────────────

            # ── Pass heatmap + GT to BEV visualiser ──────────────────────────
            from .bev_vis import set_gt_bev as _set_gt, set_heatmap as _set_hm
            _set_hm(heatmap if self.use_lidar and pts_feats is not None else None)
            if gt_bboxes_3d is not None:
                import numpy as np
                boxes = gt_bboxes_3d[0].tensor if hasattr(gt_bboxes_3d[0], 'tensor') \
                        else gt_bboxes_3d[0]        # [N, >=3] metric: x,y,z,...
                pc = self.pc_range
                x_n = ((boxes[:, 0].cpu().numpy() - pc[0]) / (pc[3] - pc[0])).clip(0, 1)
                y_n = ((boxes[:, 1].cpu().numpy() - pc[1]) / (pc[4] - pc[1])).clip(0, 1)
                _set_gt(np.stack([x_n, y_n], axis=1))
            else:
                _set_gt(None)
            # ─────────────────────────────────────────────────────────────────

            output_classes, output_coords, \
                query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, radar_feats, track_instances.query,
                track_instances.ref_pts, ref_box_sizes, img_metas,
                bev_feat=bev_feat,
                alive_mask=track_instances.obj_idxes >= 0,
                velocity=track_instances.pred_boxes[:, -2:],
                cur_ego_r=l2g_r1, cur_ego_t=l2g_t1, time_delta=time_delta)

        if self.add_branch:
            self.update_history_img_list(img_metas, img, img_feats)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        # TODO: Why no max?
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # track_scores = output_classes[-1, 0, :, 0].sigmoid()

        # Blend decoder score with heatmap confidence at each track's position.
        # TP tracks sit on strong heatmap peaks → blended score rises above 0.5 → persist.
        # FP tracks sit on noise → blended score drops below filter_score_thresh → die quickly.
        # if bev_feat is not None:
        #     ref_norm = last_ref_pts[0].sigmoid()                        # [N, 3] in [0,1]
        #     H, W = heatmap.shape[2], heatmap.shape[3]
        #     col = (ref_norm[:, 0] * W).long().clamp(0, W - 1)
        #     row = (ref_norm[:, 1] * H).long().clamp(0, H - 1)
        #     hm_at_track = heatmap[0].sigmoid()[:, row, col].max(dim=0).values  # [N]
        #     track_scores = 0.5 * track_scores + 0.5 * hm_at_track

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]

        track_instances.ref_pts = last_ref_pts[0]

        self.track_base.update(track_instances)
        # self.track_base.update_fix_label(track_instances, old_class_scores)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        return out_track_instances

    def forward_test(self,
                     points=None,
                     img=None,
                     radar=None,
                     img_metas=None,
                     timestamp=1e6,
                     l2g_r_mat=None,
                     l2g_t=None,
                     instance_idx_2_labels=None,
                     mapping=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     **kwargs,
                     ):
        """Forward test function.
        only support bs=1, single-gpu, num_frame=1 test
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # [3, 3]
        l2g_r_mat = l2g_r_mat[0][0]
        # change to [1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)[0]

        bs = len(img_metas)                      # always 1 at test time
        num_frame = len(img_metas[0]['lidar2img']) if 'lidar2img' in img_metas[0] else len(points[0])  # lidar-only: no lidar2img

        timestamp = timestamp[0]
        device = l2g_r_mat.device  # img may be None; derive device from pose tensor

        # track_instances of last frame
        if self.test_track_instances is None:
            if True:
                track_instances = self._generate_empty_tracks()
                self.test_track_instances = track_instances
            self.timestamp = timestamp[0]

        new_query = False
        no_metric_now = False
        self.track_base.link_track_id = False

        # TODO: use scene tokens?
        if True:
            assert timestamp[0] - self.timestamp >= 0
        if True:
            if timestamp[0] - self.timestamp > 10 or new_query:
                if True:
                    track_instances = self._generate_empty_tracks()
                time_delta = None
                l2g_r1 = None
                l2g_t1 = None
                l2g_r2 = None
                l2g_t2 = None
            else:
                if True:
                    track_instances = self.test_track_instances
                time_delta = timestamp[0] - self.timestamp
                l2g_r1 = self.l2g_r_mat
                l2g_t1 = self.l2g_t
                l2g_r2 = l2g_r_mat
                l2g_t2 = l2g_t
        self.timestamp = timestamp[-1]
        self.l2g_r_mat = l2g_r_mat
        self.l2g_t = l2g_t

        # for bs 1;
        lidar2img = img_metas[0].get('lidar2img')  # None for lidar-only
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points] if points is not None else None
            img_single = torch.stack([img_[i] for img_ in img], dim=0) if img is not None else None  # None for lidar-only
            radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0) if radar is not None else None

            img_metas_single = deepcopy(img_metas)
            if lidar2img is not None:  # only set for camera mode
                img_metas_single[0]['lidar2img'] = lidar2img[i]

            track_instances = self._inference_single(points_single, img_single,
                                                     radar_single,
                                                     img_metas_single,
                                                     track_instances,
                                                     l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                                                     time_delta,
                                                     gt_bboxes_3d=predictor_utils.tensors_tracking_to_detection(gt_bboxes_3d, i),
                                                     gt_labels_3d=predictor_utils.tensors_tracking_to_detection(gt_labels_3d, i))

            if True:
                track_instances = Instances.cat([track_instances, self._generate_empty_tracks()])

        # why again? select active has been performed in forward of qim.py
        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances))
        self.test_track_instances = track_instances

        results = self._active_instances2results(active_instances, img_metas)

        if mapping is not None:
            mapping = mapping[0]
        
        if self.do_pred and results[0] is not None and mapping['valid_pred'] \
                and (len(results[0]['track_ids']) > 0 and len(instance_idx_2_labels[0]) > 0):
            # only predict agent appeared at current index

            instance_idx_2_labels = instance_idx_2_labels[0]

            future_frame_num = 12

            index = mapping['index']
            valid_pred = mapping['valid_pred']

            if not hasattr(self, 'frame_index_2_mapping'):
                self.frame_index_2_mapping = {}
            self.frame_index_2_mapping[index] = mapping
            if index - 20 in self.frame_index_2_mapping:
                self.frame_index_2_mapping.pop(index - 20)

            if not hasattr(self, 'track_idx_2_boxes'):
                self.track_idx_2_boxes = defaultdict(dict)

            if True:
                # order of active_instances is different from results
                boxes_3d = predictor_utils.to_numpy(results[0]['boxes_3d'].tensor)
                output_embedding = results[0]['output_embedding']
                track_scores = predictor_utils.to_numpy(results[0]['track_scores'])
                track_ids = predictor_utils.to_numpy(results[0]['track_ids'])
                track_labels = predictor_utils.to_numpy(results[0]['labels_3d'])

            # 'boxes_3d' is in lidar, 'self.track_idx_2_boxes' is in global
            predictor_utils.update_track_idx_2_boxes(self.track_idx_2_boxes, track_ids, boxes_3d, mapping, index)

            # Input 'self.track_idx_2_boxes' is in global, output 'tracked_boxes_list' is in ego
            tracked_scores, tracked_trajs, tracked_boxes_list, tracked_boxes_is_valid_list, categories = \
                predictor_utils.extract_from_track_idx_2_boxes(self.track_idx_2_boxes, track_scores, track_ids, track_labels, mapping, index)

            gt_past_trajs, gt_past_trajs_is_valid, gt_future_trajs, gt_future_trajs_is_valid, gt_categories = \
                predictor_utils.get_gt_past_future_trajs(instance_idx_2_labels)

            labels, labels_is_valid = predictor_utils.get_labels_for_tracked_trajs(tracked_trajs, tracked_boxes_is_valid_list,
                                                                                   gt_past_trajs, gt_past_trajs_is_valid,
                                                                                   gt_future_trajs, gt_future_trajs_is_valid,
                                                                                   future_frame_num)

            if not valid_pred or len(tracked_boxes_list) == 0:
                outputs = None
            else:
                if output_embedding is not None:
                    if self.add_branch:
                        query = self.add_branch_update_query(output_embedding, boxes_3d[:, :3], device)
                        output_embedding = output_embedding + query

                    output_embedding = self.output_embedding_forward(output_embedding)

                if True:
                    loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                      device=device,
                                                      labels=[labels],
                                                      labels_is_valid=[labels_is_valid],
                                                      agents_indices=None,
                                                      mapping=[mapping],
                                                      **kwargs)

            pred_dict = None
            if outputs is not None:
                pred_outputs = outputs['pred_outputs'][0]
                pred_probs = outputs['pred_probs'][0]
                pred_outputs_single_traj = []

                if self.relative_pred:
                    normalizers = [predictor_utils.get_normalizer(tracked_boxes_is_valid_list[j], tracked_boxes_list[j])
                                   for j in range(len(tracked_boxes_list))]
                    assert len(normalizers) == len(pred_outputs)
                    for j in range(len(normalizers)):
                        np.set_printoptions(precision=3, suppress=True)
                        argmax = np.argmax(pred_probs[j])
                        pred_outputs_single_traj.append(pred_outputs[j, argmax])
                        pred_outputs[j] = normalizers[j](pred_outputs[j], reverse=True)

                self.add_pred_results(results[0], pred_outputs, pred_probs)

                pred_dict = dict(
                    instance_idx_2_labels=instance_idx_2_labels,
                    pred_outputs=pred_outputs,
                    pred_probs=pred_probs,
                    lanes=mapping['lanes'],
                    tracked_trajs=np.array(tracked_trajs),
                    tracked_trajs_is_valid=np.array(tracked_boxes_is_valid_list),
                )

        if results[0] is not None and 'output_embedding' in results[0]:
            results[0].pop('output_embedding')

        return results

    def add_pred_results(self, result_dict, pred_outputs, pred_probs):
        result_dict['pred_outputs_in_ego'] = pred_outputs
        result_dict['pred_probs_in_ego'] = pred_probs

    def _active_instances2results(self, active_instances, img_metas, do_train=False):
        '''
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        '''
        if do_train:
            assert False
            active_idxes = (active_instances.obj_idxes >= 0)
        else:
            # filter out sleep querys
            active_idxes = (active_instances.scores >= self.track_base.filter_score_thresh)
        active_instances = active_instances[active_idxes]
        if active_instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=active_instances.pred_logits,
            bbox_preds=active_instances.pred_boxes,
            track_scores=active_instances.scores,
            obj_idxes=active_instances.obj_idxes,
            output_embedding=active_instances.output_embedding,
        )
        # plugin/track/bbox_coder.py
        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        # (Pdb) img_metas[0]['box_type_3d'][0]
        # <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
        bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
        labels = bboxes_dict['labels']
        scores = bboxes_dict['scores']

        track_scores = bboxes_dict['track_scores']
        obj_idxes = bboxes_dict['obj_idxes']
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
            output_embedding=bboxes_dict['output_embedding']
        )

        return [result_dict]

    def train_step(self, data, optimizer):
        # Core model logic remains
        losses, others_dict = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # Output formatting
        outputs = dict(
            loss=loss, 
            log_vars=log_vars, 
            num_samples=len(data['img_metas'])
        )
        outputs.update(others_dict)
        
        return outputs

    def output_embedding_forward(self, output_embedding):
        if self.agents_layer_0:
            output_embedding = self.agents_layer_mlp_0(output_embedding)
        return output_embedding

    def update_history_img_list(self, img_metas, img, img_feats):
        if not hasattr(self, 'history_img_feats') or self.history_img_feats is None:
            self.history_img_feats = []
            self.history_img_metas = []
            self.history_img_list = []

        if True:
            self.history_img_feats.append(img_feats)
            self.history_img_metas.append(img_metas)
            self.history_img_list.append(img)

        if len(self.history_img_feats) > 3:
            self.history_img_feats = self.history_img_feats[1:]
            self.history_img_metas = self.history_img_metas[1:]
            self.history_img_list = self.history_img_list[1:]

    def add_branch_update_query(self, output_embedding, reference_points, device):
        reference_points = predictor_utils.reference_points_lidar_to_relative(reference_points, self.pc_range)
        reference_points = torch.tensor(reference_points, dtype=torch.float, device=device)

        query = self.add_branch_mlp(output_embedding)
        if self.history_img_feats[-1] is not None:  # skip camera cross-attn for lidar-only
            query = self.add_branch_attention(query=query.unsqueeze(1),
                                              reference_points=reference_points.unsqueeze(0),
                                              value=self.history_img_feats[-1],
                                              img_metas=self.history_img_metas[-1])
            assert query.shape == (len(output_embedding), 1, 256)
            query = query.squeeze(1)
        return query
    
    def _build_camera_visibility_mask(self, H, W, img_metas, device):
        lidar2img = torch.tensor(
            np.array(img_metas[0]['lidar2img']), dtype=torch.float32, device=device)
        # [num_cams, 4, 4]

        xs = torch.linspace(self.pc_range[0], self.pc_range[3], W, device=device)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        pts = torch.stack([grid_x, grid_y,
                        torch.zeros_like(grid_x),
                        torch.ones_like(grid_x)], dim=-1)  # [H, W, 4]
        pts_flat = pts.view(-1, 4, 1)  # [H*W, 4, 1]

        l2i = lidar2img.unsqueeze(0)  # [1, num_cams, 4, 4]
        pts_cam = torch.matmul(l2i, pts_flat.unsqueeze(1)).squeeze(-1)  # [H*W, num_cams, 4]

        depth = pts_cam[..., 2]
        u = pts_cam[..., 0] / depth.clamp(min=1e-5)
        v = pts_cam[..., 1] / depth.clamp(min=1e-5)

        img_h = img_metas[0]['img_shape'][0][0][0]
        img_w = img_metas[0]['img_shape'][0][0][1]

        visible = (depth > 0) & (u > 0) & (u < img_w) & (v > 0) & (v < img_h)
        return visible.any(dim=-1).view(H, W)  # [H, W]

    def select_topk_from_heatmap(self, heatmap, bev_feat, num_select, active_ref_pts=None, img_metas=None):
        """
        Select top-K object candidates from a BEV heatmap.

        Args:
            heatmap  (Tensor): [B, num_classes, H, W]  raw logits (pre-sigmoid)
            bev_feat (Tensor): [B, C, H, W]            BEV feature map
            num_select (int):  number of peaks to return (= number of empty slots)

        Returns:
            ref_pts    (Tensor): [num_select, 3]              inverse-sigmoid space
            query_feats(Tensor): [num_select, embed_dims * 2] content + pos
        """
        B, num_classes, H, W = heatmap.shape
        scores = heatmap.detach().sigmoid() # [B, K, H, W]

        # Suppress heatmap cells at active track locations
        if active_ref_pts is not None and len(active_ref_pts) > 0:
            pc_range = self.pc_range
            vs = self.lidar_voxel_size
            sf = self.lidar_out_size_factor
            anorm = active_ref_pts.sigmoid()
            ax = anorm[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            ay = anorm[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            ac = ((ax - pc_range[0]) / (vs[0] * sf)).long().clamp(0, W - 1)
            ar = ((ay - pc_range[1]) / (vs[1] * sf)).long().clamp(0, H - 1)
            sup_r = 2
            for r, c in zip(ar.tolist(), ac.tolist()):
                r, c = int(r), int(c)
                scores[:, :,
                    max(0, r - sup_r):min(H, r + sup_r + 1),
                    max(0, c - sup_r):min(W, c + sup_r + 1)] = 0.0

        if img_metas is not None and 'lidar2img' in img_metas[0]:  # skipped for lidar-only
            vis_mask = self._build_camera_visibility_mask(H, W, img_metas, heatmap.device)
            scores = scores * vis_mask.unsqueeze(0).unsqueeze(0).float()

        # Local-max suppression via 3×3 max-pool (keeps only local peaks).
        # NuScenes 7 classes: 0=car,1=truck,2=bus,3=trailer,4=moto,5=bicycle,6=pedestrian
        # Skip NMS for small classes (bicycle=5, pedestrian=6) as TransFusion does for
        # pedestrian/traffic_cone.
        SKIP_NMS = {5, 6}
        scores_nms = scores.clone()
        # for every cell, compute the maximum score in its 3x3 neighbourhood
        heatmap_max = F.max_pool2d(scores, 3, stride=1, padding=1)
        for c in range(num_classes):
            if c not in SKIP_NMS: # skip dense objects
                # a cell keeps its score only if it is the local maximum in 3x3 neighbourhood - otherwise 0
                scores_nms[:, c] = scores[:, c] * (scores[:, c] == heatmap_max[:, c]).float()

        # Flatten to [B, K*H*W] and take top-K across all classes
        scores_flat = scores_nms.view(B, -1)           # [B, num_classes*H*W]
        # select from all the anchors the topk=min(k, num_anchors) ones
        topk_scores, topk_inds = scores_flat.topk(      # [B, num_select]
            min(num_select, scores_flat.shape[-1]), dim=-1)

        # get the info for each chosen topk query
        topk_cls = topk_inds // (H * W)                # class index [B, num_select]
        topk_hw  = topk_inds  % (H * W)
        topk_h   = topk_hw // W                        # row    [B, num_select]
        topk_w   = topk_hw  % W                        # column [B, num_select]

        # (col, row) → metric (x, y) using voxel grid definition
        pc_range  = self.pc_range
        vs        = self.lidar_voxel_size              # [vx, vy, vz]
        sf        = self.lidar_out_size_factor
        # form h,w get the (x,y,z) in meters 
        x = pc_range[0] + (topk_w.float() + 0.5) * vs[0] * sf
        y = pc_range[1] + (topk_h.float() + 0.5) * vs[1] * sf
        z = x.new_full(x.shape, (pc_range[2] + pc_range[5]) / 2.0)

        # Normalize to [0,1] then inverse-sigmoid (matches how ref_pts are stored)
        x_n = (x - pc_range[0]) / (pc_range[3] - pc_range[0])
        y_n = (y - pc_range[1]) / (pc_range[4] - pc_range[1])
        z_n = (z - pc_range[2]) / (pc_range[5] - pc_range[2])
        xyz_norm = torch.stack([x_n, y_n, z_n], dim=-1).clamp(1e-6, 1 - 1e-6)
        # offsets are predicted in the inversed_sigmoid space so its more spread -> easier, need to store in inverse_sigmoid
        ref_pts = inverse_sigmoid(xyz_norm)            # [B, num_select, 3]

        # Sample BEV features at peak locations
        ns = topk_h.shape[1]
        b_idx = torch.arange(B, device=bev_feat.device).unsqueeze(1).expand(B, ns).reshape(-1)
        h_idx = topk_h.reshape(-1)
        w_idx = topk_w.reshape(-1)
        peak_feats = bev_feat[b_idx, :, h_idx, w_idx]      # [B*ns, C_bev]
        peak_feats = peak_feats.view(B, ns, -1)             # [B, ns, C_bev]

        # Project BEV features → embed_dims and add category embedding
        peak_feats = self.lidar_bev_proj(peak_feats)        # [B, ns, embed_dims]
        cat_embed  = self.category_embeds(topk_cls)         # [B, ns, embed_dims]
        content    = peak_feats + cat_embed                 # [B, ns, embed_dims]

        # query format used by pts_bbox_head: [num_query, embed_dims * 2]
        # first half = content, second half = positional (same as content here)
        queries = torch.cat([content, content], dim=-1)     # [B, ns, embed_dims*2]

        # Return for batch 0 (BS=1 assumption same as rest of ViP3D)
        return ref_pts[0], queries[0], topk_scores[0]       # [ns, 3], [ns, embed_dims*2], [ns]

    def _generate_gt_heatmap(self, gt_boxes, gt_labels, H, W, device):
        """
        Generate CenterPoint-style Gaussian heatmaps from GT 3D boxes.
        Fully on-GPU — no CPU transfers, no Python loops over boxes.

        Args:
            gt_boxes  (Tensor): [N, 10+] normalized boxes (output of normalize_bbox)
            gt_labels (Tensor): [N]      class indices
            H, W      (int):             heatmap spatial size
            device                       target device

        Returns:
            gt_heatmap (Tensor): [num_classes, H, W]  values in [0, 1]
        """
        gt_heatmap = torch.zeros((self.num_classes, H, W), dtype=torch.float32, device=device)
        if len(gt_boxes) == 0:
            return gt_heatmap

        pc_range = self.pc_range
        vs       = self.lidar_voxel_size
        sf       = self.lidar_out_size_factor

        # filter invalid classes
        valid = (gt_labels >= 0) & (gt_labels < self.num_classes)
        gt_boxes  = gt_boxes[valid]
        gt_labels = gt_labels[valid]
        if len(gt_boxes) == 0:
            return gt_heatmap

        # box[:2] are normalized x,y from normalize_bbox — denormalize to metric
        x_m = gt_boxes[:, 0]
        y_m = gt_boxes[:, 1]

        col = (x_m - pc_range[0]) / (vs[0] * sf)   # [N]
        row = (y_m - pc_range[1]) / (vs[1] * sf)   # [N]
        col_i = col.long()
        row_i = row.long()

        # filter out-of-bounds centers
        in_bounds = (col_i >= 0) & (col_i < W) & (row_i >= 0) & (row_i < H)
        gt_boxes  = gt_boxes[in_bounds]
        gt_labels = gt_labels[in_bounds]
        col_i     = col_i[in_bounds]
        row_i     = row_i[in_bounds]
        if len(gt_boxes) == 0:
            return gt_heatmap

        # radius per box from its footprint
        w_cells = gt_boxes[:, 2].exp() / (vs[0] * sf)
        l_cells = gt_boxes[:, 3].exp() / (vs[1] * sf)
        radii   = ((w_cells ** 2 + l_cells ** 2).sqrt() / 2).ceil().clamp(min=1).long()  # [N]

        # build a shared grid large enough for the max radius
        max_r = radii.max().item()
        ax = torch.arange(-max_r, max_r + 1, device=device, dtype=torch.float32)  # [D]
        yg, xg = torch.meshgrid(ax, ax, indexing='ij')  # [D, D]

        for n in range(len(gt_boxes)):
            r   = radii[n].item()
            sigma = (2 * r + 1) / 6.0
            # slice the shared grid to the actual radius for this box
            offset = max_r - r
            yg_n = yg[offset: offset + 2 * r + 1, offset: offset + 2 * r + 1]
            xg_n = xg[offset: offset + 2 * r + 1, offset: offset + 2 * r + 1]
            gaussian = torch.exp(-(xg_n ** 2 + yg_n ** 2) / (2 * sigma ** 2))
            gaussian[gaussian < 1e-6] = 0

            r0, c0 = row_i[n].item(), col_i[n].item()
            top    = max(0, r0 - r);  bottom = min(H, r0 + r + 1)
            left   = max(0, c0 - r);  right  = min(W, c0 + r + 1)
            g_top  = top  - (r0 - r); g_bottom = g_top + (bottom - top)
            g_left = left - (c0 - r); g_right  = g_left + (right - left)

            cls = gt_labels[n].item()
            gt_heatmap[cls, top:bottom, left:right] = torch.maximum(
                gt_heatmap[cls, top:bottom, left:right],
                gaussian[g_top:g_bottom, g_left:g_right])

        return gt_heatmap
