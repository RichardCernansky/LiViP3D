import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


@ATTENTION.register_module()
class Detr3DCamRadarCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 radar_dims=3,
                 radar_topk=8,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCamRadarCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.radar_dims = radar_dims

        self.attention_weights_radar = nn.Linear(embed_dims, radar_topk)
        self.radar_topk = radar_topk

        self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        self.radar_output_proj = nn.Linear(self.radar_dims, self.radar_dims)

        self.img_radar_fusion = nn.Sequential(
            nn.Linear(embed_dims + radar_dims, embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights_radar, val=0., bias=0.)
        xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.radar_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.img_radar_fusion, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                ref_size=None,
                spatial_shapes=None,
                level_start_index=None,
                radar_feats=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space. 
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        # [bs, embed_dim, num_query]
        output = output.sum(-1).sum(-1).sum(-1)
        # chaneg to [num_query, bs, embed_dims]
        output = output.permute(2, 0, 1)

        output = self.img_output_proj(output)

        radar_feats, radar_mask = radar_feats[:, :, :-1], radar_feats[:, :, -1]

        radar_xy = radar_feats[:, :, :2]
        ref_xy = reference_points[:, :, :2]
        radar_feats = radar_feats[:, :, 2:]

        pad_xy = torch.ones_like(radar_xy) * 1000.0

        radar_xy = radar_xy + (1.0 - radar_mask.unsqueeze(dim=-1).type(torch.float)) * (pad_xy)

        # [B, num_query, M]
        ref_radar_dist = -1.0 * torch.cdist(ref_xy, radar_xy)

        # [B, num_query, topk]
        _value, indices = torch.topk(ref_radar_dist, self.radar_topk)

        # [B, num_query, M]
        radar_mask = radar_mask.unsqueeze(dim=1).repeat(1, num_query, 1)

        # [B, num_query, topk]
        top_mask = torch.gather(radar_mask, 2, indices)

        # [B, num_query, M, radar_dim]
        radar_feats = radar_feats.unsqueeze(dim=1).repeat(1, num_query, 1, 1)
        radar_dim = radar_feats.size(-1)
        # [B, num_query, topk, radar_dim]
        indices_pad = indices.unsqueeze(dim=-1).repeat(1, 1, 1, radar_dim)

        # [B, num_query, topk, radar_dim]
        radar_feats_topk = torch.gather(
            radar_feats, dim=2, index=indices_pad, sparse_grad=False)

        attention_weights_radar = self.attention_weights_radar(query).view(
            bs, num_query, self.radar_topk)

        # [B, num_query, topk]
        attention_weights_radar = attention_weights_radar.sigmoid() * top_mask
        # [B, num_query, topk, radar_dim]
        radar_out = radar_feats_topk * attention_weights_radar.unsqueeze(dim=-1)
        # [bs, num_query, radar_dim]
        radar_out = radar_out.sum(dim=2)

        # change to (num_query, bs, embed_dims)
        radar_out = radar_out.permute(1, 0, 2)

        radar_out = self.radar_output_proj(radar_out)

        output = torch.cat((output, radar_out), dim=-1)
        output = self.img_radar_fusion(output)

        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > -1.0)
            & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask


@ATTENTION.register_module()
class Detr3DCrossAttenPetrFeature(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAttenPetrFeature, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        inp_residual = query

        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        if True:
            value = value[0]

            query = query.transpose(0, 1)
            value = value.transpose(0, 1)
            output = self.attn(query=query, key=value, value=value)[0]

            reference_points_3d = reference_points.clone()
            # petr_feature = value[0]
            # assert len(petr_feature.shape) == 2
            # petr_feature = petr_feature.view(-1, 6, 256) # num_query, num_cam, C
            # # petr_feature = petr_feature[:50, :, :]
            # petr_feature = petr_feature.permute(2, 0, 1).contiguous() # C, num_query, num_cam
            # s = petr_feature.shape
            # petr_feature = petr_feature.view(1, s[0], s[1], s[2], 1, 1)
            # output = petr_feature

        output = self.output_proj(output)  # (num_query, bs, embed_dims)

        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

# LiVip3D add
@ATTENTION.register_module()
class LiDARBEVCrossAtten(BaseModule):
    """Cross-attention between object queries and LiDAR BEV feature map."""

    def __init__(self, embed_dims=256, num_heads=8, bev_in_channels=384,
                 dropout=0.1, init_cfg=None):
        super(LiDARBEVCrossAtten, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.dropout = nn.Dropout(dropout)
        self.bev_proj = nn.Linear(bev_in_channels, embed_dims) \
            if bev_in_channels != embed_dims else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def _make_bev_pos_embed(self, H, W, device, dtype):
        """2D sine positional encoding → [H*W, 1, embed_dims]."""
        y = torch.arange(H, device=device, dtype=dtype) / H
        x = torch.arange(W, device=device, dtype=dtype) / W
        # [H, W]
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        dim = self.embed_dims // 4
        inv_freq = 1.0 / (10000 ** (torch.arange(dim, device=device, dtype=dtype) / dim))
        # [H*W, dim]
        pe_x = torch.outer(grid_x.flatten(), inv_freq)
        pe_y = torch.outer(grid_y.flatten(), inv_freq)
        # [H*W, embed_dims]
        pe = torch.cat([pe_x.sin(), pe_x.cos(), pe_y.sin(), pe_y.cos()], dim=-1)
        return pe.unsqueeze(1)  # [H*W, 1, embed_dims]

    def forward(self, query, key=None, value=None, residual=None,
                query_pos=None, bev_feat=None, **kwargs):
        """
        Args:
            query:    [num_q, B, embed_dims]
            bev_feat: [B, bev_in_channels, H, W]
        Returns:
            [num_q, B, embed_dims]
        """
        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        B, C, H, W = bev_feat.shape
        # [B, embed_dims, H, W] → [H*W, B, embed_dims]
        bev = self.bev_proj(bev_feat.flatten(2).permute(2, 0, 1))
        pos = self._make_bev_pos_embed(H, W, bev_feat.device, bev_feat.dtype)
        bev = bev + pos

        out, _ = self.attn(query=query, key=bev, value=bev)
        out = self.output_proj(out)
        return self.dropout(out) + inp_residual


@ATTENTION.register_module()
class SMCACrossAtten(BaseModule):
    """Spatially Modulated Cross-Attention (SMCA) over multi-level camera FPN features."""

    def __init__(self, embed_dims=256, num_heads=8, num_cams=3, num_levels=4,
                 pc_range=None, dropout=0.1, init_cfg=None):
        super(SMCACrossAtten, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.scale = (embed_dims // num_heads) ** -0.5
        self.dropout = nn.Dropout(dropout)

        # fuse all 4 FPN levels into one [B, N, 256, H0, W0] representation
        self.fusion_proj = nn.Linear(num_levels * embed_dims, embed_dims)

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

    def _fuse_fpn_levels(self, mlvl_feats):
        """Upsample levels 1-3 to level-0 size, concat, project.
        Args:
            mlvl_feats: list of [B, N, C, H_l, W_l]
        Returns:
            [B, N, embed_dims, H0, W0]
        """
        B, N, C, H0, W0 = mlvl_feats[0].shape
        upsampled = [mlvl_feats[0]]
        for feat in mlvl_feats[1:]:
            # [B*N, C, H_l, W_l] → upsample → [B, N, C, H0, W0]
            f = feat.view(B * N, C, feat.shape[-2], feat.shape[-1])
            f = F.interpolate(f, size=(H0, W0), mode='bilinear', align_corners=False)
            upsampled.append(f.view(B, N, C, H0, W0))
        # concat along channel dim → [B, N, num_levels*C, H0, W0]
        fused = torch.cat(upsampled, dim=2)
        # [B, N, H0, W0, num_levels*C] → linear → [B, N, H0, W0, embed_dims]
        fused = fused.permute(0, 1, 3, 4, 2)
        fused = self.fusion_proj(fused)
        # → [B, N, embed_dims, H0, W0]
        return fused.permute(0, 1, 4, 2, 3)

    def _project_to_cameras(self, reference_points, img_metas):
        """Project 3D reference points into each camera image plane.
        Returns:
            cx, cy: [B, num_q, num_cams]  normalised [0,1]
            depth:  [B, num_q, num_cams]
            valid:  [B, num_q, num_cams]  bool
            ref_3d: [B, num_q, 3]         3D coords in metric space
        """
        lidar2img = []
        for meta in img_metas:
            lidar2img.append(meta['lidar2img'])
        lidar2img = reference_points.new_tensor(np.asarray(lidar2img))  # [B, N, 4, 4]

        ref = reference_points.clone()
        ref[..., 0] = ref[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        ref[..., 1] = ref[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        ref[..., 2] = ref[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        ref_3d = ref.clone()

        B, num_q = ref.shape[:2]
        num_cam = lidar2img.shape[1]
        ref_h = torch.cat([ref, torch.ones_like(ref[..., :1])], dim=-1)  # [B, num_q, 4]
        ref_h = ref_h.view(B, 1, num_q, 4, 1).expand(-1, num_cam, -1, -1, -1)
        l2i = lidar2img.view(B, num_cam, 1, 4, 4).expand(-1, -1, num_q, -1, -1)
        pts_cam = torch.matmul(l2i, ref_h).squeeze(-1)  # [B, num_cam, num_q, 4]

        depth = pts_cam[..., 2]  # [B, num_cam, num_q]
        eps = 1e-5
        pts2d = pts_cam[..., :2] / torch.clamp(depth.unsqueeze(-1), min=eps)

        img_h = img_metas[0]['img_shape'][0][0][0]
        img_w = img_metas[0]['img_shape'][0][0][1]
        cx = pts2d[..., 0] / img_w  # [B, num_cam, num_q]
        cy = pts2d[..., 1] / img_h

        valid = (depth > eps) & (cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)

        # transpose to [B, num_q, num_cam]
        return cx.permute(0, 2, 1), cy.permute(0, 2, 1), \
               depth.permute(0, 2, 1), valid.permute(0, 2, 1), ref_3d

    def _gaussian_mask(self, cx, cy, ref_size, H, W, device, dtype):
        """Build per-query Gaussian spatial mask on the feature map.
        Args:
            cx, cy:   [B, num_q, num_cams]  normalised centre
            ref_size: [B, num_q, 3]         wlh in log space
        Returns:
            [B, num_q, num_cams, H*W]
        """
        B, num_q, num_cams = cx.shape
        wl = ref_size[..., :2].exp()  # [B, num_q, 2]  metric w, l
        # rough sigma: box footprint projected to feature-map pixels
        sigma_w = (wl[..., 0] / 20).clamp(0.02, 0.3)  # [B, num_q]
        sigma_h = (wl[..., 1] / 20).clamp(0.02, 0.3)

        # pixel grid [H, W]
        gy = torch.linspace(0, 1, H, device=device, dtype=dtype)
        gx = torch.linspace(0, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_x = grid_x.flatten().view(1, 1, 1, -1)  # [1,1,1,H*W]
        grid_y = grid_y.flatten().view(1, 1, 1, -1)

        cx = cx.unsqueeze(-1)  # [B, num_q, num_cams, 1]
        cy = cy.unsqueeze(-1)
        sigma_w = sigma_w.unsqueeze(-1).unsqueeze(-1)  # [B, num_q, 1, 1]
        sigma_h = sigma_h.unsqueeze(-1).unsqueeze(-1)

        gauss = torch.exp(
            -0.5 * ((grid_x - cx) ** 2 / sigma_w ** 2
                  + (grid_y - cy) ** 2 / sigma_h ** 2)
        )  # [B, num_q, num_cams, H*W]
        return gauss

    def forward(self, query, key=None, value=None, residual=None,
                query_pos=None, reference_points=None, ref_size=None,
                img_metas=None, **kwargs):
        """
        Args:
            query:            [num_q, B, embed_dims]
            value:            list of [B, N, C, H_l, W_l]  (4 FPN levels)
            reference_points: [B, num_q, 3]  normalised sigmoid space
            ref_size:         [B, num_q, 3]  wlh in log space
        Returns:
            [num_q, B, embed_dims]
        """
        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        num_q, B, _ = query.shape

        # ── fuse all 4 FPN levels → [B, num_cams, embed_dims, H0, W0]
        img_feat = self._fuse_fpn_levels(value)
        H0, W0 = img_feat.shape[-2], img_feat.shape[-1]

        # ── project reference points to camera planes
        cx, cy, depth, valid, ref_3d = self._project_to_cameras(
            reference_points, img_metas)
        # cx, cy, valid: [B, num_q, num_cams]

        # ── build Gaussian spatial masks [B, num_q, num_cams, H0*W0]
        gauss = self._gaussian_mask(cx, cy, ref_size, H0, W0,
                                    query.device, query.dtype)

        # ── SMCA cross-attention per camera, then average
        # query: [num_q, B, D] → [B, num_q, D]
        q = self.q_proj(query.permute(1, 0, 2))  # [B, num_q, D]

        accum = torch.zeros_like(q)
        count = torch.zeros(B, num_q, 1, device=query.device, dtype=query.dtype)

        for cam_idx in range(self.num_cams):
            cam_feat = img_feat[:, cam_idx]  # [B, D, H0, W0]
            # flatten spatial → [B, H0*W0, D]
            kv_flat = cam_feat.flatten(2).permute(0, 2, 1)
            k = self.k_proj(kv_flat)  # [B, H0*W0, D]
            v = self.v_proj(kv_flat)

            # raw attention logits [B, num_q, H0*W0]
            attn = torch.bmm(q, k.transpose(1, 2)) * self.scale

            # add log-Gaussian mask (in log domain = additive bias)
            g = gauss[:, :, cam_idx, :]  # [B, num_q, H0*W0]
            attn = attn + torch.log(g.clamp(min=1e-6))

            attn = attn.softmax(dim=-1)  # [B, num_q, H0*W0]

            # weighted sum of values [B, num_q, D]
            out = torch.bmm(attn, v)

            # mask invalid projections
            cam_valid = valid[:, :, cam_idx].float().unsqueeze(-1)  # [B, num_q, 1]
            accum = accum + out * cam_valid
            count = count + cam_valid

        # average over valid cameras
        accum = accum / count.clamp(min=1.0)
        accum = self.output_proj(accum)  # [B, num_q, D]

        # position encoding from 3D ref points
        pos_feat = self.position_encoder(
            inverse_sigmoid(ref_3d))  # [B, num_q, D]

        # back to [num_q, B, D]
        out = accum.permute(1, 0, 2)
        pos_feat = pos_feat.permute(1, 0, 2)

        return self.dropout(out) + inp_residual + pos_feat