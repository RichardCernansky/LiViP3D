import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.builder import TRANSFORMER
from .apr import build_apr


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


@TRANSFORMER.register_module()
class Detr3DCamTransformerPlus(BaseModule):
    """Implements the DeformableDETR transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 decoder=None,
                 reference_points_aug=False,
                 **kwargs):
        super(Detr3DCamTransformerPlus, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

        # self.cam_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        # move ref points to tracker
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # normal_(self.level_embeds)
        # normal_(self.cam_embeds)

    def forward(self,
                mlvl_feats,
                query_embed,
                reference_points,
                reg_branches=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, 2*embed_dim], can be splitted into
                query_feat and query_positional_encoding.
            reference_points (Tensor): The corresponding 3d ref points
                for the query with shape (num_query, 3)
                value is in inverse sigmoid space
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, 3)
                
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(dim=0).expand(bs, -1, -1)

        if self.training and self.reference_points_aug:
            reference_points = reference_points + torch.randn_like(reference_points)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER.register_module()
class Detr3DCamTrackTransformer(BaseModule):
    """Implements the DeformableDETR transformer. 
        Specially designed for track: keep xyz trajectory, and 
        kep bbox size(which should be consisten across frames)

    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 decoder=None,
                 reference_points_aug=False,
                 **kwargs):
        super(Detr3DCamTrackTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

        # self.cam_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        # move ref points to tracker
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                mlvl_feats,
                query_embed,
                reference_points,
                ref_size,
                reg_branches=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, 2*embed_dim], can be splitted into
                query_feat and query_positional_encoding.
            reference_points (Tensor): The corresponding 3d ref points
                for the query with shape (num_query, 3)
                value is in inverse sigmoid space
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (num_query, 3)
                value in log space. 
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, 3)
                
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(dim=0).expand(bs, -1, -1)
        ref_size = ref_size.unsqueeze(dim=0).expand(bs, -1, -1)

        if self.training and self.reference_points_aug:
            reference_points = reference_points + torch.randn_like(reference_points)
        reference_points = reference_points.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references, inter_box_sizes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            ref_size=ref_size,
            **kwargs)

        return inter_states, inter_references, inter_box_sizes


# livip add - 
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DCamTrackPlusTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=True, **kwargs):

        super(Detr3DCamTrackPlusTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                ref_size=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The 3d reference points
                associated with each query. shape (num_query, 3).
                value is in inevrse sigmoid space
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space. 
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_box_sizes = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                ref_size=ref_size,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                ref_pts_update = torch.cat(
                    [
                        tmp[..., :2],
                        tmp[..., 4:5],
                    ], dim=-1
                )
                ref_size_update = torch.cat(
                    [
                        tmp[..., 2:4],
                        tmp[..., 5:6]
                    ], dim=-1
                )
                assert reference_points.shape[-1] == 3

                new_reference_points = ref_pts_update + \
                                       inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                # add in log space
                # ref_size = (ref_size.exp() + ref_size_update.exp()).log()
                ref_size = ref_size + ref_size_update
                if lid > 0:
                    ref_size = ref_size.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_box_sizes.append(ref_size)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), \
                torch.stack(intermediate_box_sizes)

        return output, reference_points, ref_size

@TRANSFORMER.register_module()
class TransFusionTransformer(BaseModule):
    """2-layer TransFusion transformer: Layer0=LiDAR BEV, Layer1=SMCA camera."""

    def __init__(self, num_feature_levels=4, num_cams=3, decoder=None, **kwargs):
        super(TransFusionTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mlvl_feats, query_embed, reference_points, ref_size,
                reg_branches=None, bev_feat=None, **kwargs):
        assert query_embed is not None
        bs = mlvl_feats[0].size(0) if mlvl_feats is not None else bev_feat.size(0)  # lidar-only: no camera feats
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        ref_size = ref_size.unsqueeze(0).expand(bs, -1, -1)

        reference_points = reference_points.sigmoid()

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references, inter_box_sizes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            ref_size=ref_size,
            bev_feat=bev_feat,
            **kwargs)

        return inter_states, inter_references, inter_box_sizes


# LiVip add: custom decoder with 2 separate branches for ref points and box sizes, and special handling of invisible queries
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransFusionTransformerDecoder(BaseModule):
    """Custom 2-layer decoder: LiDAR BEV cross-attn + SMCA camera cross-attn."""

    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=512, dropout=0.1,
                 lidar_bev_attn=None, smca_attn=None, use_smca=False,
                 apr_cfg=None, pc_range=None, **kwargs):
        super(TransFusionTransformerDecoder, self).__init__()
        from mmcv.cnn.bricks.registry import ATTENTION as ATT_REG
        self.embed_dims = embed_dims
        self.num_layers = 2
        self.use_smca = use_smca

        # ── Layer 0: self-attn + LiDAR BEV cross-attn + FFN
        self.sa0 = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.ca0 = ATT_REG.build(lidar_bev_attn)
        self.ff0 = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims), nn.Dropout(dropout))
        self.n0 = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

        # ── Layer 1: self-attn + SMCA camera cross-attn + FFN
        self.sa1 = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.ca1 = ATT_REG.build(smca_attn)
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims), nn.Dropout(dropout))
        self.n1 = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

        # ── APR: owns its own module + its own rolling history of past BEV/ego-pose.
        # Called once before layer 0 and, when SMCA is active, again before layer 1
        # (using layer 0's already-refined position and content). vip3d.py hands in
        # only the few per-frame values it alone has access to (track_instances-level
        # state and the dataloader's ego pose) — everything else lives here.
        self.apr = build_apr(apr_cfg, embed_dims)
        self.pc_range = pc_range
        self.history_bev_feats = []
        self.history_ego_r = []
        self.history_ego_t = []
        # last-computed APR outputs, one slot per call site this frame, keyed 'l0'/'l1' —
        # for the APR loss (computed later in ViP3D, once GT is in scope) to read back.
        # Each value is {'info': <apr forward()'s info dict>, 'alive_mask': [num_q] bool}.
        self.last_apr = {}

    def reset_apr_history(self):
        self.history_bev_feats = []
        self.history_ego_r = []
        self.history_ego_t = []
        self.last_apr = {}

    def _push_apr_history(self, bev_feat, cur_ego_r, cur_ego_t):
        if self.apr is None or bev_feat is None or cur_ego_r is None:
            return
        self.history_bev_feats.append(bev_feat.detach())
        self.history_ego_r.append(cur_ego_r.detach())
        self.history_ego_t.append(cur_ego_t.detach())
        max_len = self.apr.history_len
        self.history_bev_feats = self.history_bev_feats[-max_len:]
        self.history_ego_r = self.history_ego_r[-max_len:]
        self.history_ego_t = self.history_ego_t[-max_len:]

    def _run_apr(self, q, reference_points, alive_mask, velocity,
                 cur_ego_r, cur_ego_t, time_delta, tag):
        """Refine q's content using APR, if enabled and there's something to refine.
        q: [num_q, B, D] (B always 1 here). reference_points: [B, num_q, 3].
        Only alive tracks are touched; everything else in q passes through
        unchanged, and reference_points itself is never modified.
        tag: 'l0' or 'l1' — which call site this is, so both can be told apart later.
        """
        if self.apr is None or alive_mask is None or not alive_mask.any() \
                or len(self.history_bev_feats) == 0:
            self.last_apr.pop(tag, None)
            return q
        q_flat = q[:, 0, :]              # [num_q, D]
        ref_flat = reference_points[0]   # [num_q, 3]
        refined, info = self.apr(
            q_flat[alive_mask], ref_flat[alive_mask], velocity[alive_mask],
            self.history_bev_feats, self.history_ego_r, self.history_ego_t,
            cur_ego_r, cur_ego_t, time_delta, self.pc_range)
        self.last_apr[tag] = {'info': info, 'alive_mask': alive_mask}
        q_flat = q_flat.clone()
        q_flat[alive_mask] = refined
        return q_flat.unsqueeze(1)       # [num_q, 1, D]

    def _update_ref(self, reg_branch, output, reference_points, ref_size, detach_size):
        """Run regression head and update reference points and box sizes."""
        tmp = reg_branch(output.permute(1, 0, 2))  # [B, num_q, code_size]
        ref_pts_update = torch.cat([tmp[..., :2], tmp[..., 4:5]], dim=-1)
        ref_size_update = torch.cat([tmp[..., 2:4], tmp[..., 5:6]], dim=-1)

        new_ref = (ref_pts_update + inverse_sigmoid(reference_points)).sigmoid()
        reference_points = new_ref.detach()

        ref_size = ref_size + ref_size_update
        if detach_size:
            ref_size = ref_size.detach()

        return reference_points, ref_size

    def forward(self, query, key=None, value=None, query_pos=None,
                reference_points=None, reg_branches=None, ref_size=None,
                bev_feat=None, alive_mask=None, velocity=None,
                cur_ego_r=None, cur_ego_t=None, time_delta=None, **kwargs):
        """
        Args:
            query:            [num_q, B, D]
            value:            list of [B, N, C, H, W]  (camera FPN levels)
            reference_points: [B, num_q, 3]  normalised sigmoid
            ref_size:         [B, num_q, 3]  wlh log space
            bev_feat:         [B, C_bev, H_bev, W_bev]
            alive_mask, velocity, cur_ego_r, cur_ego_t, time_delta:
                per-frame values only ViP3D/track_instances has, needed by APR
                (self.apr / self.history_*, both owned by this module — see
                _run_apr / _push_apr_history above).
        """
        intermediate = []
        inter_ref = []
        inter_size = []

        # Layer 0: LiDAR BEV cross-attention
        q = self._run_apr(query, reference_points, alive_mask, velocity,
                           cur_ego_r, cur_ego_t, time_delta, tag='l0')
        # self-attention
        q2, _ = self.sa0(q + query_pos, q + query_pos, q)
        q = self.n0[0](q + q2)
        # LiDAR BEV cross-attention
        q2 = self.ca0(q, query_pos=query_pos, bev_feat=bev_feat,
                      reference_points=reference_points, **kwargs)
        q = self.n0[1](q + q2)
        # FFN
        q = self.n0[2](q + self.ff0(q.permute(1, 0, 2)).permute(1, 0, 2))

        if reg_branches is not None:
            reference_points, ref_size = self._update_ref(
                reg_branches[0], q, reference_points, ref_size, detach_size=True)

        intermediate.append(q)
        inter_ref.append(reference_points)
        inter_size.append(ref_size)

        # Layer 1 
        if not self.use_smca:
            # Stage 1: skip SMCA, duplicate Layer 0 output so head shape is unchanged
            if reg_branches is not None:
                reference_points, ref_size = self._update_ref(
                    reg_branches[1], q, reference_points, ref_size, detach_size=True)
            intermediate.append(q)
            inter_ref.append(reference_points)
            inter_size.append(ref_size)
            self._push_apr_history(bev_feat, cur_ego_r, cur_ego_t)
            return (torch.stack(intermediate),
                    torch.stack(inter_ref),
                    torch.stack(inter_size))

        # second APR pass: layer 0 has already refined both position (reference_points,
        # via _update_ref above) and content (q) — this call sees the improved version
        # of both, before layer 1 ever samples the cameras.
        q = self._run_apr(q, reference_points, alive_mask, velocity,
                           cur_ego_r, cur_ego_t, time_delta, tag='l1')

        q_layer0 = q.clone()  # pure layer 0 (+ 2nd APR pass) output, before layer 1

        q2, _ = self.sa1(q + query_pos, q + query_pos, q)
        q = self.n1[0](q + q2)

        ca1_out, no_cam_mask = self.ca1(q, value=value, query_pos=query_pos,
                                        reference_points=reference_points,
                                        ref_size=ref_size, **kwargs)
        q = self.n1[1](q + ca1_out)
        q = self.n1[2](q + self.ff1(q.permute(1, 0, 2)).permute(1, 0, 2))

        # invisible queries → fall back to pure layer 0 output
        mask = no_cam_mask.permute(1, 0).unsqueeze(-1)  # [num_q, B, 1]
        # take q_layer0 for queries with no valid camera view (mask=1), otherwise take q after SMCA cross-attn and FFN
        q = torch.where(mask, q_layer0, q)
        from . import bev_vis as _bv
        if _bv.DEBUG_PRINTS:
            layer1_delta = (q - q_layer0).norm(dim=-1).mean().item()
            print(f"[Decoder] no_cam_mask={mask.float().mean():.3f}  layer1_delta_norm={layer1_delta:.4f}  (>0.1=SMCA contributing)")

        if reg_branches is not None:
            reference_points, ref_size = self._update_ref(
                reg_branches[1], q, reference_points, ref_size, detach_size=True)

        intermediate.append(q)
        inter_ref.append(reference_points)
        inter_size.append(ref_size)

        self._push_apr_history(bev_feat, cur_ego_r, cur_ego_t)
        return (torch.stack(intermediate),
                torch.stack(inter_ref),
                torch.stack(inter_size))