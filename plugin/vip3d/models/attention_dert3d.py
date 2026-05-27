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


# LiVip add: sparse deformable BEV cross-attention
@ATTENTION.register_module()
class LiDARBEVDeformCrossAtten(BaseModule):
    """Deformable cross-attention between object queries and LiDAR BEV feature map.

    Instead of attending over all H*W BEV cells (O(Q*H*W)), each query
    predicts P offset points around its reference location and samples
    only those P cells → O(Q*P).

    Args:
        embed_dims:     query/key/value dimensionality
        num_heads:      number of attention heads
        num_points:     P, number of sampling points per query
        bev_in_channels: input channels of the BEV feature map
        dropout:        dropout probability
        pc_range:       [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    def __init__(self, embed_dims=256, num_heads=8, num_points=4,
                 bev_in_channels=384, dropout=0.1, pc_range=None, init_cfg=None):
        super(LiDARBEVDeformCrossAtten, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.pc_range = pc_range
        self.dropout = nn.Dropout(dropout)

        # project BEV channels to embed_dims
        self.bev_proj = nn.Linear(bev_in_channels, embed_dims) \
            if bev_in_channels != embed_dims else nn.Identity()
        # predict (dx, dy) offsets in normalised BEV space for each of P points
        self.offset_pred = nn.Linear(embed_dims, num_points * 2)
        # predict scalar attention weight for each of P points
        self.attn_weight_pred = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        # zero-init offsets so they start centred on the reference point
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query, key=None, value=None, residual=None,
                query_pos=None, bev_feat=None, reference_points=None, **kwargs):
        """
        Args:
            query:            [num_q, B, embed_dims]
            bev_feat:         [B, bev_in_channels, H_bev, W_bev]
            reference_points: [B, num_q, 3]  normalised [0,1] (x,y,z in sigmoid space)
        Returns:
            [num_q, B, embed_dims]
        """
        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        num_q, B, D = query.shape
        P = self.num_points

        # ── project BEV feature map to embed_dims ──────────────────────────────
        _, C_bev, H_bev, W_bev = bev_feat.shape
        # [B, C_bev, H, W] → [B, H*W, embed_dims]
        bev_flat = bev_feat.flatten(2).permute(0, 2, 1)          # [B, H*W, C_bev]
        bev_flat = self.bev_proj(bev_flat)                        # [B, H*W, D]

        # ── predict per-query sampling offsets and weights ─────────────────────
        q_b = query.permute(1, 0, 2)  # [B, num_q, D]

        # reference in BEV normalised coords [B, num_q, 2] (x=col, y=row in [0,1])
        if reference_points is not None:
            ref_xy = reference_points[..., :2]  # [B, num_q, 2]
        else:
            ref_xy = torch.full((B, num_q, 2), 0.5,
                                device=query.device, dtype=query.dtype)

        offsets = self.offset_pred(q_b).view(B, num_q, P, 2)     # [B, num_q, P, 2]
        offsets = offsets.tanh() * 0.5                            # clamp to ±0.5

        # sampling locations in [-1, 1] for F.grid_sample
        # livip add - discard + offsets no need for training the offsets
        sample_xy = ref_xy.unsqueeze(2).expand(-1, -1, P, -1)  # [B, num_q, P, 2]
        sample_xy = sample_xy.clamp(0.0, 1.0)
        # grid_sample expects grid in [-1, 1]
        sample_grid = sample_xy * 2 - 1                           # [B, num_q, P, 2]

        # ── sample BEV features at predicted locations ─────────────────────────
        # bev_feat: [B, C_bev, H, W]
        # grid_sample wants [B, C, H_out, W_out] ← grid [B, H_out, W_out, 2]
        # reshape: treat num_q as H_out, P as W_out
        grid = sample_grid.view(B, num_q, P, 2)                   # [B, num_q, P, 2]
        # project raw bev_feat (not the flattened one) to embed_dims via bev_proj weight
        # We use bev_flat reshaped back for this: just sample from raw bev_feat then project
        sampled = F.grid_sample(
            bev_feat.float(),                                      # [B, C_bev, H, W]
            grid.float(),                                          # [B, num_q, P, 2]
            mode='bilinear', align_corners=False, padding_mode='zeros'
        )  # [B, C_bev, num_q, P]
        sampled = sampled.permute(0, 2, 3, 1)                     # [B, num_q, P, C_bev]
        sampled = self.bev_proj(sampled)                           # [B, num_q, P, D]

        # attention weights 
        attn_w = self.attn_weight_pred(q_b)                       # [B, num_q, P]
        attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)             # [B, num_q, P, 1]

        # weighted sum over P sampled features
        out = (attn_w * sampled).sum(dim=2)                       # [B, num_q, D]
        out = self.output_proj(out)                               # [B, num_q, D]
        out = out.permute(1, 0, 2)                                # [num_q, B, D]

        result = self.dropout(out) + inp_residual

        # ── TensorBoard: BEV attention visualisation ─────────────────────────
        from . import bev_vis as _bv
        _bv.visualize_lidar_bev_attn(
            bev_feat, reference_points,
            sample_xy,              # [B, N, P, 2] in [0,1]
            attn_w.squeeze(-1))     # [B, N, P]
        # ─────────────────────────────────────────────────────────────────────

        from . import bev_vis as _bv
        if _bv.DEBUG_PRINTS:
            delta = (result - inp_residual).norm(dim=-1).mean().item()
            base  = inp_residual.norm(dim=-1).mean().item()
            print(f"[LiDAR-BEV-Deform] residual_ratio={delta/(base+1e-8):.4f}  (>0.05=contributing)")
            offsets_mag = offsets.detach().abs().mean().item()
            print(f"[LiDAR-BEV-Deform] offset_mag={offsets_mag:.4f}  (0=no offset learning, >0=active)")
        return result



# LiVip add
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

        from . import bev_vis as _bv
        if _bv.DEBUG_PRINTS:
            depth_ok = (depth > eps).float().mean().item()
            cx_ok    = ((cx > 0) & (cx < 1)).float().mean().item()
            cy_ok    = ((cy > 0) & (cy < 1)).float().mean().item()
            print(f"[SMCA-proj] lidar2img.shape={lidar2img.shape}  img={img_h}x{img_w}")
            print(f"[SMCA-proj] depth_ok={depth_ok:.3f}  cx_ok={cx_ok:.3f}  cy_ok={cy_ok:.3f}")

        # transpose to [B, num_q, num_cam]
        return cx.permute(0, 2, 1), cy.permute(0, 2, 1), depth.permute(0, 2, 1), valid.permute(0, 2, 1), ref_3d
    
    def _compute_sigma(self, reference_points, ref_size, img_metas, H_feat, W_feat):
        """
        Project box width/length as axis-aligned extents into feature-map pixel space,
        compute circumscribed circle radius → sigma. Matches TransFusion's approach.
        Returns [B, num_q, num_cams]
        """
        lidar2img = reference_points.new_tensor(
            np.array([m['lidar2img'] for m in img_metas]))  # [B, num_cams, 4, 4]

        pc = self.pc_range
        ref = reference_points.clone()
        ref[..., 0] = ref[..., 0] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1] = ref[..., 1] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2] = ref[..., 2] * (pc[5] - pc[2]) + pc[2]

        B, num_q = ref.shape[:2]
        num_cams = lidar2img.shape[1]

        wl = ref_size[..., :2].clamp(-3, 3).exp()  # [B, num_q, 2] metric w, l; clamp prevents unbounded growth from unclamped log-size accumulation across frames  #DEBUG-FIX
        hw = wl[..., 0:1] / 2             # half-width
        hl = wl[..., 1:2] / 2             # half-length
        z  = torch.zeros_like(hw)

        # 5 points: center + 4 axis-aligned half-extents [B, num_q, 5, 4]
        pts = torch.cat([
            torch.cat([ref,                                    torch.ones_like(ref[..., :1])], -1).unsqueeze(2),
            torch.cat([ref + torch.cat([ hw, z, z], -1),      torch.ones_like(ref[..., :1])], -1).unsqueeze(2),
            torch.cat([ref + torch.cat([-hw, z, z], -1),      torch.ones_like(ref[..., :1])], -1).unsqueeze(2),
            torch.cat([ref + torch.cat([z,  hl, z], -1),      torch.ones_like(ref[..., :1])], -1).unsqueeze(2),
            torch.cat([ref + torch.cat([z, -hl, z], -1),      torch.ones_like(ref[..., :1])], -1).unsqueeze(2),
        ], dim=2)  # [B, num_q, 5, 4]

        # project into each camera
        pts_e = pts.view(B, 1, num_q, 5, 4, 1).expand(-1, num_cams, -1, -1, -1, -1)
        l2i   = lidar2img.view(B, num_cams, 1, 1, 4, 4).expand(-1, -1, num_q, 5, -1, -1)
        pts_cam = torch.matmul(l2i, pts_e).squeeze(-1)  # [B, num_cams, num_q, 5, 4]

        # ── OLD (broken) sigma — behind-camera corners inflate sigma to 4M ──────
        # depth = pts_cam[..., 2].clamp(min=1e-5)
        # u = pts_cam[..., 0] / depth  # [B, num_cams, num_q, 5] in pixels
        # v = pts_cam[..., 1] / depth
        # img_h = img_metas[0]['img_shape'][0][0][0]
        # img_w = img_metas[0]['img_shape'][0][0][1]
        # u_feat = u / img_w * W_feat  # [B, num_cams, num_q, 5]
        # v_feat = v / img_h * H_feat
        # u_range = u_feat.amax(dim=-1) - u_feat.amin(dim=-1)  # [B, num_cams, num_q]
        # v_range = v_feat.amax(dim=-1) - v_feat.amin(dim=-1)
        # radius = torch.ceil(torch.stack([u_range, v_range], dim=-1).norm(p=2, dim=-1) / 2)
        # sigma  = (radius * 2 + 1) / 6.0
        # sigma  = sigma.clamp(min=1.0)  # at least 1 feature pixel

        # ── NEW (fixed) sigma — front-mask only, clamp max=50 ───────────────
        front = pts_cam[..., 2] > 0                           # [B, num_cams, num_q, 5]
        depth = pts_cam[..., 2].clamp(min=1e-5)
        u = pts_cam[..., 0] / depth                           # [B, num_cams, num_q, 5]
        v = pts_cam[..., 1] / depth
        img_h = img_metas[0]['img_shape'][0][0][0]
        img_w = img_metas[0]['img_shape'][0][0][1]
        u_feat = u / img_w * W_feat
        v_feat = v / img_h * H_feat
        # only use corners in front of the camera; behind-camera corners divided
        # by clamped near-zero depth produce millions-of-pixels coordinates
        INF = 1e6
        u_for_max = torch.where(front, u_feat, torch.full_like(u_feat, -INF))
        u_for_min = torch.where(front, u_feat, torch.full_like(u_feat,  INF))
        v_for_max = torch.where(front, v_feat, torch.full_like(v_feat, -INF))
        v_for_min = torch.where(front, v_feat, torch.full_like(v_feat,  INF))
        u_range = (u_for_max.amax(dim=-1) - u_for_min.amin(dim=-1)).clamp(0)
        v_range = (v_for_max.amax(dim=-1) - v_for_min.amin(dim=-1)).clamp(0)
        # fall back to small default where no corner is in front of that camera
        any_front = front.any(dim=-1)                         # [B, num_cams, num_q]
        u_range = torch.where(any_front, u_range, torch.full_like(u_range, 4.0))
        v_range = torch.where(any_front, v_range, torch.full_like(v_range, 4.0))
        radius = torch.ceil(torch.stack([u_range, v_range], dim=-1).norm(p=2, dim=-1) / 2)
        sigma  = (radius * 2 + 1) / 6.0
        sigma  = sigma.clamp(min=1.0, max=50.0)               # guard against residual outliers

        return sigma.permute(0, 2, 1)  # [B, num_q, num_cams]


    def _gaussian_mask(self, cx, cy, sigma, H, W, device, dtype):
        """
        Args:
            cx, cy: [B, num_q, num_cams]  normalised [0,1]
            sigma:  [B, num_q, num_cams]  in feature-map pixel units
        Returns:
            [B, num_q, num_cams, H*W]
        """
        gy = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        gx = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_x = grid_x.flatten().view(1, 1, 1, -1)  # [1, 1, 1, H*W]
        grid_y = grid_y.flatten().view(1, 1, 1, -1)

        # convert normalised cx/cy to feature-map pixel coords
        cx_pix = cx.unsqueeze(-1) * W   # [B, num_q, num_cams, 1]
        cy_pix = cy.unsqueeze(-1) * H
        sigma  = sigma.unsqueeze(-1)    # [B, num_q, num_cams, 1]

        gauss = torch.exp(
            -0.5 * ((grid_x - cx_pix) ** 2 + (grid_y - cy_pix) ** 2) / sigma ** 2
        )
        return gauss  # [B, num_q, num_cams, H*W]

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
        from . import bev_vis as _bv
        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        num_q, B, _ = query.shape

        # ── fuse all 4 FPN levels → [B, num_cams, embed_dims, H0, W0]
        img_feat = self._fuse_fpn_levels(value)
        H0, W0 = img_feat.shape[-2], img_feat.shape[-1]

        # ── project reference points to camera planes
        cx, cy, depth, valid, ref_3d = self._project_to_cameras(reference_points, img_metas)
        # cx, cy, valid: [B, num_q, num_cams]
        if _bv.DEBUG_PRINTS:
            per_cam = valid.float().mean(dim=[0, 1]).tolist()
            print(f"[SMCA] per-cam valid: {[f'{v:.3f}' for v in per_cam]}")
            print(f"[SMCA] ref_pts range: x=[{reference_points[...,0].min():.3f},{reference_points[...,0].max():.3f}] y=[{reference_points[...,1].min():.3f},{reference_points[...,1].max():.3f}] z=[{reference_points[...,2].min():.3f},{reference_points[...,2].max():.3f}]")

        # ── build Gaussian spatial masks [B, num_q, num_cams, H0*W0]
        sigma = self._compute_sigma(reference_points, ref_size, img_metas, H0, W0)
        if _bv.DEBUG_PRINTS:
            print(f"[SMCA] sigma mean={sigma.mean():.2f}  max={sigma.max():.2f}  (expect 2-15px)")
        gauss = self._gaussian_mask(cx, cy, sigma, H0, W0, query.device, query.dtype)

        # ── SMCA cross-attention per camera, then average
        # query: [num_q, B, D] → [B, num_q, D]
        q = self.q_proj(query.permute(1, 0, 2))  # [B, num_q, D]

        accum = torch.zeros_like(q)
        count = torch.zeros(B, num_q, 1, device=query.device, dtype=query.dtype)
        if _bv.DEBUG_PRINTS:
            print(f"[SMCA] queries_visible_any_cam={valid.any(dim=-1).float().mean():.3f}")
        _do_vis = (_bv.ENABLED and _bv._step % _bv.VIS_INTERVAL == 0
                   and _bv._top_q_idx is not None)
        _attn_collect = []  # per-camera attn for top-K queries (vis only)

        # store raw Gaussian mask for top-K queries so we can visualize sigma
        if _do_vis:
            idx = _bv._top_q_idx.to(gauss.device)
            gauss_topk = gauss[0, idx, :, :]  # [K, num_cams, H0*W0]
            _bv.store_smca_gauss(
                [gauss_topk[:, ci, :] for ci in range(self.num_cams)], H0, W0)

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

            if cam_idx == 0 and _bv.DEBUG_PRINTS:
                max_ent = torch.log(torch.tensor(attn.shape[-1], dtype=attn.dtype, device=attn.device))
                entropy = -(attn * attn.clamp(1e-10).log()).sum(-1).mean() / max_ent
                print(f"[SMCA cam0] attn_entropy_norm={entropy.item():.3f}  (0=focused, 1=uniform)")

            if _do_vis:
                idx = _bv._top_q_idx.to(attn.device)
                _attn_collect.append(attn[0, idx, :])  # [K, H0*W0]

            # weighted sum of values [B, num_q, D]
            out = torch.bmm(attn, v)

            # mask invalid projections
            cam_valid = valid[:, :, cam_idx].float().unsqueeze(-1)  # [B, num_q, 1]
            accum = accum + out * cam_valid
            count = count + cam_valid

        if _do_vis and _attn_collect:
            _bv.store_smca_attn(_attn_collect, H0, W0)

        # average over valid cameras
        accum = accum / count.clamp(min=1.0)
        accum = self.output_proj(accum)  # [B, num_q, D]

        # position encoding from 3D ref points
        pos_feat = self.position_encoder(
            inverse_sigmoid(ref_3d))  # [B, num_q, D]

        # back to [num_q, B, D]
        out = accum.permute(1, 0, 2)
        pos_feat = pos_feat.permute(1, 0, 2)
        
        # after Layer 0 puts the query out of scope of all cameras, return mask to indicate which queries have no valid camera view
        no_cam_mask = (count.squeeze(-1) == 0)  # [B, num_q]  True
        if _bv.DEBUG_PRINTS:
            print(f"[SMCA] no_cam_mask_rate={no_cam_mask.float().mean():.3f}  (0=all visible, 1=none)")
        return self.dropout(out) + inp_residual + pos_feat, no_cam_mask

