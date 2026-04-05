Change 1 — attention_dert3d.py (append at end of file)

@ATTENTION.register_module()
class LiDARBEVCrossAtten(BaseModule):
    """Cross-attention from object queries to LiDAR BEV feature map.
    TransFusion decoder layer 0.
    """
    def __init__(self, embed_dims=256, num_heads=8, bev_in_channels=384,
                 dropout=0.1, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.dropout = nn.Dropout(dropout)
        # projection from bev_channels to embed_dims if needed, else identity
        self.bev_proj = nn.Linear(bev_in_channels, embed_dims) \
            if bev_in_channels != embed_dims else nn.Identity()
        # standart multihead attention for cross-attention from query to bev features
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def _make_bev_pos_embed(self, H, W, device, dtype):
        """2D sine positional encoding. Returns [H*W, embed_dims]."""
        half = self.embed_dims // 2
        grid_y = torch.arange(H, device=device, dtype=dtype) / H
        grid_x = torch.arange(W, device=device, dtype=dtype) / W
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x)          # [H, W]
        dim_t = 10000 ** (2 * torch.arange(half // 2, device=device, dtype=dtype) / half)
        pos_x = grid_x.flatten().unsqueeze(-1) / dim_t            # [H*W, half//2]
        pos_y = grid_y.flatten().unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], -1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], -1).flatten(-2)
        return torch.cat([pos_x, pos_y], dim=-1)                  # [H*W, embed_dims]

    def forward(self, query, key=None, value=None, residual=None,
                query_pos=None, bev_feat=None, **kwargs):
        """
        query    : [num_q, B, C]
        bev_feat : [B, bev_in_channels, H, W]
        Returns  : [num_q, B, C]  (residual added internally)
        """
        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        B, C_bev, H, W = bev_feat.shape
        bev_flat = bev_feat.flatten(2).permute(2, 0, 1)           # [H*W, B, C_bev]
        bev_flat = self.bev_proj(bev_flat)                         # [H*W, B, C]

        pos = self._make_bev_pos_embed(H, W, bev_flat.device, bev_flat.dtype)
        bev_flat = bev_flat + pos.unsqueeze(1).expand(-1, B, -1)  # [H*W, B, C]

        out, _ = self.attn(query=query, key=bev_flat, value=bev_flat)
        out = self.output_proj(out)
        return self.dropout(out) + inp_residual


@ATTENTION.register_module()
class SMCACrossAtten(BaseModule):
    """Spatially Modulated Cross-Attention for camera features.
    Single FPN level (finest). Gaussian mask biases attention toward
    the projected bounding box in image space.
    TransFusion decoder layer 1.
    """
    def __init__(self, embed_dims=256, num_heads=8, num_cams=3,
                 pc_range=None, dropout=0.1, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims  = embed_dims
        self.num_heads   = num_heads
        self.num_cams    = num_cams
        self.pc_range    = pc_range
        self.head_dim    = embed_dims // num_heads
        self.scale       = self.head_dim ** -0.5
        self.drop        = nn.Dropout(dropout)

        self.q_proj      = nn.Linear(embed_dims, embed_dims)
        self.k_proj      = nn.Linear(embed_dims, embed_dims)
        self.v_proj      = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # MLP that encodes the 3D position x,y,z of the reference point into a positional bias added to the output
        # dim=3 -> embed_dims, with 2 layers and ReLU in between
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
        )

    def init_weight(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.output_proj]:
            xavier_init(m, distribution='uniform', bias=0.)

    # projects all 300 query positions onto all 3 cameras simultaneously
    def _project(self, reference_points, img_metas):
        """Return cx, cy [B,num_q,num_cam], depth, valid, ref_3d."""
        lidar2img = reference_points.new_tensor(
            np.asarray([m['lidar2img'] for m in img_metas]))     # [B, num_cam, 4, 4]
        B, num_q, _ = reference_points.shape
        ref = reference_points.clone()
        ref_3d = reference_points.clone()
        ref[..., 0] = ref[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        ref[..., 1] = ref[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        ref[..., 2] = ref[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        ref_h = torch.cat([ref, torch.ones_like(ref[..., :1])], -1)          # [B,num_q,4]
        ref_h = ref_h.view(B, 1, num_q, 4, 1).expand(-1, self.num_cams, -1, -1, -1)
        l2i   = lidar2img.view(B, self.num_cams, 1, 4, 4).expand(-1, -1, num_q, -1, -1)
        cam   = torch.matmul(l2i, ref_h).squeeze(-1)                          # [B,num_cam,num_q,4]

        eps   = 1e-5
        depth = cam[..., 2]                                                    # [B,num_cam,num_q]
        valid = depth > eps
        pts2d = cam[..., :2] / cam[..., 2:3].clamp(min=eps)                   # [B,num_cam,num_q,2]
        img_h, img_w = img_metas[0]['img_shape'][0][0][:2]
        pts2d[..., 0] /= img_w
        pts2d[..., 1] /= img_h
        # valid only if projected point is in front of camera and within image boundaries
        valid = valid & (pts2d[..., 0] > 0) & (pts2d[..., 0] < 1) \
                      & (pts2d[..., 1] > 0) & (pts2d[..., 1] < 1)

        cx    = pts2d[..., 0].permute(0, 2, 1)   # [B, num_q, num_cam]
        cy    = pts2d[..., 1].permute(0, 2, 1)
        depth = depth.permute(0, 2, 1)
        valid = valid.permute(0, 2, 1)
        return cx, cy, depth, valid, ref_3d

    def _gaussian(self, cx, cy, depth, ref_size, H, W, device, dtype):
        """Returns [B, num_q, num_cam, H*W] Gaussian masks."""
        B, num_q, num_cam = cx.shape
        gy = torch.arange(H, device=device, dtype=dtype) / (H - 1)
        gx = torch.arange(W, device=device, dtype=dtype) / (W - 1)
        gy, gx = torch.meshgrid(gy, gx)                           # [H, W]

        # sigma: avg box footprint in world → project to image (normalized)
        box_wl = ref_size[..., :2].exp().mean(-1)                 # [B, num_q]  metres
        sigma  = (box_wl.unsqueeze(-1) / depth.clamp(min=1.) / 20.).clamp(0.02, 0.3)
        # [B, num_q, num_cam, 1]
        sigma  = sigma.unsqueeze(-1)
        cx     = cx.unsqueeze(-1)
        cy     = cy.unsqueeze(-1)
        gx     = gx.flatten().view(1, 1, 1, -1)
        gy     = gy.flatten().view(1, 1, 1, -1)
        return torch.exp(-((gx - cx)**2 + (gy - cy)**2) / (2 * sigma**2 + 1e-8))

    def forward(self, query, key=None, value=None, residual=None,
                query_pos=None, reference_points=None, ref_size=None,
                img_metas=None, **kwargs):
        """
        query            : [num_q, B, C]
        value            : list of FPN levels; value[0] used  [B, num_cam, C, H, W]
        reference_points : [B, num_q, 3] normalised [0,1]
        ref_size         : [B, num_q, 3] log space
        Returns          : [num_q, B, C]  (residual + pos_feat added internally)
        """
        # save original query
        inp_residual = query
        # add positional encoding
        if query_pos is not None:
            query = query + query_pos

        num_q, B, C = query.shape
        img_feat     = value[0]                                    # [B,num_cam,C,H_f,W_f]
        _, _, _, H_f, W_f = img_feat.shape

        # get projected reference points and validity mask
        cx, cy, depth, valid, ref_3d = self._project(reference_points, img_metas)
        gauss = self._gaussian(cx, cy, depth, ref_size, H_f, W_f,
                               query.device, query.dtype)          # [B,num_q,num_cam,H_f*W_f]
        gauss = gauss * valid.float().unsqueeze(-1)                # zero invalid cameras

        query_b = query.permute(1, 0, 2)                           # [B, num_q, C]
        Q = self.q_proj(query_b).view(B, num_q, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)                                  # [B, nH, num_q, hd]

        cam_outs = []
        for cam_i in range(self.num_cams):
            feat = img_feat[:, cam_i]                              # [B, C, H_f, W_f]
            ff   = feat.flatten(2).permute(0, 2, 1)               # [B, H*W, C]
            K = self.k_proj(ff).view(B, H_f*W_f, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # [1, 8, 32, 5600]  Keys
            V = self.v_proj(ff).view(B, H_f*W_f, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [1, 8, 5600, 32]  Values

            attn = torch.matmul(Q, K) * self.scale                # [B, nH, num_q, H*W]
            g    = gauss[:, :, cam_i].unsqueeze(1)                 # [B, 1, num_q, H*W]
            attn = attn + torch.log(g.clamp(min=1e-6))            # log-domain Gaussian mask
            attn = torch.softmax(attn, dim=-1)

            out  = torch.matmul(attn, V)                           # [B, nH, num_q, hd]
            out  = out.permute(0, 2, 1, 3).reshape(B, num_q, C)
            cam_outs.append(out)

        cam_stack = torch.stack(cam_outs, dim=2)                   # [B, num_q, num_cam, C]
        valid_sum = valid.float().sum(dim=2, keepdim=True).clamp(min=1)
        output    = (cam_stack * valid.float().unsqueeze(-1)).sum(2) / valid_sum  # [B,num_q,C]

        output   = self.output_proj(output).permute(1, 0, 2)      # [num_q, B, C]
        pos_feat = self.position_encoder(inverse_sigmoid(ref_3d)).permute(1, 0, 2)
        return self.drop(output) + inp_residual + pos_feat

Change 2 — transformer.py (append at end of file)
@TRANSFORMER.register_module()
class TransFusionTransformer(BaseModule):
    """Wrapper for TransFusion 2-layer decoder."""

    def __init__(self, decoder=None, **kwargs):
        super().__init__(**kwargs)
        self.decoder    = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mlvl_feats, query_embed, reference_points, ref_size,
                reg_branches=None, bev_feat=None, **kwargs):
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos        = query_pos.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)
        query            = query.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1).sigmoid()
        ref_size         = ref_size.unsqueeze(0).expand(bs, -1, -1)

        return self.decoder(
            query=query, key=None, value=mlvl_feats,
            query_pos=query_pos, reference_points=reference_points,
            reg_branches=reg_branches, ref_size=ref_size,
            bev_feat=bev_feat, **kwargs)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransFusionTransformerDecoder(BaseModule):
    """2-layer TransFusion decoder.
    Layer 0: self-attn + LiDAR BEV cross-attn + FFN
    Layer 1: self-attn + SMCA camera cross-attn + FFN
    """

    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=512, dropout=0.1,
                 lidar_bev_attn=None, smca_attn=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_layers = 2

        from .attention_dert3d import LiDARBEVCrossAtten, SMCACrossAtten

        def _ffn(d, h, drop):
            return nn.Sequential(
                nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(drop),
                nn.Linear(h, d), nn.Dropout(drop))

        # LiDAR BEV
        self.sa0 = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.ca0 = LiDARBEVCrossAtten(**(lidar_bev_attn or
                    dict(embed_dims=embed_dims, num_heads=num_heads)))
        self.ff0 = _ffn(embed_dims, ffn_dims, dropout)
        self.n0  = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.d0  = nn.Dropout(dropout)

        # SMCA camera
        self.sa1 = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.ca1 = SMCACrossAtten(**(smca_attn or
                    dict(embed_dims=embed_dims, num_heads=num_heads)))
        self.ff1 = _ffn(embed_dims, ffn_dims, dropout)
        self.n1  = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.d1  = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, query_pos=None,
                reference_points=None, reg_branches=None, ref_size=None,
                bev_feat=None, **kwargs):
        """
        query            : [num_q, bs, C]
        reference_points : [bs, num_q, 3] in [0,1]
        ref_size         : [bs, num_q, 3] log space
        bev_feat         : [bs, C_bev, H, W]
        value            : list of camera FPN levels [bs, num_cam, C, H, W]
        """
        intermediate, inter_ref, inter_size = [], [], []
        output = query

        # Layer 0: LiDAR BEV 
        q = k = output + query_pos

        # self attention between query and itself with residual
        sa, _ = self.sa0(q, k, output)
        output = self.n0[0](output + self.d0(sa))

        # cross attention between query and bev features with residual
        ca     = self.ca0(output, query_pos=query_pos, bev_feat=bev_feat, **kwargs)
        output = self.n0[1](ca)                    # ca already contains residual

        output = self.n0[2](output + self.ff0(output))
        output = output.permute(1, 0, 2)           # [bs, num_q, C]

        # L0 - regress the detection values (including center xy and depth, but not size) to update the reference points for the next layer 
        if reg_branches is not None:
            tmp = reg_branches[0](output)
            ref_pts_delta = torch.cat([tmp[..., :2], tmp[..., 4:5]], -1)
            ref_size_delta = torch.cat([tmp[..., 2:4], tmp[..., 5:6]], -1)
            new_ref = (ref_pts_delta + inverse_sigmoid(reference_points)).sigmoid()
            reference_points = new_ref.detach()
            ref_size = ref_size + ref_size_delta   # layer 0: gradient allowed

        intermediate.append(output)
        inter_ref.append(reference_points)
        inter_size.append(ref_size)
        output = output.permute(1, 0, 2)           # back to [num_q, bs, C]

        # Layer 1: SMCA camera 
        q = k = output + query_pos

        # self attention between query and itself with residual
        sa, _ = self.sa1(q, k, output)
        output = self.n1[0](output + self.d1(sa))

        # cross attention between query and camera features with residual
        ca     = self.ca1(output, value=value, query_pos=query_pos,
                          reference_points=reference_points, ref_size=ref_size,
                          **kwargs)
        output = self.n1[1](ca)

        # [300, 256]
        output = self.n1[2](output + self.ff1(output))

        output = output.permute(1, 0, 2)
        if reg_branches is not None:
            tmp = reg_branches[1](output)
            ref_pts_delta = torch.cat([tmp[..., :2], tmp[..., 4:5]], -1)
            ref_size_delta = torch.cat([tmp[..., 2:4], tmp[..., 5:6]], -1)
            new_ref = (ref_pts_delta + inverse_sigmoid(reference_points)).sigmoid()
            reference_points = new_ref.detach()
            ref_size = (ref_size + ref_size_delta).detach()   # layer 1+: detach

        intermediate.append(output)
        inter_ref.append(reference_points)
        inter_size.append(ref_size)

        return (torch.stack(intermediate),
                torch.stack(inter_ref),
                torch.stack(inter_size))

Note: needs from mmdet.models.utils.transformer import inverse_sigmoid imported at top of transformer.py.

Change 3 — head_plus_raw.py (append at end of file)
@HEADS.register_module()
class TransFusionDetHead(DeformableDETR3DCamHeadTrackPlusRaw):
    """TransFusion detection head: identical to parent but accepts bev_feat
    and passes it to the 2-layer TransFusion transformer."""

    def forward(self, mlvl_feats, radar_feats,
                query_embeds, ref_points, ref_size, img_metas,
                bev_feat=None, petr_feature=False):
        # identical positional encoding preprocessing from parent
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['input_shape']
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape'][0][0]
            img_masks[img_id, :img_h, :img_w] = 0

        for i, feat in enumerate(mlvl_feats):
            B, N, C, H, W = feat.size()
            mlvl_masks = F.interpolate(
                img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            pos_enc = self.positional_encoding(mlvl_masks)
            pos_enc = pos_enc.unsqueeze(1).repeat(1, N, 1, 1, 1)
            lvl_enc = self.level_embeds[i].view(1, 1, -1, 1, 1)
            cam_enc = self.cam_embeds.view(1, N, C, 1, 1)
            mlvl_feats[i] = feat + pos_enc + lvl_enc + cam_enc

        hs, inter_references, inter_box_sizes = self.transformer(
            mlvl_feats, query_embeds, ref_points, ref_size,
            reg_branches=self.reg_branches,
            img_metas=img_metas,
            radar_feats=radar_feats,
            bev_feat=bev_feat,
        )

        hs = hs.permute(0, 2, 1, 3)   # [num_dec, bs, num_q, C]
        outputs_classes, outputs_coords = [], []

        for lvl in range(hs.shape[0]):
            reference = ref_points.sigmoid() if lvl == 0 else inter_references[lvl - 1]
            ref_size_base = ref_size if lvl == 0 else inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            xywlzh        = self.reg_branches[lvl](hs[lvl])
            direction_pred = self.direction_branches[lvl](hs[lvl])
            velo_pred      = self.velo_branches[lvl](hs[lvl])

            xywlzh[..., 0:2] = (xywlzh[..., 0:2] + reference[..., 0:2]).sigmoid()
            xywlzh[..., 4:5] = (xywlzh[..., 4:5] + reference[..., 2:3]).sigmoid()
            last_ref_points  = torch.cat([xywlzh[..., 0:2], xywlzh[..., 4:5]], -1)

            xywlzh[..., 0:1] = xywlzh[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            xywlzh[..., 1:2] = xywlzh[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            xywlzh[..., 4:5] = xywlzh[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            xywlzh[..., 2:4] = xywlzh[..., 2:4] + ref_size_base[..., 0:2]
            xywlzh[..., 5:6] = xywlzh[..., 5:6] + ref_size_base[..., 2:3]

            outputs_classes.append(outputs_class)
            outputs_coords.append(torch.cat([xywlzh, direction_pred, velo_pred], dim=2))

        last_ref_points = inverse_sigmoid(last_ref_points)
        return (torch.stack(outputs_classes), torch.stack(outputs_coords),
                hs[-1], last_ref_points)


Change 4 — vip3d.py (lines 500–503)
Change:


output_classes, output_coords, \
    query_feats, last_ref_pts = self.pts_bbox_head(
    img_feats, radar_feats, track_instances.query,
    track_instances.ref_pts, ref_box_sizes, img_metas)
To:


output_classes, output_coords, \
    query_feats, last_ref_pts = self.pts_bbox_head(
    img_feats, radar_feats, track_instances.query,
    track_instances.ref_pts, ref_box_sizes, img_metas,
    bev_feat=bev_feat if (self.use_lidar and pts_feats is not None) else None)
Change 5 — Config livip3d_resnet50_3cam.py
Replace the entire pts_bbox_head dict:


pts_bbox_head=dict(
    type='TransFusionDetHead',
    num_classes=7,
    in_channels=256,
    num_cams=3,
    num_feature_levels=4,
    transformer=dict(
        type='TransFusionTransformer',
        decoder=dict(
            type='TransFusionTransformerDecoder',
            embed_dims=256,
            num_heads=8,
            ffn_dims=512,
            dropout=0.1,
            lidar_bev_attn=dict(
                embed_dims=256,
                num_heads=8,
                bev_in_channels=384,
                dropout=0.1,
            ),
            smca_attn=dict(
                embed_dims=256,
                num_heads=8,
                num_cams=3,
                pc_range=point_cloud_range,
                dropout=0.1,
            ),
        )),
    pc_range=point_cloud_range,
    positional_encoding=dict(
        type='SinePositionalEncoding',
        num_feats=128,
        normalize=True,
        offset=-0.5),
),
That's all 5 changes. A few things to double-check before applying:

transformer.py needs from mmdet.models.utils.transformer import inverse_sigmoid at the top — check it's already there or add it
head_plus_raw.py TransFusionDetHead inherits from DeformableDETR3DCamHeadTrackPlusRaw, so it gets _init_layers, init_weights, __init__ for free — only forward is overridden
The __init__.py in plugin/vip3d/models/ may need TransFusionDetHead, TransFusionTransformer, TransFusionTransformerDecoder, LiDARBEVCrossAtten, SMCACrossAtten added to its exports
Want me to check what's currently in __init__.py?