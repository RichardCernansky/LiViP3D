"""
Appearance-guided Past Motion Refinement (APR).

Inspired by MASAR (Bencheikh Lehocine et al., 2026): instead of trusting a
track's continuation state unconditionally, generate several candidate past
trajectories, sample each against the raw BEV features actually recorded at
that past moment, and let the evidence decide which (if any) is believable.

This module only touches the query CONTENT that feeds into the existing
LiDAR/SMCA decoder layers — it never changes where those layers sample this
frame's own evidence, and it never writes to track_instances.ref_pts itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AppearanceGuidedPastMotionRefinement(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 num_hypotheses=3,
                 history_len=2,
                 bev_in_channels=384,
                 num_heads=8,
                 dropout=0.1,
                 loss_weight=0.2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_hypotheses = num_hypotheses
        self.history_len = history_len
        # apr_loss (scoring + regression) starts out several times larger than every
        # other loss term combined — reg_loss is raw L1 error in metres, nothing
        # normalizes that away — so, like loss_cls/loss_bbox, it carries its own
        # weight rather than entering the total at raw scale.
        self.loss_weight = loss_weight

        # q_mo = MLP_init(q_obj)
        self.motion_init = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # E_past: one learned offset embedding per hypothesis, shared across all tracks
        self.hypothesis_embed = nn.Embedding(num_hypotheses, embed_dims)

        # PE(P): per-timestep position stamp — MASAR: "for past-conditioning ... + PE(P),
        # where PE(.) denotes sinusoidal positional encoding" (Sec III-D). This codebase's
        # own convention for "encode a position into a feature" (attention_dert3d.py's
        # position_encoder) is a small MLP over normalized coords rather than literal
        # sin/cos — matched here instead of introducing a scheme used nowhere else in
        # this codebase.
        self.time_pos_encoder = nn.Sequential(
            nn.Linear(2, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
        )

        # factorized self-attention: hypotheses compare against each other
        self.mode_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.mode_norm = nn.LayerNorm(embed_dims)

        # factorized self-attention: timesteps within one hypothesis refine each other
        self.time_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.time_norm = nn.LayerNorm(embed_dims)

        self.motion_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims),
        )
        self.motion_ffn_norm = nn.LayerNorm(embed_dims)

        # per (hypothesis, timestep): a 2D correction on top of the constant-velocity seed
        self.traj_reg = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True), nn.Linear(embed_dims, 2),
        )

        # project sampled raw BEV channels to embed_dims (mirrors LiDARBEVDeformCrossAtten).
        # Unlike that module, this branch's output feeds a loss (score_head, below) and a
        # residual update directly — nothing downstream normalizes it for us — so, unlike
        # LiDARBEVDeformCrossAtten, it needs its own norm right here rather than relying on
        # a later decoder-layer norm to absorb whatever scale the raw BEV features carry.
        self.bev_proj = nn.Linear(bev_in_channels, embed_dims) \
            if bev_in_channels != embed_dims else nn.Identity()
        self.bev_norm = nn.LayerNorm(embed_dims)

        self.aggregate = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims),
        )
        self.aggregate_norm = nn.LayerNorm(embed_dims)

        self.score_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2), nn.ReLU(inplace=True), nn.Linear(embed_dims // 2, 1),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.traj_reg[-1].weight)
        nn.init.zeros_(self.traj_reg[-1].bias)

    @staticmethod
    def _denorm_xyz(ref_pts, pc_range):
        # ref_pts arrives already in normalized [0,1] sigmoid space — the decoder applies
        # .sigmoid() once, at the top of TransFusionTransformerDecoder.forward() (and
        # again inside _update_ref), before either APR call site ever sees it. Sigmoiding
        # it again here would double-squash an already-bounded value toward ~[0.5, 0.73]
        # instead of denormalizing it.
        p = ref_pts
        x = p[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        y = p[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        z = p[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _norm_xy(xy, pc_range):
        x = (xy[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        y = (xy[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        return torch.stack([x, y], dim=-1)

    def forward(self, query, ref_pts, velocity,
                history_bev_feats, history_ego_r, history_ego_t,
                cur_ego_r, cur_ego_t, time_delta, pc_range):
        """
        Args:
            query:    [N, D]   query content for the currently-alive tracks
            ref_pts:  [N, 3]   this frame's starting reference points, normalized [0,1]
                                sigmoid space (already .sigmoid()'d by the decoder)
            velocity: [N, 2]   last known vx, vy (current lidar/ego frame, m/s)
            history_bev_feats: list of up to `history_len` [1, C, Hbev, Wbev] tensors,
                                oldest first, NOT including this frame
            history_ego_r/t:   matching list of (rotation [3,3], translation [1,3]) for
                                those same past frames' ego2global transform
            cur_ego_r/t:       this frame's own ego2global (rotation, translation)
            time_delta:        seconds between consecutive frames (scalar tensor)
            pc_range:          [xmin, ymin, zmin, xmax, ymax, zmax]

        Returns:
            query_out: [N, D]  content-refined query (residual update)
            info: dict with 'score' [N, num_hypotheses] and 'verified_traj' [N, H, 2],
                  or None if there was no history to use.
        """
        H = len(history_bev_feats)
        N = query.shape[0]
        if H == 0 or N == 0:
            return query, None
        H = min(H, self.history_len)
        # Stored oldest-first (_push_apr_history appends each frame). Reversed here so
        # index 0 = most recent past (1 step back), matching `steps` below (steps[0]=1)
        # and compute_apr_loss's GT lookup (h=1 -> frame_idx-1 -> gt_xy[:, 0]) — without
        # this, the "1-step-back" seed got realigned/sampled against the oldest buffered
        # frame's ego pose and BEV instead of the newest, and vice versa.
        history_bev_feats = history_bev_feats[-H:][::-1]
        history_ego_r = history_ego_r[-H:][::-1]
        history_ego_t = history_ego_t[-H:][::-1]

        D = self.embed_dims
        Mh = self.num_hypotheses
        device = query.device

        # Various inputs (ego pose from the dataloader; ref_pts, when it originates
        # from a heatmap grid-index computation rather than a network layer) can
        # silently carry float64 into this function. Plain arithmetic (+, -, *) would
        # quietly promote to float64 with no error and no visible symptom until it
        # hits a matmul far downstream, which raises instead of promoting — so pin
        # everything to a single dtype here, once, rather than chase it per-line.
        dtype = query.dtype
        ref_pts = ref_pts.to(dtype)
        velocity = velocity.to(dtype)
        cur_ego_r = cur_ego_r.to(dtype)
        cur_ego_t = cur_ego_t.to(dtype)
        history_ego_r = [r.to(dtype) for r in history_ego_r]
        history_ego_t = [t.to(dtype) for t in history_ego_t]
        history_bev_feats = [b.to(dtype) for b in history_bev_feats]

        # time_delta is None on whichever frame has no "next" frame (it was designed
        # for velo_update's forward-looking use) — but APR only ever needs the interval
        # between CONSECUTIVE PAST frames, which is well-defined regardless of that and,
        # for this fixed 2Hz dataset, is just the standard keyframe interval.
        if time_delta is None:
            time_delta = torch.tensor(0.5, device=device, dtype=dtype)
        else:
            time_delta = time_delta.to(device=device, dtype=dtype)

        # 1) base motion summary, broadcast into Mh candidate hypotheses (Algorithm 1
        # line 3: Qmo = qmo + Epast — happens once, before MotionDecoder's own steps)
        q_mo = self.motion_init(query)                                    # [N, D]
        Q_mo = q_mo.unsqueeze(1) + self.hypothesis_embed.weight.unsqueeze(0)  # [N, Mh, D]

        # 2) constant-velocity seed positions, stepped backward, current-ego frame, metric
        # — this is `P` in MotionDecoder(Qmo, P): the only position estimate available
        # before the factorized attention blocks run.
        ref_metric = self._denorm_xyz(ref_pts, pc_range)                  # [N, 3]
        steps = torch.arange(1, H + 1, device=device, dtype=ref_metric.dtype)  # [H]
        vel_pad = F.pad(velocity, (0, 1))                                 # [N, 3], vz=0
        # ref_metric − displacement
        seed = ref_metric.unsqueeze(1) - vel_pad.unsqueeze(1) * steps.view(1, H, 1) * time_delta  # [N, H, 3]

        # 3) MotionDecoder's own two factorized attention blocks, matching MASAR's stated
        # order (Sec III-D: "factorized temporal attention, factorized mode attention") —
        # both operate on the same [N, Mh, H, D] volume, unlike doing mode-attention once
        # up front: (a) broadcast, (b) stamp each step with its position (PE(P)),
        # (c) attend along Th fixing Mh, (d) attend along Mh fixing Th.
        Q_vol = Q_mo.unsqueeze(2).expand(-1, -1, H, -1)                   # [N, Mh, H, D]
        seed_xy_norm = self._norm_xy(seed[..., :2], pc_range).clamp(0.0, 1.0)  # [N, H, 2]
        pos_embed = self.time_pos_encoder(seed_xy_norm).unsqueeze(1)      # [N, 1, H, D]
        Q_vol = Q_vol + pos_embed                                         # [N, Mh, H, D]

        t_in = Q_vol.reshape(N * Mh, H, D).permute(1, 0, 2)               # [H, N*Mh, D]
        t_attn, _ = self.time_attn(t_in, t_in, t_in)
        Q_vol = self.time_norm(t_in + t_attn).permute(1, 0, 2).reshape(N, Mh, H, D)

        m_in = Q_vol.permute(1, 0, 2, 3).reshape(Mh, N * H, D)            # [Mh, N*H, D]
        m_attn, _ = self.mode_attn(m_in, m_in, m_in)
        Q_vol = self.mode_norm(m_in + m_attn).reshape(Mh, N, H, D).permute(1, 0, 2, 3)  # [N, Mh, H, D]

        Q_vol = self.motion_ffn_norm(Q_vol + self.motion_ffn(Q_vol))

        # 4) regress a correction on top of the constant-velocity seed
        delta = self.traj_reg(Q_vol)                                      # [N, Mh, H, 2]
        seed_xy = seed[..., :2].unsqueeze(1).expand(-1, Mh, -1, -1)       # [N, Mh, H, 2]
        hyp_xy = (seed_xy + delta).to(dtype)                              # [N, Mh, H, 2], current-ego frame
        # ^ pinned to `dtype` here deliberately: plain `+` silently promotes to a wider
        # dtype with no error if either side is (e.g. `seed` inheriting float64 from
        # somewhere upstream via ordinary arithmetic) — unlike `@` below, which raises
        # instead of promoting. Pin once here so nothing downstream can inherit a silent
        # promotion and only fail much later, at the matmul, far from the actual cause.

        # 5) for each stored past frame: realign this hypothesis' claimed position into
        #    THAT frame's own ego pose, then sample THAT frame's own buffered BEV there
        sampled_per_step = []
        realigned_per_step = []
        for h in range(H):
            pts_cur = hyp_xy[:, :, h, :]                                  # [N, Mh, 2]
            pts_cur_3 = F.pad(pts_cur, (0, 1))                            # z=0, [N, Mh, 3]

            g = pts_cur_3 @ cur_ego_r.T + cur_ego_t                       # current ego -> global
            local = (g - history_ego_t[h]) @ history_ego_r[h]            # global -> ego(t-k)
            realigned_per_step.append(local[..., :2])                    # [N, Mh, 2], frame (t-h)'s
                                                                            # own ego coords, real metres
                                                                            # — directly comparable to that
                                                                            # frame's own GT box centre

            norm_xy = self._norm_xy(local[..., :2], pc_range).clamp(0.0, 1.0)  # [N, Mh, 2]
            grid = (norm_xy * 2 - 1).reshape(1, N * Mh, 1, 2)             # grid_sample wants [-1,1]

            bev = history_bev_feats[h].float()                           # [1, C, Hbev, Wbev]
            samp = F.grid_sample(bev, grid.float(), mode='bilinear',
                                  align_corners=False, padding_mode='zeros')  # [1, C, N*Mh, 1]
            samp = samp.squeeze(0).squeeze(-1).permute(1, 0)              # [N*Mh, C]
            sampled_per_step.append(self.bev_norm(self.bev_proj(samp)).view(N, Mh, D))

        sampled = torch.stack(sampled_per_step, dim=2)                    # [N, Mh, H, D]
        realigned_xy = torch.stack(realigned_per_step, dim=2)             # [N, Mh, H, 2]

        # 6) aggregate evidence across the H sampled timesteps, score each hypothesis
        F_traj = self.aggregate_norm(self.aggregate(sampled.mean(dim=2)))  # [N, Mh, D]
        score = self.score_head(F_traj).squeeze(-1)                       # [N, Mh]
        weights = score.softmax(dim=-1)

        # 7) SOFT update of the query's content — every hypothesis contributes, weighted
        query_out = query + (weights.unsqueeze(-1) * F_traj).sum(dim=1)   # [N, D]

        # 8) HARD selection, kept for logging / optional downstream use (e.g. as an
        #    alternative to velo_update's raw single-point extrapolation)
        best = score.argmax(dim=-1)                                       # [N]
        verified_traj = hyp_xy[torch.arange(N, device=device), best]      # [N, H, 2]

        return query_out, {
            'score': score,                # [N, Mh] — raw (pre-softmax) per-hypothesis score
            'verified_traj': verified_traj,  # [N, H, 2] — winning hypothesis, current-ego frame
            'realigned_xy': realigned_xy,    # [N, Mh, H, 2] — every hypothesis, each step already
                                              # in that step's own past-ego frame; this is what the
                                              # APR loss compares against real GT positions
        }


def build_apr(args, embed_dims):
    if not args:
        return None
    return AppearanceGuidedPastMotionRefinement(
        embed_dims=embed_dims,
        num_hypotheses=args.get('num_hypotheses', 3),
        history_len=args.get('history_len', 2),
        bev_in_channels=args.get('bev_in_channels', 384),
        num_heads=args.get('num_heads', 8),
        dropout=args.get('dropout', 0.1),
        loss_weight=args.get('loss_weight', 0.2),
    )


def compute_apr_loss(decoder, gt_instances_seq, frame_idx, obj_idxes):
    """
    Ground-truth supervision for APR's two call sites this frame (decoder.last_apr,
    stashed by _run_apr in transformer.py). Algorithm 1 in the MASAR paper never
    included a loss — it's purely the forward procedure above; this is the separate
    training objective that supervises what it produces, adapted to what forward()
    actually outputs (a 2D correction, not a full Laplacian mean+scale) and to having
    two call sites (l0/l1) per frame instead of one homogeneous L_d loop.

    For every currently-alive track, look up its REAL past position (by stable
    dataset identity, via obj_idxes — not a distance threshold, since exact identity
    is already available on the training path) at each buffered history step, then:

      - regression: the hypothesis closest to the real position (oracle selection,
        same winner-take-all convention predictor_decoder.py already uses via
        argmin) gets pushed closer via L1 — only the winner.
      - scoring: every hypothesis's score is pushed toward a soft target derived
        from ITS OWN distance to the real position — not just the winner — so
        score becomes a calibrated per-hypothesis confidence rather than only a
        relative ranking among that round's candidates.

    Args:
        decoder: TransFusionTransformerDecoder instance (.apr, .last_apr)
        gt_instances_seq: List[Instances], the whole clip's GT — self.criterion.gt_instances
        frame_idx: int, this frame's index within the clip — self.criterion._current_frame_idx
        obj_idxes: [num_q] long tensor, track_instances.obj_idxes as of THIS frame's
                   APR calls (valid to read straight after pts_bbox_head returns —
                   this frame's own matching hasn't touched it yet at that point)

    Returns:
        scalar loss tensor, or None if there was nothing to supervise this frame.
    """
    if decoder.apr is None or not decoder.last_apr:
        return None

    total = None
    for _tag, entry in decoder.last_apr.items():
        info = entry['info']
        alive_mask = entry['alive_mask']
        realigned_xy = info['realigned_xy']          # [N, Mh, H, 2]
        score = info['score']                         # [N, Mh]
        N, Mh, H, _ = realigned_xy.shape
        if N == 0:
            continue

        track_ids = obj_idxes[alive_mask].detach().cpu().tolist()

        gt_xy = realigned_xy.new_zeros(N, H, 2)
        valid = torch.zeros(N, H, dtype=torch.bool, device=realigned_xy.device)
        for h in range(1, H + 1):
            past_idx = frame_idx - h
            if past_idx < 0:
                continue
            past_gt = gt_instances_seq[past_idx]
            if len(past_gt) == 0:
                continue
            id_to_row = {int(v): i for i, v in enumerate(past_gt.obj_ids.detach().cpu().tolist())}
            for n, oid in enumerate(track_ids):
                row = id_to_row.get(int(oid))
                if row is not None:
                    gt_xy[n, h - 1] = past_gt.boxes[row, 0:2]
                    valid[n, h - 1] = True

        track_valid = valid.any(dim=-1)                # [N] — has at least one real position to check
        if not track_valid.any():
            continue

        dist = (realigned_xy - gt_xy.unsqueeze(1)).norm(dim=-1)    # [N, Mh, H]
        valid_f = valid.unsqueeze(1).float()                        # [N, 1, H]
        step_count = valid_f.sum(dim=-1).clamp(min=1)                # [N, 1]
        ade = (dist * valid_f).sum(dim=-1) / step_count              # [N, Mh]

        # scoring loss — every hypothesis, soft target from its own accuracy.
        # Detached: target is meant to be a frozen label (like any GT-derived target),
        # but ade depends on realigned_xy (the model's own live output), not just gt_xy —
        # left un-detached, BCE's gradient w.r.t. its target argument (-score, unbounded,
        # unlike the bounded sigmoid(score)-target gradient into score itself) leaks back
        # through ade/dist into realigned_xy on top of the intended regression path.
        target = torch.exp(-ade).detach()                            # [N, Mh], in (0, 1]
        scoring_loss = F.binary_cross_entropy_with_logits(
            score[track_valid], target[track_valid], reduction='mean')

        # regression loss — oracle-closest hypothesis only, winner-take-all
        best = ade.argmin(dim=-1)                                    # [N]
        idx = torch.arange(N, device=realigned_xy.device)
        best_xy = realigned_xy[idx, best]                             # [N, H, 2]
        reg_diff = (best_xy - gt_xy).abs() * valid.unsqueeze(-1).float()
        reg_loss = reg_diff[track_valid].sum() / valid[track_valid].sum().clamp(min=1)

        term = scoring_loss + reg_loss
        total = term if total is None else total + term

    if total is None:
        return None
    return total * decoder.apr.loss_weight
