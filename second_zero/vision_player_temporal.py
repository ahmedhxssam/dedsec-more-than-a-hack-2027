"""Identity-preserving temporal player interactions before team aggregation.

Player slots remain separate through temporal encoding and social attention.
They are pooled only afterward. Slots are observation tracks, not guaranteed
identities; direct position jumps mark temporal messages as unavailable.
No contact, distance, or interpolated coordinate is a possession label.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .contextual import velocity
from .set_ownership import player_set_features, FEATURES as SET_FEATURES

VERSION = 'vision-player-temporal-v1'
FEATURE_CONTEXT = 250
PRE_DILATIONS = (1, 2, 4)
POST_DILATIONS = (8, 16, 32, 64)
TEAM_DILATIONS = (1, 2, 4, 8, 16)
RADIUS = max(sum(PRE_DILATIONS), 31) + sum(POST_DILATIONS + TEAM_DILATIONS)
FEATURES = SET_FEATURES + ['estimated_relative_x', 'estimated_relative_y',
    'estimated_distance', 'estimated_relative_velocity_x', 'estimated_relative_velocity_y',
    'estimated_radial_speed', 'ball_estimated', 'estimated_relative_available',
    'estimated_relative_velocity_available', 'estimate_span', 'track_break']
BREAK_CHANNEL = len(FEATURES) - 1


def features(ball, players, team, pitch=(105.3, 68)):
    """Native25Hz observations plus explicitly uncertain bounded linear geometry."""
    ball, players, team = np.asarray(ball, np.float32), np.asarray(players, np.float32), np.asarray(team)
    base = player_set_features(ball, players, team, pitch)
    n, p = players.shape[:2]; ix = np.arange(n); seen = np.isfinite(ball).all(1)
    prev = np.maximum.accumulate(np.where(seen, ix, -n))
    nxt = np.minimum.accumulate(np.where(seen, ix, 2 * n)[::-1])[::-1]
    span = nxt - prev
    filled = (~seen) & (prev >= 0) & (nxt < n) & (span <= 250)
    estimated = ball.copy()
    known = ix[seen]
    if len(known) >= 2:
        for axis in (0, 1):
            estimated[filled, axis] = np.interp(ix[filled], known, ball[known, axis])
    pv, bv = velocity(players), velocity(estimated)
    valid_player = np.isfinite(players).all(2)
    valid_motion = np.isfinite(pv).all(2) & (np.linalg.norm(pv, axis=2) < 15)
    valid_ball = np.isfinite(estimated).all(1)
    valid_ball_motion = np.isfinite(bv).all(1) & (np.linalg.norm(bv, axis=1) < 60)
    rel = players - estimated[:, None]; distance = np.linalg.norm(rel, axis=2)
    rv = pv - bv[:, None]
    available = valid_player & valid_ball[:, None]
    rv_available = valid_motion & valid_ball_motion[:, None]
    rel[~available] = np.nan; distance[~available] = np.nan; rv[~rv_available] = np.nan
    radial = np.sum(rel * rv, axis=2) / np.maximum(distance, .1)
    jump = np.zeros((n, p), bool)
    if n > 1:
        jump[1:] = valid_player[1:] & valid_player[:-1] & (np.linalg.norm(players[1:] - players[:-1], axis=2) > 2.)
    repeat = lambda x: np.broadcast_to(x[:, None], (n, p))
    extra = np.stack([rel[..., 0] / 20, rel[..., 1] / 20, distance / 20,
        rv[..., 0] / 30, rv[..., 1] / 30, radial / 30, repeat(filled), available,
        rv_available, repeat(np.minimum(span, 250) / 250), jump], axis=-1)
    extra = np.clip(np.nan_to_num(extra, nan=0, posinf=0, neginf=0), -5, 5).astype(np.float16)
    out = np.zeros((*base.shape[:-1], len(FEATURES)), np.float16)
    out[..., :len(SET_FEATURES)] = base
    for t in (0, 1):
        slots = np.flatnonzero(team == t)
        out[:, t, :len(slots), len(SET_FEATURES):] = extra[:, slots]
    return out


def crop_features(raw, lo, hi):
    """Compute small crops with sufficient raw context; no feature cache needed."""
    n = len(raw['ball']); start, stop = max(0, lo - FEATURE_CONTEXT), min(n, hi + FEATURE_CONTEXT)
    x = features(raw['ball'][start:stop], raw['players'][start:stop], raw['team'], raw['pitch'])
    take = x[max(0, lo) - start:min(n, hi) - start]
    return np.pad(take, ((max(0, -lo), max(0, hi - n)), (0, 0), (0, 0), (0, 0)))


class TrackTemporalBlock(nn.Module):
    """Dilated same-slot messages that cannot cross a directly observed jump."""
    def __init__(self, width, dilation, dropout=.1):
        super().__init__(); self.dilation = dilation
        self.transforms = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))
        self.bias = nn.Parameter(torch.zeros(width)); self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, breaks):
        # batch/track,T,C; no reduction over time or across player identities.
        d, n = self.dilation, h.shape[1]
        value = self.transforms[1](h) + self.bias
        if n > d:
            cumulative = breaks.cumsum(1)
            permitted = (cumulative[:, d:] - cumulative[:, :-d] == 0).to(h.dtype).unsqueeze(-1)
            left = self.transforms[0](h[:, :-d]) * permitted
            right = self.transforms[2](h[:, d:]) * permitted
            value = value + F.pad(left, (0, 0, d, 0)) + F.pad(right, (0, 0, 0, d))
        return h + .5 * self.dropout(F.gelu(self.norm(value)))


class SocialAttention(nn.Module):
    """Shared player interaction graph with relative-team and distance biases."""
    def __init__(self, width, heads=2):
        super().__init__(); self.width = width; self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.distance_scale = nn.Parameter(torch.zeros(heads))
        self.relative_team_bias = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(width, width); self.norm = nn.LayerNorm(width)

    def forward(self, h, positions, active, position_available=None):
        b, n, teams, p, width = h.shape; k = teams * p
        tokens = h.reshape(b * n, k, width)
        xyz = positions.reshape(b * n, k, 2)
        mask = active.reshape(b * n, k)
        qkv = self.qkv(tokens).reshape(b * n, k, 3, self.heads, width // self.heads)
        q, key, value = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        distance = torch.linalg.vector_norm(xyz[:, :, None] - xyz[:, None, :], dim=-1)
        if position_available is not None:
            observed = position_available.reshape(b * n, k)
            distance = distance * (observed[:, :, None] & observed[:, None, :]).to(distance.dtype)
        team_id = torch.arange(k, device=h.device) // p
        same = (team_id[:, None] == team_id[None, :]).to(h.dtype)
        same = same.expand(b * n, -1, -1)
        bias = (-F.softplus(self.distance_scale)[None, :, None, None] * distance[:, None]
                + self.relative_team_bias[None, :, None, None] * same[:, None])
        logits = q @ key.transpose(-2, -1) / (width // self.heads) ** .5 + bias
        logits = logits.masked_fill(~mask[:, None, None], -1e4)
        attention = logits.softmax(-1) * mask[:, None, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (attention @ value).transpose(1, 2).reshape(b * n, k, width)
        result = self.norm(tokens + self.output(context))
        return result.reshape(b, n, teams, p, width)


class PlayerTemporalOwner(nn.Module):
    def __init__(self, channels=len(FEATURES), player_width=24, team_width=32, auxiliary_boundary=True, mask_missing_positions=False):
        super().__init__(); self.channels = channels; self.player_width = player_width
        self.team_width = team_width; self.auxiliary_boundary = auxiliary_boundary
        self.mask_missing_positions = mask_missing_positions
        self.input = nn.Sequential(nn.Linear(channels, player_width), nn.GELU())
        self.before = nn.ModuleList(TrackTemporalBlock(player_width, d) for d in PRE_DILATIONS)
        self.social = SocialAttention(player_width)
        self.after = nn.ModuleList(TrackTemporalBlock(player_width, d) for d in POST_DILATIONS)
        self.pool_attention = nn.Linear(player_width, 1)
        self.team_input = nn.Conv1d(6 * player_width + 2, team_width, 1)
        self.team_blocks = nn.ModuleList(nn.Conv1d(team_width, team_width, 3, dilation=d, padding=d) for d in TEAM_DILATIONS)
        self.team_norms = nn.ModuleList(nn.LayerNorm(team_width) for _ in TEAM_DILATIONS)
        self.owner_output = nn.Conv1d(team_width, 1, 1)
        self.boundary_output = nn.Conv1d(team_width, 1, 1)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, return_boundary=False):
        b, n, teams, p, channels = x.shape
        if teams != 2 or channels != self.channels:
            raise ValueError('Expected B,T,2,P,C player trajectories')
        present = x[..., 12] > .5
        flat_mask = present.permute(0, 2, 3, 1).reshape(b * teams * p, 1, n).to(x.dtype)
        active = F.max_pool1d(flat_mask, 63, stride=1, padding=31)[:, 0] > 0
        active = active.reshape(b, teams, p, n).permute(0, 3, 1, 2)
        breaks = x[..., BREAK_CHANNEL].permute(0, 2, 3, 1).reshape(b * teams * p, n)
        h = self.input(x).permute(0, 2, 3, 1, 4).reshape(b * teams * p, n, self.player_width)
        for block in self.before:
            h = block(h, breaks)
        h = h.reshape(b, teams, p, n, self.player_width).permute(0, 3, 1, 2, 4)
        h = self.social(h, x[..., 8:10], active, present if self.mask_missing_positions else None)
        h = h.permute(0, 2, 3, 1, 4).reshape(b * teams * p, n, self.player_width)
        for block in self.after:
            h = block(h, breaks)
        h = h.reshape(b, teams, p, n, self.player_width).permute(0, 3, 1, 2, 4)
        mask = active[..., None].to(h.dtype)
        mean = (h * mask).sum(3) / mask.sum(3).clamp_min(1.)
        maximum = h.masked_fill(~active[..., None], -1e4).amax(3)
        maximum = torch.where(active.any(3)[..., None], maximum, torch.zeros_like(maximum))
        attention = self.pool_attention(h).masked_fill(~active[..., None], -1e4).softmax(3) * mask
        attention = attention / attention.sum(3, keepdim=True).clamp_min(1e-8)
        attended = (attention * h).sum(3)
        own = torch.cat([mean, maximum, attended, mask.sum(3) / 11], dim=-1)
        paired = torch.cat([own, own.flip(2)], dim=-1)
        z = paired.permute(0, 2, 3, 1).reshape(b * teams, paired.shape[-1], n)
        z = F.gelu(self.team_input(z))
        for block, norm in zip(self.team_blocks, self.team_norms):
            residual = norm(block(z).transpose(1, 2)).transpose(1, 2)
            z = z + .5 * self.dropout(F.gelu(residual))
        scores = self.owner_output(z).reshape(b, teams, n)
        owner = scores[:, 1] - scores[:, 0]
        if return_boundary:
            boundary = self.boundary_output(z).reshape(b, teams, n).transpose(1, 2)
            return owner, boundary
        return owner


def actor_owner_logit(scores, x):
    """Team log odds from the globally supervised player-control scores."""
    b, n, teams, players, _ = scores.shape
    present = x[..., 12].permute(0, 2, 3, 1).reshape(b * teams * players, 1, n)
    active = F.max_pool1d(present, 63, stride=1, padding=31).reshape(b, teams, players, n).permute(0, 3, 1, 2) > .5
    team_scores = scores[..., 0].masked_fill(~active, -1e4).logsumexp(3)
    both = active.any(3).all(2)
    return torch.where(both, team_scores[..., 1] - team_scores[..., 0], torch.zeros_like(team_scores[..., 0]))


def infer_raw(model, raw, chunk=1024, return_control=False):
    model.eval(); n = len(raw['ball']); out = np.empty(n, np.float32)
    device = next(model.parameters()).device
    boundaries = np.empty((n, 2), np.float32)
    captured = {}; control = np.empty(n, np.float32) if return_control else None
    def remember(module, inputs, output):
        captured['scores'] = output
    handle = model.pool_attention.register_forward_hook(remember) if return_control else None
    try:
        with torch.inference_mode():
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                x = crop_features(raw, start - RADIUS, stop + RADIUS)
                tensor = torch.from_numpy(x[None].astype(np.float32)).to(device)
                owner, boundary = model(tensor, return_boundary=True)
                out[start:stop] = owner[0, RADIUS:RADIUS + stop - start].cpu().numpy()
                boundaries[start:stop] = boundary[0, RADIUS:RADIUS + stop - start].cpu().numpy()
                if return_control:
                    actor = actor_owner_logit(captured['scores'], tensor)
                    control[start:stop] = actor[0, RADIUS:RADIUS + stop - start].cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    return (out, boundaries, control) if return_control else (out, boundaries)
