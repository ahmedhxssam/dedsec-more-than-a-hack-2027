"""Native-frame joint controller/ball inference with explicit geometric feedback.

Inspired by BallRadar's controller-conditioned reconstruction, not a copy of its
code or possession convention. Observations, estimates and missingness remain
separate. No nearest player is assigned a positive control label.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .vision_player_temporal import features as player_features, TrackTemporalBlock, SocialAttention
from .temporal_ownership import trajectory_features
from .sprint_model_interpolation import interpolated_features

VERSION = 'joint-controller-ball-v1'
PLAYER_CHANNELS = 35
TEAM_CHANNELS = 122
PLAYER_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
BALL_DILATIONS = (1, 2, 4, 8, 16, 32)
FEEDBACK_DILATIONS = (1, 2, 4, 8, 16)
OWNER_DILATIONS = (1, 2, 4, 8, 16, 32)
RADIUS = max(sum(PLAYER_DILATIONS[:5]), 125) + sum(PLAYER_DILATIONS[5:] + BALL_DILATIONS + FEEDBACK_DILATIONS + OWNER_DILATIONS)
FEATURE_CONTEXT = 250


def features(raw):
    ball, players, team, pitch = [np.asarray(raw[k]) for k in ('ball', 'players', 'team', 'pitch')]
    n = len(ball); ix = np.arange(n); observed = np.isfinite(ball).all(1)
    prev = np.maximum.accumulate(np.where(observed, ix, -n))
    nxt = np.minimum.accumulate(np.where(observed, ix, 2*n)[::-1])[::-1]
    eligible = (prev >= 0) & (nxt < n) & (nxt-prev <= 250)
    linear = ball.copy(); known = ix[observed]
    for axis in range(2):
        if len(known) >= 2:
            linear[eligible, axis] = np.interp(ix[eligible], known, ball[known, axis])
    bounded = np.isfinite(linear).all(1)
    x = player_features(ball, players, team, pitch)
    p = x.shape[2]
    position = np.zeros((n, 2, p, 2), np.float32)
    present = np.zeros((n, 2, p), np.float32)
    for t in (0, 1):
        ids = np.flatnonzero(team == t)
        present[:, t, :len(ids)] = np.isfinite(players[:, ids]).all(2)
        position[:, t, :len(ids)] = np.nan_to_num((players[:, ids]-pitch/2)/10, nan=0)
    # Preserve stationary observations. Their kinematic confidence is separate.
    x = np.concatenate([x, present[..., None]], -1).astype(np.float32)
    geometry = np.concatenate([trajectory_features(ball, players, team, pitch),
                               interpolated_features(ball, players, team, pitch)], -1)
    return {'player': x, 'team': geometry,
            'position': position, 'base_ball': np.nan_to_num((linear-pitch/2)/10, nan=0).astype(np.float32),
            'ball_flags': np.stack([observed, bounded], -1).astype(np.float32)}


def crop_features(raw, start, stop):
    n = len(raw['ball']); a, b = max(0, start-FEATURE_CONTEXT), min(n, stop+FEATURE_CONTEXT)
    local = {k: (v[a:b] if k in ('ball', 'players') else v) for k, v in raw.items()}
    values = features(local)
    left, right = max(0, -start), max(0, stop-n)
    return {k: np.pad(v[max(0, start)-a:min(n, stop)-a],
                     ((left, right),)+((0, 0),)*(v.ndim-1)) for k, v in values.items()}


class Temporal(nn.Module):
    def __init__(self, channels, width, dilations):
        super().__init__()
        self.input = nn.Conv1d(channels, width, 1)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in dilations)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        z = F.gelu(self.input(x.transpose(1, 2)))
        for layer, norm in zip(self.layers, self.norms):
            z = z + .5*self.dropout(F.gelu(norm(layer(z).transpose(1, 2)).transpose(1, 2)))
        return z.transpose(1, 2)


def control_probabilities(score, active):
    b, n, teams, p = score.shape
    logits = score.masked_fill(~active, -1e4).reshape(b, n, teams*p)
    mask = active.reshape(b, n, teams*p).to(score.dtype)
    prob = logits.softmax(-1)*mask
    prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-8)
    return prob.reshape_as(score), logits.log_softmax(-1)


def pool(h, scores, active):
    mask = active[..., None].to(h.dtype)
    mean = (h*mask).sum(3)/mask.sum(3).clamp_min(1)
    maximum = h.masked_fill(~active[..., None], -1e4).amax(3)
    maximum = torch.where(active.any(3)[..., None], maximum, torch.zeros_like(maximum))
    attention = scores.masked_fill(~active, -1e4).softmax(3)*active.to(h.dtype)
    attention = attention/attention.sum(3, keepdim=True).clamp_min(1e-8)
    attended = (attention[..., None]*h).sum(3)
    return torch.cat([mean, maximum, attended, mask.sum(3)/11], -1)


class JointControllerBall(nn.Module):
    def __init__(self, width=32, feedback=True):
        super().__init__(); self.width = width; self.feedback = feedback
        self.input = nn.Sequential(nn.Linear(PLAYER_CHANNELS, width), nn.GELU())
        self.before = nn.ModuleList(TrackTemporalBlock(width, d) for d in PLAYER_DILATIONS[:5])
        self.social = SocialAttention(width, heads=2)
        self.after = nn.ModuleList(TrackTemporalBlock(width, d) for d in PLAYER_DILATIONS[5:])
        self.actor = nn.Linear(width, 1)
        pooled = 3*width+1
        self.ball_context = Temporal(pooled+TEAM_CHANNELS+4, 48, BALL_DILATIONS)
        self.ball_output = nn.Linear(48, 3)
        nn.init.zeros_(self.ball_output.weight); nn.init.zeros_(self.ball_output.bias)
        self.feedback_input = nn.Sequential(nn.Linear(width+7, width), nn.GELU())
        self.feedback_blocks = nn.ModuleList(TrackTemporalBlock(width, d) for d in FEEDBACK_DILATIONS)
        self.final_actor = nn.Linear(width, 1)
        self.owner = Temporal(2*pooled+TEAM_CHANNELS, 64, OWNER_DILATIONS)
        self.owner_output = nn.Linear(64, 1)

    def forward(self, values, feedback_override=None):
        x = values['player']; b, n, teams, p, _ = x.shape
        present = x[..., -1] > .5
        active = F.max_pool1d(present.permute(0,2,3,1).reshape(b*2*p,1,n).float(),
                              251, stride=1, padding=125).reshape(b,2,p,n).permute(0,3,1,2) > .5
        breaks = x[..., 33].permute(0,2,3,1).reshape(b*2*p,n)
        h = self.input(x).permute(0,2,3,1,4).reshape(b*2*p,n,self.width)
        for layer in self.before: h = layer(h, breaks)
        h = h.reshape(b,2,p,n,self.width).permute(0,3,1,2,4)
        h = self.social(h, values['position']/5, active, present)
        h = h.permute(0,2,3,1,4).reshape(b*2*p,n,self.width)
        for layer in self.after: h = layer(h, breaks)
        h = h.reshape(b,2,p,n,self.width).permute(0,3,1,2,4)
        initial_scores = self.actor(h)[...,0]
        initial_prob, initial_logp = control_probabilities(initial_scores, active)
        initial_pool = pool(h, initial_scores, active)
        weighted = initial_prob * present.to(x.dtype)
        center = (weighted[...,None]*values['position']).sum((2,3))/weighted.sum((2,3))[...,None].clamp_min(1e-8)
        flags = values['ball_flags']; base_ball = values['base_ball']
        reference = torch.where(flags[...,1:2] > .5, base_ball, center)
        ball_features = torch.cat([initial_pool.mean(2), values['team'].mean(2), reference, flags], -1)
        raw_ball = self.ball_output(self.ball_context(ball_features))
        # A controller-conditioned correction to the endpoint estimate, not a
        # physical-contact assertion. Observed samples remain exactly observed.
        inferred = reference + 4*torch.tanh(raw_ball[...,:2])
        ball = torch.where(flags[...,:1] > .5, base_ball, inferred)
        log_scale = raw_ball[...,2].clamp(-3., 2.)
        relative = (values['position']-ball[:,:,None,None])/2
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        global_feedback = torch.stack([log_scale, flags[...,0], flags[...,1]], -1)[:,:,None,None].expand(-1,-1,2,p,-1)
        geometry = torch.cat([relative, distance, global_feedback, initial_prob[...,None]], -1)
        use = self.feedback if feedback_override is None else feedback_override
        if not use: geometry = torch.zeros_like(geometry)
        refined = self.feedback_input(torch.cat([h, geometry], -1))
        refined = refined.permute(0,2,3,1,4).reshape(b*2*p,n,self.width)
        for layer in self.feedback_blocks: refined = layer(refined, breaks)
        refined = refined.reshape(b,2,p,n,self.width).permute(0,3,1,2,4)
        scores = self.final_actor(refined)[...,0]
        prob, logp = control_probabilities(scores, active)
        pooled = pool(refined, scores, active)
        paired = torch.cat([pooled, pooled.flip(2), values['team']], -1)
        z = self.owner(paired.permute(0,2,1,3).reshape(b*2,n,-1))
        logits = self.owner_output(z)[...,0].reshape(b,2,n)
        team_actor = scores.masked_fill(~active,-1e4).logsumexp(3)
        actor_logit = torch.where(active.any(3).all(2), team_actor[...,1]-team_actor[...,0], torch.zeros_like(logits[:,0]))
        return {'owner': logits[:,1]-logits[:,0], 'actor_owner': actor_logit,
                'actor_logp': logp, 'initial_actor_logp': initial_logp,
                'ball': ball, 'inferred_ball': inferred, 'log_scale': log_scale,
                'active': active, 'probability': prob}


def infer_raw(model, raw, chunk=1024):
    model.eval(); device = next(model.parameters()).device; n = len(raw['ball'])
    outputs = {k: np.empty((n, 2) if k == 'ball' else n, np.float32)
               for k in ('owner','actor_owner','ball','log_scale')}
    with torch.inference_mode():
        for start in range(0,n,chunk):
            stop = min(start+chunk,n)
            values = crop_features(raw,start-RADIUS,stop+RADIUS)
            values = {k:torch.from_numpy(v[None]).to(device) for k,v in values.items()}
            result = model(values)
            for k in outputs: outputs[k][start:stop] = result[k][0,RADIUS:RADIUS+stop-start].cpu().numpy()
    return outputs
