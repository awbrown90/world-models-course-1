import os
import time
import random
import threading
import argparse
from dataclasses import dataclass
from collections import deque
from contextlib import nullcontext

import cv2
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vector_quantize_pytorch import VectorQuantize
from game_library import ENV_REGISTRY


# ============================================================
# GLOBALS
# ============================================================
PALETTE_RGB = np.array(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ============================================================
# CONFIG
# ============================================================
@dataclass
class WMConfig:
    name: str
    token_dim: int
    hidden: int
    num_blocks: int
    history_len: int
    unroll_steps: int
    train_batch_size: int
    num_bg_envs: int
    num_heads: int

    lr: float = 2e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.995

    top_k: int = 3
    token_temperature: float = 0.90
    rollout_sample_uncertain_only: bool = True
    rollout_conf_thresh: float = 0.99

    self_feed_start: float = 0.10
    self_feed_max: float = 0.35
    self_feed_ramp_steps: int = 10000
    self_feed_step_bonus: float = 0.08

    prefix_halluc_prob: float = 0.50
    max_prefix_halluc: int = 2

    p_blank_history: float = 0.10
    p_randomize_frame: float = 0.08
    p_dropout_tokens: float = 0.15
    dropout_frac: float = 0.10

    late_step_weight: float = 1.25

    changed_weight: float = 8.0
    fg_weight: float = 4.0
    new_fg_weight: float = 20.0

    buffer_size: int = 60000
    vocab_size: int = 32

    eval_steps: int = 64
    health_trials: int = 4
    health_steps: int = 80
    min_fg_pixels: int = 6
    max_fg_pixels: int = 900
    min_delta_pixels: int = 2
    max_frozen_steps: int = 10


def get_capacity_config(capacity: str) -> WMConfig:
    capacity = capacity.lower()

    if capacity == "light":
        return WMConfig(
            name="light",
            token_dim=24,
            hidden=128,
            num_blocks=4,
            history_len=4,
            unroll_steps=4,
            train_batch_size=128,
            num_bg_envs=20,
            num_heads=4,
            self_feed_max=0.30,
            self_feed_ramp_steps=8000,
        )

    if capacity == "medium":
        return WMConfig(
            name="medium",
            token_dim=32,
            hidden=192,
            num_blocks=8,
            history_len=4,
            unroll_steps=6,
            train_batch_size=96,
            num_bg_envs=16,
            num_heads=6,
            self_feed_max=0.35,
            self_feed_ramp_steps=10000,
        )

    if capacity == "heavy":
        return WMConfig(
            name="heavy",
            token_dim=48,
            hidden=256,
            num_blocks=12,
            history_len=6,
            unroll_steps=8,
            train_batch_size=64,
            num_bg_envs=12,
            num_heads=8,
            top_k=5,
            rollout_conf_thresh=0.985,
            self_feed_max=0.30,
            self_feed_ramp_steps=12000,
            prefix_halluc_prob=0.60,
            max_prefix_halluc=3,
            late_step_weight=1.35,
        )

    raise ValueError(f"Unknown capacity '{capacity}'")


# ============================================================
# VQ-VAE
# ============================================================
class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=16, num_embeddings=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )
        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=num_embeddings,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    @torch.inference_mode()
    def get_tokens(self, x):
        z = self.encoder(x)
        b, c, h_t, w_t = z.shape
        z_flat = z.view(b, c, -1).transpose(1, 2)
        _, indices, _ = self.vq(z_flat)
        return indices.view(b, h_t, w_t)

    @torch.inference_mode()
    def decode_tokens(self, indices):
        if hasattr(self.vq, "codebook"):
            cb = self.vq.codebook
        elif hasattr(self.vq, "_codebook") and hasattr(self.vq._codebook, "embed"):
            cb = self.vq._codebook.embed
        else:
            cb = self.vq._codebook.codebook

        quantized = cb[indices]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


# ============================================================
# MODEL BLOCKS
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# WORLD MODELS
# ============================================================
class ConvWorldModel(nn.Module):
    def __init__(self, vocab_size, num_actions, token_dim, hidden, num_blocks, history_len, h_t, w_t):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.history_len = history_len
        self.h_t = h_t
        self.w_t = w_t

        self.token_emb = nn.Embedding(vocab_size, token_dim)
        self.action_emb = nn.Embedding(num_actions, token_dim)

        ys = torch.linspace(-1.0, 1.0, h_t).view(h_t, 1).expand(h_t, w_t)
        xs = torch.linspace(-1.0, 1.0, w_t).view(1, w_t).expand(h_t, w_t)
        coord = torch.stack([xs, ys], dim=0).unsqueeze(0)
        self.register_buffer("coord_grid", coord, persistent=False)

        in_ch = history_len * token_dim + token_dim + 2

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])
        self.token_head = nn.Conv2d(hidden, vocab_size, kernel_size=1)

    def forward(self, hist_tokens, action):
        b, t, h, w = hist_tokens.shape
        tok = self.token_emb(hist_tokens)
        tok = tok.permute(0, 1, 4, 2, 3).contiguous()
        tok = tok.view(b, t * self.token_dim, h, w)

        act = self.action_emb(action)
        act = act[:, :, None, None].expand(b, self.token_dim, h, w)
        coord = self.coord_grid.expand(b, -1, -1, -1)

        x = torch.cat([tok, act, coord], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.token_head(x)


class TransformerWorldModel(nn.Module):
    def __init__(self, vocab_size, num_actions, token_dim, hidden, num_blocks, num_heads, history_len, h_t, w_t):
        super().__init__()
        self.vocab_size = vocab_size
        self.history_len = history_len
        self.h_t = h_t
        self.w_t = w_t
        self.num_patches = h_t * w_t
        self.model_dim = hidden

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.action_emb = nn.Embedding(num_actions, hidden)

        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, hidden) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, history_len, self.num_patches, hidden) * 0.02)

        self.blocks = nn.ModuleList([TransformerBlock(hidden, num_heads=num_heads) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.token_head = nn.Linear(hidden, vocab_size)

    def forward(self, hist_tokens, action):
        b, t, h, w = hist_tokens.shape
        x = self.token_emb(hist_tokens).view(b, t, self.num_patches, self.model_dim)
        x = x + self.spatial_pos.unsqueeze(1) + self.temporal_pos[:, :t]

        act = self.action_emb(action)
        x[:, -1] = x[:, -1] + act[:, None, :]

        x = x.reshape(b, t * self.num_patches, self.model_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        last = x[:, -self.num_patches:, :]
        logits = self.token_head(last)  # [B, P, V]
        logits = logits.view(b, h, w, self.vocab_size).permute(0, 3, 1, 2).contiguous()
        return logits


class HybridWorldModel(nn.Module):
    def __init__(self, vocab_size, num_actions, token_dim, hidden, num_blocks, num_heads, history_len, h_t, w_t):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.history_len = history_len
        self.h_t = h_t
        self.w_t = w_t
        self.num_patches = h_t * w_t

        self.token_emb = nn.Embedding(vocab_size, token_dim)
        self.action_emb = nn.Embedding(num_actions, token_dim)

        ys = torch.linspace(-1.0, 1.0, h_t).view(h_t, 1).expand(h_t, w_t)
        xs = torch.linspace(-1.0, 1.0, w_t).view(1, w_t).expand(h_t, w_t)
        coord = torch.stack([xs, ys], dim=0).unsqueeze(0)
        self.register_buffer("coord_grid", coord, persistent=False)

        in_ch = history_len * token_dim + token_dim + 2
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
        )

        conv_blocks = max(2, num_blocks // 2)
        tr_blocks = max(2, num_blocks - conv_blocks)

        self.conv_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(conv_blocks)])
        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, hidden) * 0.02)
        self.tr_blocks = nn.ModuleList([TransformerBlock(hidden, num_heads=num_heads) for _ in range(tr_blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.token_head = nn.Conv2d(hidden, vocab_size, kernel_size=1)

    def forward(self, hist_tokens, action):
        b, t, h, w = hist_tokens.shape
        tok = self.token_emb(hist_tokens)
        tok = tok.permute(0, 1, 4, 2, 3).contiguous()
        tok = tok.view(b, t * self.token_dim, h, w)

        act = self.action_emb(action)
        act = act[:, :, None, None].expand(b, self.token_dim, h, w)
        coord = self.coord_grid.expand(b, -1, -1, -1)

        x = torch.cat([tok, act, coord], dim=1)
        x = self.in_proj(x)
        x = self.conv_blocks(x)

        seq = x.flatten(2).transpose(1, 2) + self.spatial_pos
        for block in self.tr_blocks:
            seq = block(seq)
        seq = self.norm(seq)

        x = seq.transpose(1, 2).view(b, -1, h, w)
        return self.token_head(x)


def build_world_model(model_type, cfg: WMConfig, vocab_size, num_actions, tok_h, tok_w):
    if model_type == "conv":
        return ConvWorldModel(
            vocab_size=vocab_size,
            num_actions=num_actions,
            token_dim=cfg.token_dim,
            hidden=cfg.hidden,
            num_blocks=cfg.num_blocks,
            history_len=cfg.history_len,
            h_t=tok_h,
            w_t=tok_w,
        )
    if model_type == "transformer":
        return TransformerWorldModel(
            vocab_size=vocab_size,
            num_actions=num_actions,
            token_dim=cfg.token_dim,
            hidden=cfg.hidden,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            history_len=cfg.history_len,
            h_t=tok_h,
            w_t=tok_w,
        )
    if model_type == "hybrid":
        return HybridWorldModel(
            vocab_size=vocab_size,
            num_actions=num_actions,
            token_dim=cfg.token_dim,
            hidden=cfg.hidden,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            history_len=cfg.history_len,
            h_t=tok_h,
            w_t=tok_w,
        )
    raise ValueError(f"Unknown model_type '{model_type}'")


# ============================================================
# REPLAY
# ============================================================
class SequenceReplayBuffer:
    def __init__(self, max_size=60000):
        self.max_size = max_size
        self.data = [None] * max_size
        self.size = 0
        self.pos = 0
        self.lock = threading.Lock()

    def add(self, hist_tokens, future_actions, future_targets):
        with self.lock:
            self.data[self.pos] = (
                hist_tokens.clone(),
                future_actions.clone(),
                future_targets.clone(),
            )
            self.pos = (self.pos + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        with self.lock:
            if self.size < batch_size:
                return None
            idx = np.random.randint(0, self.size, size=batch_size)
            return [self.data[i] for i in idx]

    def __len__(self):
        return self.size


# ============================================================
# HELPERS
# ============================================================
def load_state_dict_compat(module, path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        module.load_state_dict(obj["state_dict"])
    else:
        module.load_state_dict(obj)


def save_checkpoint(ema_model, path, train_steps, self_feed_steps):
    ckpt = {
        "ema": ema_model.state_dict(),
        "train_steps": train_steps,
        "self_feed_steps": self_feed_steps,
    }
    torch.save(ckpt, path)


def try_load_checkpoint(train_model, ema_model, path, device):
    if not os.path.exists(path):
        return 0, 0, False

    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict) or "ema" not in obj:
        ema_model.load_state_dict(obj)
        train_model.load_state_dict(obj)
        return 0, 0, True

    ema_model.load_state_dict(obj["ema"])
    train_model.load_state_dict(obj["ema"])
    return int(obj.get("train_steps", 0)), int(obj.get("self_feed_steps", 0)), True


def ema_update_(ema_model, model, decay=0.995):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k] * (1.0 - decay))


def detect_background_token(vqvae, height, width, device):
    blank = torch.zeros(1, 1, height, width, device=device)
    with torch.inference_mode():
        toks = vqvae.get_tokens(blank).view(-1)
    vals, counts = torch.unique(toks, return_counts=True)
    return int(vals[counts.argmax()].item())


def detect_token_grid(vqvae, height, width, device):
    dummy = torch.zeros(1, 1, height, width, device=device)
    with torch.inference_mode():
        toks = vqvae.get_tokens(dummy)
    return int(toks.shape[1]), int(toks.shape[2])


def to_bgr_binary(img01):
    cls = (img01 > 0.5).astype(np.int64)
    rgb = (PALETTE_RGB[cls] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_idle_action(env):
    if hasattr(env, "idle_action"):
        return int(env.idle_action)
    return 2 if getattr(env, "num_actions", 3) > 2 else 0


def make_sample_from_deques(token_deque, action_deque, history_len, unroll_steps):
    toks = list(token_deque)
    acts = list(action_deque)
    hist = torch.stack(toks[:history_len], dim=0)
    future_targets = torch.stack(toks[history_len:history_len + unroll_steps], dim=0)
    future_actions = torch.tensor(
        acts[history_len - 1:history_len - 1 + unroll_steps],
        dtype=torch.long
    )
    return hist, future_actions, future_targets


def self_feed_prob(self_feed_steps, cfg: WMConfig):
    alpha = min(1.0, self_feed_steps / float(cfg.self_feed_ramp_steps))
    return cfg.self_feed_start + alpha * (cfg.self_feed_max - cfg.self_feed_start)


def corrupt_history_tokens(hist, bg_token_id, vocab_size, cfg: WMConfig):
    out = hist.clone()
    b, t, h, w = out.shape
    device = out.device

    blank_mask = torch.rand(b, device=device) < cfg.p_blank_history
    blank_ids = torch.where(blank_mask)[0]
    for bi in blank_ids.tolist():
        ti = random.randint(0, t - 1)
        out[bi, ti].fill_(bg_token_id)

    rand_mask = torch.rand(b, device=device) < cfg.p_randomize_frame
    rand_ids = torch.where(rand_mask)[0]
    for bi in rand_ids.tolist():
        ti = random.randint(0, t - 1)
        out[bi, ti] = torch.randint(
            low=0, high=vocab_size, size=(h, w), device=device, dtype=out.dtype
        )

    drop_mask = torch.rand(b, device=device) < cfg.p_dropout_tokens
    drop_ids = torch.where(drop_mask)[0]
    num_drop = max(1, int(h * w * cfg.dropout_frac))
    for bi in drop_ids.tolist():
        ti = random.randint(0, t - 1)
        flat = out[bi, ti].view(-1)
        idx = torch.randperm(flat.numel(), device=device)[:num_drop]
        flat[idx] = bg_token_id

    return out


def sample_tokens_from_logits(logits, cfg: WMConfig):
    b, v, h, w = logits.shape
    flat = logits.permute(0, 2, 3, 1).reshape(-1, v) / cfg.token_temperature

    probs = F.softmax(flat, dim=-1)
    maxp, argmax = probs.max(dim=-1)
    pred = argmax.clone()

    if cfg.rollout_sample_uncertain_only:
        uncertain = maxp < cfg.rollout_conf_thresh
        if uncertain.any():
            sel = flat[uncertain]
            kk = min(cfg.top_k, sel.shape[-1])
            topv, topi = torch.topk(sel, k=kk, dim=-1)
            sub_probs = F.softmax(topv, dim=-1)
            picks = torch.multinomial(sub_probs, 1).squeeze(1)
            sampled = topi.gather(1, picks[:, None]).squeeze(1)
            pred[uncertain] = sampled
    else:
        kk = min(cfg.top_k, flat.shape[-1])
        topv, topi = torch.topk(flat, k=kk, dim=-1)
        sub_probs = F.softmax(topv, dim=-1)
        picks = torch.multinomial(sub_probs, 1).squeeze(1)
        pred = topi.gather(1, picks[:, None]).squeeze(1)

    return pred.view(b, h, w)


def choose_training_action(env_name, env, cur_action, steps_left):
    if hasattr(env, "sample_random_action"):
        return env.sample_random_action(cur_action, steps_left)

    if steps_left <= 0:
        n = env.num_actions

        if env_name == "pong" and n == 3:
            probs = [0.3, 0.3, 0.4]
            cur_action = int(np.random.choice([0, 1, 2], p=probs))
            steps_left = int(np.random.randint(2, 7))

        elif env_name == "platformer" and n == 4:
            probs = [0.25, 0.25, 0.35, 0.15]
            cur_action = int(np.random.choice([0, 1, 2, 3], p=probs))
            steps_left = int(np.random.randint(2, 7))

        elif env_name == "space_invaders" and n == 6:
            probs = [0.12, 0.12, 0.16, 0.12, 0.24, 0.24]
            cur_action = int(np.random.choice(np.arange(6), p=probs))
            steps_left = int(np.random.randint(1, 5))

        else:
            cur_action = int(np.random.randint(0, n))
            steps_left = int(np.random.randint(1, 5))

    return cur_action, steps_left - 1


def prime_active(keys, env_name: str):
    if env_name == "space_invaders":
        return keys[pygame.K_TAB] or keys[pygame.K_p], "TAB/P"
    return keys[pygame.K_SPACE] or keys[pygame.K_p], "SPACE/P"


def autoplay_action(env_name, env, cur_action, steps_left):
    return choose_training_action(env_name, env, cur_action, steps_left)


@torch.inference_mode()
def compute_token_metrics(pred_tokens, target_tokens, prev_tokens):
    token_acc = (pred_tokens == target_tokens).float().mean().item()
    changed = (target_tokens != prev_tokens)
    if changed.any():
        changed_acc = (pred_tokens[changed] == target_tokens[changed]).float().mean().item()
    else:
        changed_acc = 1.0
    return token_acc, changed_acc


@torch.inference_mode()
def eval_teacher_and_free_rollout(model, vqvae, env_name, env_cls, device, cfg: WMConfig, num_actions):
    env = env_cls()
    idle_action = get_idle_action(env)

    # Bootstrap with GT frames
    hist_frames = [env.reset()]
    hist_actions = []
    cur_action = idle_action
    hold_steps = 0

    for _ in range(cfg.history_len - 1):
        cur_action, hold_steps = choose_training_action(env_name, env, cur_action, hold_steps)
        f = env.step(cur_action)
        hist_frames.append(f)
        hist_actions.append(cur_action)

    hist_tensor = torch.from_numpy(np.stack(hist_frames, axis=0)).to(device)
    gt_hist_tokens = vqvae.get_tokens(hist_tensor).unsqueeze(0)
    dream_hist_tokens = gt_hist_tokens.clone()

    tf_token_accs = []
    tf_changed_accs = []
    tf_fg_maes = []

    fr_token_accs = []
    fr_changed_accs = []
    fr_fg_maes = []

    fr_alive_steps = 0
    fr_frozen_run = 0
    fr_prev_bin = None

    for _ in range(cfg.eval_steps):
        cur_action, hold_steps = choose_training_action(env_name, env, cur_action, hold_steps)
        gt_frame = env.step(cur_action)
        gt_frame_t = torch.from_numpy(gt_frame).unsqueeze(0).to(device)
        gt_next_tokens = vqvae.get_tokens(gt_frame_t)

        action_t = torch.tensor([cur_action], dtype=torch.long, device=device)

        # teacher-forced
        logits_tf = model(gt_hist_tokens, action_t)
        pred_tf = sample_tokens_from_logits(logits_tf, cfg)

        tok_acc, ch_acc = compute_token_metrics(pred_tf, gt_next_tokens, gt_hist_tokens[:, -1])
        tf_token_accs.append(tok_acc)
        tf_changed_accs.append(ch_acc)

        pred_tf_pix = vqvae.decode_tokens(pred_tf)
        tf_fg_maes.append(abs((pred_tf_pix > 0.5).float().sum().item() - (gt_frame_t > 0.5).float().sum().item()))

        # free rollout
        logits_fr = model(dream_hist_tokens, action_t)
        pred_fr = sample_tokens_from_logits(logits_fr, cfg)

        tok_acc, ch_acc = compute_token_metrics(pred_fr, gt_next_tokens, dream_hist_tokens[:, -1])
        fr_token_accs.append(tok_acc)
        fr_changed_accs.append(ch_acc)

        pred_fr_pix = vqvae.decode_tokens(pred_fr)
        fr_fg_maes.append(abs((pred_fr_pix > 0.5).float().sum().item() - (gt_frame_t > 0.5).float().sum().item()))

        cur_bin = (pred_fr_pix > 0.5).float()
        fg = cur_bin.sum().item()
        healthy_fg = (fg >= cfg.min_fg_pixels) and (fg <= cfg.max_fg_pixels)

        if fr_prev_bin is None:
            delta = cfg.min_delta_pixels + 1
        else:
            delta = (cur_bin != fr_prev_bin).float().sum().item()

        if delta < cfg.min_delta_pixels:
            fr_frozen_run += 1
        else:
            fr_frozen_run = 0

        healthy = healthy_fg and (fr_frozen_run < cfg.max_frozen_steps)
        if healthy:
            fr_alive_steps += 1

        fr_prev_bin = cur_bin
        gt_hist_tokens = torch.cat([gt_hist_tokens[:, 1:], gt_next_tokens[:, None]], dim=1)
        dream_hist_tokens = torch.cat([dream_hist_tokens[:, 1:], pred_fr[:, None]], dim=1)

    return {
        "tf_token_acc": float(np.mean(tf_token_accs)) if tf_token_accs else 0.0,
        "tf_changed_acc": float(np.mean(tf_changed_accs)) if tf_changed_accs else 0.0,
        "tf_fg_mae": float(np.mean(tf_fg_maes)) if tf_fg_maes else 0.0,
        "fr_token_acc": float(np.mean(fr_token_accs)) if fr_token_accs else 0.0,
        "fr_changed_acc": float(np.mean(fr_changed_accs)) if fr_changed_accs else 0.0,
        "fr_fg_mae": float(np.mean(fr_fg_maes)) if fr_fg_maes else 0.0,
        "fr_alive_frac": fr_alive_steps / max(cfg.eval_steps, 1),
    }


def classify_plateau(eval_history, eval_stats):
    if len(eval_history) < 6:
        return "warming_up", 0.0

    prev = list(eval_history)[-6:-3]
    recent = list(eval_history)[-3:]

    prev_best = max(prev)
    recent_best = max(recent)
    delta = recent_best - prev_best

    tf_changed = eval_stats["tf_changed_acc"]
    fr_changed = eval_stats["fr_changed_acc"]

    if delta > 0.03:
        return "improving", delta
    if delta > 0.01:
        return "slow_improving", delta

    if tf_changed >= 0.98 and fr_changed >= 0.94:
        return "plateaued_near_limit", delta

    if tf_changed < 0.94 and fr_changed < 0.85:
        return "likely_capacity_limited", delta

    if tf_changed >= 0.96 and fr_changed < 0.88:
        return "rollout_limited", delta

    return "plateaued", delta


# ============================================================
# TRAINER
# ============================================================
def background_trainer(model, ema_model, ema_lock, optimizer, buffer, device, bg_token_id, cfg: WMConfig, stats):
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    print(f"[TRAIN] started on {device} with batch_size={cfg.train_batch_size}")

    while stats["running"]:
        batch = buffer.sample(cfg.train_batch_size)
        if batch is None:
            time.sleep(0.01)
            continue

        b_hist = torch.stack([x[0] for x in batch], dim=0).to(device, non_blocking=True)
        b_act = torch.stack([x[1] for x in batch], dim=0).to(device, non_blocking=True)
        b_tgt = torch.stack([x[2] for x in batch], dim=0).to(device, non_blocking=True)

        cur_hist = corrupt_history_tokens(b_hist, bg_token_id, cfg.vocab_size, cfg)
        p_self = self_feed_prob(stats["self_feed_steps"], cfg)

        prefix_steps = 0
        if random.random() < cfg.prefix_halluc_prob:
            prefix_steps = random.randint(1, min(cfg.max_prefix_halluc, cfg.unroll_steps - 1))

        if prefix_steps > 0:
            with torch.no_grad():
                for j in range(prefix_steps):
                    a_pref = b_act[:, j]
                    pref_logits = ema_model(cur_hist, a_pref)
                    pref_pred = sample_tokens_from_logits(pref_logits, cfg)
                    cur_hist = torch.cat([cur_hist[:, 1:], pref_pred[:, None]], dim=1)

        total_loss = 0.0
        total_step_weight = 0.0

        first_step_pred = None
        first_step_target = None
        first_step_prev = None

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            for k in range(prefix_steps, cfg.unroll_steps):
                a = b_act[:, k]
                target = b_tgt[:, k]
                prev = cur_hist[:, -1]

                logits = model(cur_hist, a)

                ce_map = F.cross_entropy(logits, target, reduction="none")
                changed = (target != prev).float()
                fg = (target != bg_token_id).float()
                new_fg = ((fg > 0.5) & (prev == bg_token_id)).float()

                weights = (
                    1.0
                    + cfg.changed_weight * changed
                    + cfg.fg_weight * fg
                    + cfg.new_fg_weight * new_fg
                )

                step_loss_per = (ce_map * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
                step_loss = step_loss_per.mean()

                step_w = (cfg.late_step_weight ** (k - prefix_steps))
                total_loss = total_loss + step_w * step_loss
                total_step_weight += step_w

                with torch.no_grad():
                    pred_tokens = sample_tokens_from_logits(logits.detach(), cfg)

                    if first_step_pred is None:
                        first_step_pred = pred_tokens.clone()
                        first_step_target = target.clone()
                        first_step_prev = prev.clone()

                    pk = min(cfg.self_feed_max, p_self + cfg.self_feed_step_bonus * (k - prefix_steps))
                    use_pred = (torch.rand(pred_tokens.shape[0], device=device) < pk)
                    mixed_next = torch.where(use_pred[:, None, None], pred_tokens, target)

                cur_hist = torch.cat([cur_hist[:, 1:], mixed_next[:, None]], dim=1)

            loss = total_loss / max(total_step_weight, 1e-8)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with ema_lock:
            ema_update_(ema_model, model, decay=cfg.ema_decay)

        with torch.no_grad():
            token_acc = (first_step_pred == first_step_target).float().mean().item()
            changed_mask = (first_step_target != first_step_prev)
            if changed_mask.any():
                changed_acc = (first_step_pred[changed_mask] == first_step_target[changed_mask]).float().mean().item()
            else:
                changed_acc = 1.0

        stats["loss"] = 0.95 * stats["loss"] + 0.05 * float(loss.item())
        stats["token_acc"] = 0.95 * stats["token_acc"] + 0.05 * token_acc
        stats["changed_acc"] = 0.95 * stats["changed_acc"] + 0.05 * changed_acc
        stats["self_feed_p"] = p_self
        stats["train_steps"] += 1
        stats["self_feed_steps"] += 1

        time.sleep(0.002)


# ============================================================
# COLLECTOR
# ============================================================
def background_collector(env_name, env_cls, vqvae_bg, buffer, device, cfg: WMConfig, stats):
    envs = [env_cls() for _ in range(cfg.num_bg_envs)]
    token_deques = [deque(maxlen=cfg.history_len + cfg.unroll_steps) for _ in range(cfg.num_bg_envs)]
    action_deques = [deque(maxlen=cfg.history_len + cfg.unroll_steps - 1) for _ in range(cfg.num_bg_envs)]

    idle_action = get_idle_action(envs[0])
    cur_actions = np.full(cfg.num_bg_envs, idle_action, dtype=np.int64)
    hold_steps = np.zeros(cfg.num_bg_envs, dtype=np.int32)

    all_frames = []
    all_actions = []

    for i, env in enumerate(envs):
        frames = [env.reset()]
        acts = []
        for _ in range(cfg.history_len + cfg.unroll_steps - 1):
            cur_actions[i], hold_steps[i] = choose_training_action(env_name, env, cur_actions[i], hold_steps[i])
            a = int(cur_actions[i])
            f = env.step(a)
            frames.append(f)
            acts.append(a)

        all_frames.extend(frames)
        all_actions.append(acts)

    batch_frames = torch.from_numpy(np.stack(all_frames, axis=0)).to(device, non_blocking=True)
    with torch.inference_mode():
        all_tokens = vqvae_bg.get_tokens(batch_frames).cpu()

    ptr = 0
    seq_len = cfg.history_len + cfg.unroll_steps
    for i in range(cfg.num_bg_envs):
        toks_i = all_tokens[ptr:ptr + seq_len]
        ptr += seq_len
        for t in toks_i:
            token_deques[i].append(t.clone())
        for a in all_actions[i]:
            action_deques[i].append(a)

        hist, future_actions, future_targets = make_sample_from_deques(
            token_deques[i], action_deques[i], cfg.history_len, cfg.unroll_steps
        )
        buffer.add(hist, future_actions, future_targets)

    print(f"[COLLECT] started with {cfg.num_bg_envs} envs")

    while stats["running"]:
        next_frames = []
        chosen_actions = []

        for i, env in enumerate(envs):
            cur_actions[i], hold_steps[i] = choose_training_action(env_name, env, cur_actions[i], hold_steps[i])
            a = int(cur_actions[i])
            f = env.step(a)
            chosen_actions.append(a)
            next_frames.append(f)

        torch_frames = torch.from_numpy(np.stack(next_frames, axis=0)).to(device, non_blocking=True)
        with torch.inference_mode():
            next_tokens = vqvae_bg.get_tokens(torch_frames).cpu()

        for i in range(cfg.num_bg_envs):
            token_deques[i].append(next_tokens[i].clone())
            action_deques[i].append(int(chosen_actions[i]))

            if len(token_deques[i]) == cfg.history_len + cfg.unroll_steps and len(action_deques[i]) == cfg.history_len + cfg.unroll_steps - 1:
                hist, future_actions, future_targets = make_sample_from_deques(
                    token_deques[i], action_deques[i], cfg.history_len, cfg.unroll_steps
                )
                buffer.add(hist, future_actions, future_targets)

        stats["collector_samples"] += cfg.num_bg_envs
        time.sleep(0.003)


# ============================================================
# MAIN
# ============================================================
def run_interactive_wm(env_name: str, capacity: str, model_type: str):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{env_name}'. Choices: {list(ENV_REGISTRY.keys())}")

    cfg = get_capacity_config(capacity)
    env_cls = ENV_REGISTRY[env_name]
    env = env_cls()

    if not hasattr(env, "num_actions"):
        raise ValueError(f"{env_cls.__name__} must define self.num_actions")

    num_actions = int(env.num_actions)
    height = int(env.height)
    width = int(env.width)

    vq_path = os.path.join("models", "vqvae-models", f"vqvae_{env_name}_live.pth")
    wm_dir = os.path.join("models", "wm-models")
    os.makedirs(wm_dir, exist_ok=True)

    wm_path = os.path.join(wm_dir, f"wm_{env_name}_{model_type}_{cfg.name}_live.pth")
    best_wm_path = os.path.join(wm_dir, f"wm_{env_name}_{model_type}_{cfg.name}_live_best.pth")

    stats = {
        "running": True,
        "loss": 1.0,
        "train_steps": 0,
        "self_feed_steps": 0,
        "token_acc": 0.0,
        "changed_acc": 0.0,
        "collector_samples": 0,
        "self_feed_p": 0.0,
        "bg_token_id": 0,
        "tf_token_acc": 0.0,
        "tf_changed_acc": 0.0,
        "tf_fg_mae": 0.0,
        "fr_token_acc": 0.0,
        "fr_changed_acc": 0.0,
        "fr_fg_mae": 0.0,
        "fr_alive_frac": 0.0,
        "best_score": -1e9,
        "plateau_status": "warming_up",
        "plateau_delta": 0.0,
    }
    eval_history = deque(maxlen=8)

    print(f"Using device: {DEVICE}")
    print(f"Environment: {env_name} | actions={num_actions} | size={height}x{width}")
    print(
        f"Model: {model_type} | Capacity: {cfg.name} | token_dim={cfg.token_dim} | "
        f"hidden={cfg.hidden} | blocks={cfg.num_blocks} | history={cfg.history_len} | "
        f"unroll={cfg.unroll_steps}"
    )

    if not os.path.exists(vq_path):
        raise FileNotFoundError(
            f"Could not find VQ-VAE weights at:\n  {vq_path}\n"
            f"Train it first with:\n  python interactive_vqvae.py --env {env_name}"
        )

    print("Loading VQ-VAE (UI)...")
    vqvae_ui = VQVAE().to(DEVICE)
    load_state_dict_compat(vqvae_ui, vq_path, DEVICE)
    vqvae_ui.eval()
    for p in vqvae_ui.parameters():
        p.requires_grad_(False)

    print("Loading VQ-VAE (collector)...")
    vqvae_bg = VQVAE().to(DEVICE)
    load_state_dict_compat(vqvae_bg, vq_path, DEVICE)
    vqvae_bg.eval()
    for p in vqvae_bg.parameters():
        p.requires_grad_(False)

    tok_h, tok_w = detect_token_grid(vqvae_ui, height, width, DEVICE)
    bg_token_id = detect_background_token(vqvae_ui, height, width, DEVICE)
    stats["bg_token_id"] = bg_token_id

    print(f"Detected token grid: {tok_h}x{tok_w}")
    print(f"Detected background token id: {bg_token_id}")

    print("Initializing world model...")
    train_model = build_world_model(
        model_type=model_type,
        cfg=cfg,
        vocab_size=cfg.vocab_size,
        num_actions=num_actions,
        tok_h=tok_h,
        tok_w=tok_w,
    ).to(DEVICE)

    ema_model = build_world_model(
        model_type=model_type,
        cfg=cfg,
        vocab_size=cfg.vocab_size,
        num_actions=num_actions,
        tok_h=tok_h,
        tok_w=tok_w,
    ).to(DEVICE)

    loaded = False
    try:
        resumed_train_steps, resumed_self_feed_steps, loaded = try_load_checkpoint(train_model, ema_model, wm_path, DEVICE)
        stats["train_steps"] = resumed_train_steps
        stats["self_feed_steps"] = 0
        if loaded:
            print(f"Resumed model weights from {wm_path} at train_steps={resumed_train_steps}")
            print("Self-feed schedule reset to 0 for a gentle ramp.")
        else:
            ema_model.load_state_dict(train_model.state_dict())
    except RuntimeError as e:
        print(f"[WARN] Could not load checkpoint due to shape mismatch or architecture change:\n{e}")
        print("[WARN] Starting from scratch for this env/model/capacity.")
        ema_model.load_state_dict(train_model.state_dict())

    optimizer = optim.AdamW(train_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    buffer = SequenceReplayBuffer(max_size=cfg.buffer_size)
    ema_lock = threading.Lock()

    train_thread = threading.Thread(
        target=background_trainer,
        args=(train_model, ema_model, ema_lock, optimizer, buffer, DEVICE, bg_token_id, cfg, stats),
        daemon=True
    )
    collect_thread = threading.Thread(
        target=background_collector,
        args=(env_name, env_cls, vqvae_bg, buffer, DEVICE, cfg, stats),
        daemon=True
    )

    idle_action = get_idle_action(env)
    vis_token_deque = deque(maxlen=cfg.history_len + cfg.unroll_steps)
    vis_action_deque = deque(maxlen=cfg.history_len + cfg.unroll_steps - 1)

    boot_frames = [env.reset()]
    boot_actions = []
    for _ in range(cfg.history_len + cfg.unroll_steps - 1):
        f = env.step(idle_action)
        boot_frames.append(f)
        boot_actions.append(idle_action)

    boot_tensor = torch.from_numpy(np.stack(boot_frames, axis=0)).to(DEVICE)
    with torch.inference_mode():
        boot_tokens = vqvae_ui.get_tokens(boot_tensor).cpu()

    for t in boot_tokens:
        vis_token_deque.append(t.clone())
    for a in boot_actions:
        vis_action_deque.append(a)

    if len(vis_token_deque) == cfg.history_len + cfg.unroll_steps:
        hist, future_actions, future_targets = make_sample_from_deques(
            vis_token_deque, vis_action_deque, cfg.history_len, cfg.unroll_steps
        )
        buffer.add(hist, future_actions, future_targets)

    gt_hist_tokens = torch.stack(list(vis_token_deque)[-cfg.history_len:], dim=0).unsqueeze(0).to(DEVICE)
    dream_hist_tokens = gt_hist_tokens.clone()

    with torch.inference_mode():
        init_pred_pixels = vqvae_ui.decode_tokens(gt_hist_tokens[:, -1])
    last_pred_pixels = init_pred_pixels.detach().cpu().numpy()
    last_pred_token = gt_hist_tokens[:, -1].detach().clone()

    train_thread.start()
    collect_thread.start()

    pygame.init()
    render_h = 512
    render_w = int(render_h * (width / height))
    screen = pygame.display.set_mode((render_w * 2, render_h))
    pygame.display.set_caption(f"Interactive WM - {env_name} [{model_type}/{cfg.name}]")
    clock = pygame.time.Clock()

    auto_play = False
    auto_action = idle_action
    auto_steps_left = 0

    step = 0
    running = True
    _, prime_hint = prime_active(pygame.key.get_pressed(), env_name)

    print("\n" + "=" * 66)
    print(env.get_instructions())
    print("R = toggle autoplay")
    print(f"PRIME = {prime_hint}")
    print(f"VQ-VAE: {vq_path}")
    print(f"WM latest: {wm_path}")
    print(f"WM best:   {best_wm_path}")
    print("=" * 66 + "\n")

    try:
        while running and stats["running"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    stats["running"] = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                        stats["running"] = False
                    elif event.key == pygame.K_r:
                        auto_play = not auto_play

            keys = pygame.key.get_pressed()
            priming, prime_hint = prime_active(keys, env_name)

            if auto_play:
                auto_action, auto_steps_left = autoplay_action(env_name, env, auto_action, auto_steps_left)
                action = auto_action
            else:
                action = env.get_action(keys)

            next_frame = env.step(action)
            next_frame_t = torch.from_numpy(next_frame).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                next_target_token = vqvae_ui.get_tokens(next_frame_t)

            vis_token_deque.append(next_target_token.squeeze(0).detach().cpu())
            vis_action_deque.append(int(action))
            if len(vis_token_deque) == cfg.history_len + cfg.unroll_steps and len(vis_action_deque) == cfg.history_len + cfg.unroll_steps - 1:
                hist, future_actions, future_targets = make_sample_from_deques(
                    vis_token_deque, vis_action_deque, cfg.history_len, cfg.unroll_steps
                )
                buffer.add(hist, future_actions, future_targets)

            got_ema = ema_lock.acquire(blocking=False)
            if got_ema:
                try:
                    with torch.inference_mode():
                        action_t = torch.tensor([action], dtype=torch.long, device=DEVICE)
                        logits = ema_model(dream_hist_tokens, action_t)
                        pred_next_token = sample_tokens_from_logits(logits, cfg)
                        pred_pixels = vqvae_ui.decode_tokens(pred_next_token)

                        last_pred_pixels = pred_pixels.detach().cpu().numpy()
                        last_pred_token = pred_next_token.detach().clone()
                finally:
                    ema_lock.release()
            else:
                pred_pixels = torch.from_numpy(last_pred_pixels).to(DEVICE)
                pred_next_token = last_pred_token

            gt_hist_tokens = torch.cat([gt_hist_tokens[:, 1:], next_target_token[:, None]], dim=1)
            if priming:
                dream_hist_tokens = gt_hist_tokens.clone()
            else:
                dream_hist_tokens = torch.cat([dream_hist_tokens[:, 1:], pred_next_token[:, None]], dim=1)

            gt_img_bgr = to_bgr_binary(next_frame.squeeze())
            pred_img_bgr = to_bgr_binary(pred_pixels.squeeze().detach().cpu().numpy())

            gt_up = cv2.resize(gt_img_bgr, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
            pred_up = cv2.resize(pred_img_bgr, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

            cv2.putText(gt_up, f"Ground Truth - {env_name}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(gt_up, f"Replay: {len(buffer)}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(gt_up, f"Collector: {stats['collector_samples']}", (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(gt_up, f"Prime: {prime_hint}", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(gt_up, f"Autoplay: {'ON' if auto_play else 'OFF'}", (14, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255) if auto_play else (180, 180, 180), 2)
            cv2.putText(gt_up, f"Model: {model_type}/{cfg.name}", (14, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            status_text = "PRIMING" if priming else "DREAMING"
            status_color = (0, 255, 0) if priming else (0, 0, 255)
            cv2.putText(pred_up, status_text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
            cv2.putText(pred_up, f"Loss: {stats['loss']:.4f}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(pred_up, f"GPU steps: {stats['train_steps']}", (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(pred_up, f"Train tok/chg: {stats['token_acc']:.3f}/{stats['changed_acc']:.3f}", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(pred_up, f"TF tok/chg: {stats['tf_token_acc']:.3f}/{stats['tf_changed_acc']:.3f}", (14, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(pred_up, f"FR tok/chg: {stats['fr_token_acc']:.3f}/{stats['fr_changed_acc']:.3f}", (14, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(pred_up, f"TF/FR fgMAE: {stats['tf_fg_mae']:.1f}/{stats['fr_fg_mae']:.1f}", (14, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(pred_up, f"FR alive: {stats['fr_alive_frac']:.3f}", (14, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(pred_up, f"Self-feed p: {stats['self_feed_p']:.3f}", (14, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            plateau_color = (255, 255, 255)
            if stats["plateau_status"] in ("improving", "slow_improving"):
                plateau_color = (0, 255, 255)
            elif stats["plateau_status"] == "likely_capacity_limited":
                plateau_color = (0, 128, 255)
            elif stats["plateau_status"] == "plateaued_near_limit":
                plateau_color = (255, 255, 0)
            elif stats["plateau_status"] == "rollout_limited":
                plateau_color = (255, 128, 0)

            cv2.putText(pred_up, f"Status: {stats['plateau_status']}", (14, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.50, plateau_color, 2)
            cv2.putText(pred_up, f"Plateau d: {stats['plateau_delta']:.4f}", (14, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.50, plateau_color, 2)
            cv2.putText(pred_up, f"Frame: {step}", (14, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 0), 2)

            combined_bgr = np.hstack((gt_up, pred_up))
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            combined_rgb = np.transpose(combined_rgb, (1, 0, 2))
            screen.blit(pygame.surfarray.make_surface(combined_rgb), (0, 0))
            pygame.display.flip()

            clock.tick(30)
            step += 1

            if step % 1000 == 0:
                with ema_lock:
                    eval_stats = eval_teacher_and_free_rollout(
                        ema_model,
                        vqvae_ui,
                        env_name,
                        env_cls,
                        DEVICE,
                        cfg,
                        num_actions,
                    )

                    stats["tf_token_acc"] = eval_stats["tf_token_acc"]
                    stats["tf_changed_acc"] = eval_stats["tf_changed_acc"]
                    stats["tf_fg_mae"] = eval_stats["tf_fg_mae"]
                    stats["fr_token_acc"] = eval_stats["fr_token_acc"]
                    stats["fr_changed_acc"] = eval_stats["fr_changed_acc"]
                    stats["fr_fg_mae"] = eval_stats["fr_fg_mae"]
                    stats["fr_alive_frac"] = eval_stats["fr_alive_frac"]

                    save_checkpoint(
                        ema_model,
                        wm_path,
                        stats["train_steps"],
                        stats["self_feed_steps"],
                    )

                    score = (
                        2.5 * stats["fr_changed_acc"]
                        + 1.5 * stats["tf_changed_acc"]
                        + 1.0 * stats["fr_token_acc"]
                        + 0.5 * stats["fr_alive_frac"]
                        - 0.002 * stats["fr_fg_mae"]
                        - 0.001 * stats["tf_fg_mae"]
                    )
                    eval_history.append(score)

                    status, delta = classify_plateau(eval_history, eval_stats)
                    stats["plateau_status"] = status
                    stats["plateau_delta"] = delta

                    if score > stats["best_score"]:
                        stats["best_score"] = score
                        save_checkpoint(
                            ema_model,
                            best_wm_path,
                            stats["train_steps"],
                            stats["self_feed_steps"],
                        )

                print(
                    f"[step {step}] "
                    f"loss={stats['loss']:.4f} "
                    f"tok_acc={stats['token_acc']:.3f} "
                    f"changed_acc={stats['changed_acc']:.3f} "
                    f"tf_tok={stats['tf_token_acc']:.3f} "
                    f"tf_chg={stats['tf_changed_acc']:.3f} "
                    f"fr_tok={stats['fr_token_acc']:.3f} "
                    f"fr_chg={stats['fr_changed_acc']:.3f} "
                    f"fr_fg_mae={stats['fr_fg_mae']:.1f} "
                    f"fr_alive={stats['fr_alive_frac']:.3f} "
                    f"self_feed={stats['self_feed_p']:.3f} "
                    f"status={stats['plateau_status']} "
                    f"plateau_delta={stats['plateau_delta']:.4f} "
                    f"replay={len(buffer)} "
                    f"collector={stats['collector_samples']}"
                )

    finally:
        stats["running"] = False
        train_thread.join(timeout=2.0)
        collect_thread.join(timeout=2.0)

        with ema_lock:
            save_checkpoint(
                ema_model,
                wm_path,
                stats["train_steps"],
                stats["self_feed_steps"],
            )

        print(f"\nSaved latest WM to {wm_path}")
        print(f"Best WM path: {best_wm_path}")
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run an interactive world model for a specific arcade environment.")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="Which environment to load, e.g. pong, platformer, space_invaders"
    )
    parser.add_argument(
        "--capacity",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Capacity preset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="conv",
        choices=["conv", "transformer", "hybrid"],
        help="World model architecture"
    )
    args = parser.parse_args()

    run_interactive_wm(args.env, args.capacity, args.model)