"""
Implementation-safe record candidate for the 10-minute / 16MB track.

This script keeps the official repository interfaces intact:
- same FineWeb shard format
- same SentencePiece-based BPB evaluation
- same "single train_gpt.py + compressed model artifact" submission shape

The design intentionally favors end-to-end reliability over speculative tricks:
- shared-depth transformer block instead of token-routed recurrence
- complete state-dict serialization and exact roundtrip validation
- explicit train/eval/selftest modes
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional for selftest-only environments
    np = None

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional for selftest-only environments
    spm = None
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def env_bool(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


def align_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", str(REPO_ROOT / "data/datasets/fineweb10B_sp1024"))
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", str(REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    log_file = os.environ.get("LOG_FILE", str(SCRIPT_DIR / "train.log"))
    artifact_path = os.environ.get("ARTIFACT_PATH", str(SCRIPT_DIR / "model.compact.ptz"))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1_200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 12))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    warmup_seq_len = int(os.environ.get("WARMUP_SEQ_LEN", 512))
    seq_warmup_steps = int(os.environ.get("SEQ_WARMUP_STEPS", 600))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.25))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    ffn_mult = float(os.environ.get("FFN_MULT", 2.75))
    num_recur_steps = int(os.environ.get("NUM_RECUR_STEPS", 8))
    min_recur_steps = int(os.environ.get("MIN_RECUR_STEPS", 6))
    recur_warmup_steps = int(os.environ.get("RECUR_WARMUP_STEPS", 2_500))
    eval_recur_steps = int(os.environ.get("EVAL_RECUR_STEPS", os.environ.get("NUM_RECUR_STEPS", 8)))
    tie_embeddings = env_bool("TIE_EMBEDDINGS", True)
    rope_base = float(os.environ.get("ROPE_BASE", 10_000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    depth_embed_scale = float(os.environ.get("DEPTH_EMBED_SCALE", 0.5))
    smear_init = float(os.environ.get("SMEAR_INIT", 0.08))

    mtp_horizons = int(os.environ.get("MTP_HORIZONS", 2))
    mtp_weight = float(os.environ.get("MTP_WEIGHT", 0.12))
    mtp_decay = float(os.environ.get("MTP_DECAY", 0.5))

    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.035))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.03))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.7))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))

    quant_mode = os.environ.get("QUANT_MODE", "int8")
    keep_float_max_numel = int(os.environ.get("KEEP_FLOAT_MAX_NUMEL", 65_536))
    int4_min_numel = int(os.environ.get("INT4_MIN_NUMEL", 131_072))
    int8_clip_percentile = float(os.environ.get("INT8_CLIP_PERCENTILE", 99.99984))
    int4_clip_percentile = float(os.environ.get("INT4_CLIP_PERCENTILE", 99.995))
    fake_quant_bits = int(os.environ.get("FAKE_QUANT_BITS", 0))
    fake_quant_start_step = int(os.environ.get("FAKE_QUANT_START_STEP", 0))
    fake_quant_min_numel = int(os.environ.get("FAKE_QUANT_MIN_NUMEL", 131_072))
    quant_keep_fp32_patterns = tuple(
        pattern
        for pattern in os.environ.get(
            "QUANT_KEEP_FP32_PATTERNS",
            "attn_scale,mlp_scale,resid_mix,q_gain,depth_embed,mtp_biases",
        ).split(",")
        if pattern
    )

    compile_model = env_bool("COMPILE_MODEL", True)
    compile_muon = env_bool("COMPILE_MUON", True)
    ttc_enabled = env_bool("TTC_ENABLED", False)
    ttc_recur_steps = int(os.environ.get("TTC_RECUR_STEPS", os.environ.get("EVAL_RECUR_STEPS", os.environ.get("NUM_RECUR_STEPS", 8))))
    ttc_top2_margin = float(os.environ.get("TTC_TOP2_MARGIN", 0.35))
    ttc_min_token_fraction = float(os.environ.get("TTC_MIN_TOKEN_FRACTION", 0.03))
    backout_after_depth = int(os.environ.get("BACKOUT_AFTER_DEPTH", 4))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.0))


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    if np is None:
        raise RuntimeError("numpy is required for shard loading")
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens % seq_len != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS must stay divisible by WORLD_SIZE*GRAD_ACCUM*SEQ_LEN, got local_tokens={local_tokens}, seq_len={seq_len}"
            )
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model_logits(model, x, args.eval_recur_steps)
                if args.ttc_enabled and args.ttc_recur_steps > args.eval_recur_steps:
                    top2 = logits.float().topk(2, dim=-1).values
                    hard = (top2[..., 0] - top2[..., 1]) < args.ttc_top2_margin
                    if hard.float().mean().item() >= args.ttc_min_token_fraction:
                        deep_logits = model_logits(model, x, args.ttc_recur_steps)
                        logits = torch.where(hard.unsqueeze(-1), deep_logits, logits)
                batch_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    reduction="mean",
                ).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "attn_scales",
    "mlp_scale",
    "mlp_scales",
    "resid_mix",
    "resid_mixes",
    "q_gain",
    "depth_embed",
    "depth_embeds",
    "skip_weight",
    "skip_weights",
    "mtp_biases",
)
INT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str], args: Hyperparameters) -> Tensor:
    if any(pattern in name for pattern in args.quant_keep_fp32_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT_STORE_DTYPE).contiguous()
    return t


def choose_quant_bits(name: str, t: Tensor, args: Hyperparameters) -> int:
    if not t.is_floating_point():
        return 0
    if t.numel() <= args.keep_float_max_numel:
        return 0
    if any(pattern in name for pattern in args.quant_keep_fp32_patterns):
        return 0
    if args.quant_mode == "int8":
        return 8
    if args.quant_mode == "int4":
        return 4 if t.ndim >= 2 else 0
    if args.quant_mode == "mixed":
        if t.ndim >= 2 and t.numel() >= args.int4_min_numel:
            return 4
        return 8
    raise ValueError(f"Unknown QUANT_MODE={args.quant_mode}")


def pack_signed_int4(q: Tensor) -> Tensor:
    flat = q.reshape(-1).to(torch.int16) + 8
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.tensor([8], dtype=flat.dtype)))
    packed = ((flat[0::2] << 4) | flat[1::2]).to(torch.uint8)
    return packed.contiguous()


def unpack_signed_int4(packed: Tensor, numel: int, device: torch.device) -> Tensor:
    packed = packed.to(device=device, dtype=torch.uint8)
    hi = (packed >> 4).to(torch.int16)
    lo = (packed & 0x0F).to(torch.int16)
    nibbles = torch.empty(packed.numel() * 2, device=device, dtype=torch.int16)
    nibbles[0::2] = hi
    nibbles[1::2] = lo
    return (nibbles[:numel] - 8).to(torch.int8)


def quantile_abs(t32: Tensor, q: float, dim: int | None = None) -> Tensor:
    if t32.numel() == 0:
        if dim is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return torch.empty((t32.shape[dim],), dtype=torch.float32)
    return torch.quantile(t32.abs(), q, dim=dim)


def quantize_float_tensor_int8(t: Tensor, q: float) -> tuple[Tensor, Tensor, dict[str, object]]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = quantile_abs(t32, q, dim=1)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        quant = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return quant, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), {"scheme": "per_row", "axis": 0, "bits": 8}
    clip_abs = float(quantile_abs(t32, q).item())
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    quant = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return quant, scale.contiguous(), {"scheme": "per_tensor", "bits": 8}


def quantize_float_tensor_int4(t: Tensor, q: float) -> tuple[Tensor, Tensor, dict[str, object]]:
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError("int4 quantization is only implemented for 2D tensors")
    clip_abs = quantile_abs(t32, q, dim=1)
    clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
    scale = (clip_abs / 7.0).clamp_min(1.0 / 7.0)
    quant = torch.clamp(torch.round(clipped / scale[:, None]), -7, 7).to(torch.int8).contiguous()
    packed = pack_signed_int4(quant)
    qmeta = {"scheme": "per_row_packed_int4", "axis": 0, "bits": 4, "shape": list(t.shape)}
    return packed, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), qmeta


def quantize_state_dict_compact(state_dict: dict[str, Tensor], args: Hyperparameters):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "compact_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["compact_payload_bytes"] += tensor_nbytes(t)
            continue

        bits = choose_quant_bits(name, t, args)
        if bits == 0:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, args)
            passthrough[name] = kept
            stats["compact_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if bits == 4:
            q, s, meta = quantize_float_tensor_int4(t, args.int4_clip_percentile / 100.0)
        else:
            q, s, meta = quantize_float_tensor_int8(t, args.int8_clip_percentile / 100.0)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        qmeta[name] = meta
        stats["compact_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "compact_symm_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_compact(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, quant in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        bits = int(meta.get("bits", 8))
        scheme = meta.get("scheme", "per_tensor")
        if bits == 4:
            shape = tuple(int(v) for v in meta["shape"])
            unpacked = unpack_signed_int4(quant, math.prod(shape), device=torch.device("cpu")).view(*shape).float()
            out[name] = (unpacked * s.float().view(shape[0], *([1] * (len(shape) - 1)))).to(dtype=dtype).contiguous()
            continue
        if scheme == "per_row" or s.ndim > 0:
            out[name] = (quant.float() * s.float().view(quant.shape[0], *([1] * (quant.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (quant.float() * float(s.item())).to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        bits = int(getattr(self, "fake_quant_bits", 0))
        min_numel = int(getattr(self, "fake_quant_min_numel", 0))
        if self.training and bits > 0 and weight.ndim == 2 and weight.numel() >= min_numel:
            levels = 127.0 if bits >= 8 else 7.0
            scale = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0 / levels) / levels
            q = torch.clamp(torch.round(weight / scale), -levels, levels)
            weight = weight + ((q * scale) - weight).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def set_fake_quant(model: nn.Module, bits: int, min_numel: int) -> None:
    target = model.module if hasattr(model, "module") else model
    for module in target.modules():
        if isinstance(module, CastedLinear):
            module.fake_quant_bits = int(bits)
            module.fake_quant_min_numel = int(min_numel)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if num_heads % num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_mult: float):
        super().__init__()
        hidden = align_up(int(dim * ffn_mult), 128)
        self.w_gate = CastedLinear(dim, hidden, bias=False)
        self.w_up = CastedLinear(dim, hidden, bias=False)
        self.w_down = CastedLinear(hidden, dim, bias=False)
        self.w_down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SharedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_mult: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLU(dim, ffn_mult)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + mlp_out
        return x, attn_out


class SmearLayer(nn.Module):
    def __init__(self, init: float):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        alpha = self.alpha.to(dtype=x.dtype).clamp(0.0, 1.0)
        return x + alpha * prev


class BackoutLayer(nn.Module):
    def __init__(self, d_model: int, init: float):
        super().__init__()
        self.gate = nn.Parameter(torch.full((d_model,), float(init), dtype=torch.float32))

    def forward(self, x: Tensor, x_early: Tensor) -> Tensor:
        scale = torch.tanh(self.gate).to(dtype=x.dtype)[None, None, :]
        return x - scale * x_early


class SharedDepthGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_mult: float,
        num_recur_steps: int,
        eval_recur_steps: int,
        tie_embeddings: bool,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        depth_embed_scale: float,
        smear_init: float,
        backout_after_depth: int,
        backout_init: float,
        mtp_horizons: int,
        mtp_weight: float,
        mtp_decay: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_recur_steps = num_recur_steps
        self.eval_recur_steps = max(eval_recur_steps, num_recur_steps)
        self.depth_embed_scale = depth_embed_scale
        self.backout_after_depth = backout_after_depth
        self.mtp_horizons = mtp_horizons
        self.mtp_weight = mtp_weight
        self.mtp_decay = mtp_decay

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearLayer(smear_init)
        self.block = SharedBlock(model_dim, num_heads, num_kv_heads, ffn_mult, rope_base, qk_gain_init)
        self.backout = BackoutLayer(model_dim, backout_init)
        self.depth_embed = nn.Parameter(torch.zeros(num_recur_steps, model_dim, dtype=torch.float32))
        self.attn_scales = nn.Parameter(torch.ones(num_recur_steps, model_dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(num_recur_steps, model_dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.zeros(num_recur_steps, 2, model_dim, dtype=torch.float32))
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_biases = nn.Parameter(torch.zeros(mtp_horizons, vocab_size, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.01 if self.tie_embeddings else 0.02)
        nn.init.normal_(self.depth_embed, mean=0.0, std=0.01)
        with torch.no_grad():
            self.resid_mix[:, 0].fill_(1.0)
            self.resid_mix[:, 1].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def hidden(self, input_ids: Tensor, recur_steps: int | None = None) -> Tensor:
        steps = self.eval_recur_steps if recur_steps is None else recur_steps
        x = self.smear(self.tok_emb(input_ids))
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        x_early = None
        for depth in range(steps):
            depth_idx = min(depth, self.num_recur_steps - 1)
            x_in = x + self.depth_embed_scale * self.depth_embed[depth_idx].to(dtype=x.dtype)[None, None, :]
            mix = self.resid_mix[depth_idx].to(dtype=x.dtype)
            x_in = mix[0][None, None, :] * x_in + mix[1][None, None, :] * x0
            h = self.block.attn(self.block.attn_norm(x_in))
            x = x_in + self.attn_scales[depth_idx].to(dtype=x.dtype)[None, None, :] * h
            m = self.block.mlp(self.block.mlp_norm(x))
            x = x + self.mlp_scales[depth_idx].to(dtype=x.dtype)[None, None, :] * m
            if depth == max(0, self.backout_after_depth - 1):
                x_early = x
            elif depth >= self.backout_after_depth and x_early is not None:
                x = self.backout(x, x_early)
        return self.final_norm(x)

    def logits(self, input_ids: Tensor, recur_steps: int | None = None) -> Tensor:
        x = self.hidden(input_ids, recur_steps=recur_steps).reshape(-1, self.tok_emb.embedding_dim)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.view(input_ids.size(0), input_ids.size(1), -1)

    def forward(self, input_ids: Tensor, target_ids: Tensor, recur_steps: int | None = None) -> Tensor:
        logits = self.logits(input_ids, recur_steps=recur_steps)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")
        if self.training and self.mtp_horizons > 0 and self.mtp_weight > 0.0:
            aux_losses = []
            for horizon in range(1, self.mtp_horizons + 1):
                if input_ids.size(1) <= horizon:
                    break
                shifted_logits = logits[:, :-horizon, :] + self.mtp_biases[horizon - 1][None, None, :].to(dtype=logits.dtype)
                shifted_targets = target_ids[:, horizon:]
                aux_losses.append(
                    (self.mtp_decay ** (horizon - 1))
                    * F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
                        shifted_targets.reshape(-1),
                        reduction="mean",
                    )
                )
            if aux_losses:
                loss = loss + self.mtp_weight * torch.stack(aux_losses).mean()
        return loss


def build_model(args: Hyperparameters, device: torch.device) -> SharedDepthGPT:
    model = SharedDepthGPT(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_mult=args.ffn_mult,
        num_recur_steps=args.num_recur_steps,
        eval_recur_steps=args.eval_recur_steps,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        depth_embed_scale=args.depth_embed_scale,
        smear_init=args.smear_init,
        backout_after_depth=args.backout_after_depth,
        backout_init=args.backout_init,
        mtp_horizons=args.mtp_horizons,
        mtp_weight=args.mtp_weight,
        mtp_decay=args.mtp_decay,
    ).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(model)
        set_fake_quant(model, 0, args.fake_quant_min_numel)
    return model


def build_optimizers(base_model: SharedDepthGPT, args: Hyperparameters) -> tuple[list[torch.optim.Optimizer], Muon]:
    block_named_params = list(base_model.block.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.extend(
        [
            base_model.depth_embed,
            base_model.attn_scales,
            base_model.mlp_scales,
            base_model.resid_mix,
            base_model.mtp_biases,
            base_model.smear.alpha,
            base_model.backout.gate,
        ]
    )
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=torch.cuda.is_available(),
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=torch.cuda.is_available(),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=torch.cuda.is_available(),
        )
        optimizers.insert(1, optimizer_head)
    return optimizers, optimizer_muon


def load_quantized_artifact(path: Path) -> dict[str, Tensor]:
    with open(path, "rb") as f:
        blob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    return dequantize_state_dict_compact(quant_state)


def train_recur_steps_for_step(args: Hyperparameters, step: int) -> int:
    min_steps = max(1, min(args.min_recur_steps, args.num_recur_steps))
    if args.recur_warmup_steps <= 0 or min_steps >= args.num_recur_steps:
        return args.num_recur_steps
    frac = min(step / args.recur_warmup_steps, 1.0)
    depth = min_steps + int(round((args.num_recur_steps - min_steps) * frac))
    return max(min_steps, min(depth, args.num_recur_steps))


def train_fake_quant_bits_for_step(args: Hyperparameters, step: int) -> int:
    if args.fake_quant_bits <= 0:
        return 0
    return args.fake_quant_bits if step >= args.fake_quant_start_step else 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def model_logits(model: nn.Module, input_ids: Tensor, recur_steps: int) -> Tensor:
    return unwrap_model(model).logits(input_ids, recur_steps=recur_steps)


def serialize_artifact(
    base_model: SharedDepthGPT,
    args: Hyperparameters,
    code_bytes: int,
    log0,
) -> tuple[int, int, int]:
    quant_obj, quant_stats = quantize_state_dict_compact(base_model.state_dict(), args)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        f.write(quant_blob)
    model_bytes = artifact_path.stat().st_size
    total_bytes = model_bytes + code_bytes
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["compact_payload_bytes"], 1)
    log0(
        f"Serialized model compact+zlib: {model_bytes} bytes "
        f"(payload:{quant_stats['compact_payload_bytes']} raw_torch:{len(quant_raw)} payload_ratio:{ratio:.2f}x)"
    )
    log0(f"Code size: {code_bytes} bytes")
    log0(f"Total submission size compact+zlib: {total_bytes} bytes")
    if total_bytes >= 16_000_000:
        raise RuntimeError(f"Submission exceeds 16,000,000 bytes: {total_bytes}")
    return model_bytes, code_bytes, total_bytes


def run_selftest(args: Hyperparameters) -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    tiny = copy.copy(args)
    tiny.vocab_size = 128
    tiny.model_dim = 64
    tiny.num_heads = 4
    tiny.num_kv_heads = 2
    tiny.ffn_mult = 2.0
    tiny.num_recur_steps = 3
    tiny.min_recur_steps = 2
    tiny.recur_warmup_steps = 4
    tiny.eval_recur_steps = 4
    tiny.tie_embeddings = True
    tiny.smear_init = 0.08
    tiny.backout_after_depth = 2
    tiny.backout_init = 0.0
    tiny.mtp_horizons = 1
    tiny.quant_mode = "mixed"
    tiny.fake_quant_bits = 4
    tiny.fake_quant_start_step = 0
    tiny.fake_quant_min_numel = 64
    tiny.keep_float_max_numel = 32
    model = build_model(tiny, device)
    set_fake_quant(model, tiny.fake_quant_bits, tiny.fake_quant_min_numel)
    model.train()
    x = torch.randint(0, tiny.vocab_size, (2, 16), device=device)
    y = torch.randint(0, tiny.vocab_size, (2, 16), device=device)
    loss = model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Selftest loss is not finite")
    loss.backward()
    state = model.state_dict()
    obj, _ = quantize_state_dict_compact(state, tiny)
    roundtrip = dequantize_state_dict_compact(obj)
    missing = set(state) ^ set(roundtrip)
    if missing:
        raise RuntimeError(f"Selftest state mismatch: {sorted(missing)}")
    fresh = build_model(tiny, device)
    fresh.load_state_dict(roundtrip, strict=True)
    with torch.inference_mode():
        ref = model.logits(x, recur_steps=tiny.eval_recur_steps)
        out = fresh.logits(x, recur_steps=tiny.eval_recur_steps)
    if ref.shape != out.shape:
        raise RuntimeError("Selftest logits shape mismatch")
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"selftest_ok loss={loss.item():.4f} max_logit_diff={max_diff:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "train", "eval", "selftest"), default="full")
    args_cli = parser.parse_args()
    args = Hyperparameters()

    global zeropower_via_newtonschulz5
    if args.compile_muon and torch.cuda.is_available():
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    if args_cli.mode == "selftest":
        run_selftest(args)
        return

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for train/eval modes; use --mode selftest for local validation")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    log_path = Path(args.log_file)
    if master_process:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    code_bytes = Path(__file__).stat().st_size
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if spm is None:
        raise RuntimeError("sentencepiece is required for train/eval modes")
    if np is None:
        raise RuntimeError("numpy is required for train/eval modes")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Expected SentencePiece .model file, got {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = build_model(args, device)
    compiled_model: nn.Module
    if args.compile_model:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False, mode="reduce-overhead")
    else:
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    optimizers, optimizer_muon = build_optimizers(base_model, args)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"architecture:shared_depth recur_steps:{args.num_recur_steps} model_dim:{args.model_dim}")
    log0(
        f"recur_schedule:min:{args.min_recur_steps} warmup_steps:{args.recur_warmup_steps} "
        f"eval_recur_steps:{args.eval_recur_steps}"
    )
    log0(f"ffn_mult:{args.ffn_mult} num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"quant_mode:{args.quant_mode} tie_embeddings:{args.tie_embeddings} "
        f"mtp_horizons:{args.mtp_horizons} mtp_weight:{args.mtp_weight}"
    )
    log0(
        f"smear_init:{args.smear_init} backout_after_depth:{args.backout_after_depth} "
        f"backout_init:{args.backout_init}"
    )
    log0(
        f"fake_quant_bits:{args.fake_quant_bits} fake_quant_start_step:{args.fake_quant_start_step} "
        f"fake_quant_min_numel:{args.fake_quant_min_numel}"
    )
    log0(
        f"ttc_enabled:{int(args.ttc_enabled)} ttc_recur_steps:{args.ttc_recur_steps} "
        f"ttc_top2_margin:{args.ttc_top2_margin} ttc_min_token_fraction:{args.ttc_min_token_fraction}"
    )
    log0(
        f"embed_lr:{args.embed_lr} head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"warmup_seq_len:{args.warmup_seq_len} seq_warmup_steps:{args.seq_warmup_steps} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"artifact_path:{args.artifact_path}")
    log0(f"seed:{args.seed}")

    if args_cli.mode == "eval":
        base_model.load_state_dict(load_quantized_artifact(Path(args.artifact_path)), strict=True)
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        model_bytes = Path(args.artifact_path).stat().st_size
        total_bytes = model_bytes + code_bytes
        log0(
            f"final_compact_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
            f"bytes_model:{model_bytes} bytes_code:{code_bytes} bytes_total:{total_bytes}"
        )
        if distributed:
            dist.destroy_process_group()
        return

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            seq_len = args.warmup_seq_len if warmup_step < args.warmup_steps // 2 else args.train_seq_len
            set_fake_quant(model, train_fake_quant_bits_for_step(args, warmup_step), args.fake_quant_min_numel)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y, train_recur_steps_for_step(args, warmup_step))
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps} seq_len:{seq_len}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        current_seq_len = args.warmup_seq_len if step < args.seq_warmup_steps else args.train_seq_len
        current_recur_steps = train_recur_steps_for_step(args, step)
        current_fake_quant_bits = train_fake_quant_bits_for_step(args, step)
        set_fake_quant(model, current_fake_quant_bits, args.fake_quant_min_numel)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, current_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, current_recur_steps)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"seq_len:{current_seq_len} recur_steps:{current_recur_steps} fake_quant_bits:{current_fake_quant_bits} "
                f"train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    set_fake_quant(model, 0, args.fake_quant_min_numel)
    model_bytes, code_bytes, total_bytes = serialize_artifact(base_model, args, code_bytes, log0)

    if distributed:
        dist.barrier()
    base_model.load_state_dict(load_quantized_artifact(Path(args.artifact_path)), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_compact_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_compact_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
        f"bytes_model:{model_bytes} bytes_code:{code_bytes} bytes_total:{total_bytes}"
    )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
