# Codex Safe Shared Depth

This record folder is built to be implementation-safe first: the training path, compact export path, artifact reload path, and final BPB evaluation all use one consistent state format and are validated inside `train_gpt.py`.

## What is in this version

- Shared-depth transformer: one recurrent block reused for `NUM_RECUR_STEPS`, with per-step depth embeddings and residual control vectors.
- Real compute-saving depth schedule: training ramps from `MIN_RECUR_STEPS` to `NUM_RECUR_STEPS`, so early training uses fewer recurrent passes and higher effective throughput.
- Modernized block internals: GQA attention, RoPE, QK RMS normalization, SwiGLU MLP, logit softcap.
- Local token mixing: a lightweight smear layer adds a trainable 1-token lookback before the recurrent stack.
- Optional backout path: after a configurable depth, the model can subtract a learned amount of an earlier hidden state to counter oversmoothing during deeper recurrence.
- Stable optimization: Muon for matrix parameters, Adam for embeddings and low-dimensional control tensors.
- Simple sample-efficiency helpers that are low-risk to implement:
  - sequence-length warmup (`WARMUP_SEQ_LEN`, `SEQ_WARMUP_STEPS`)
  - lightweight multi-token auxiliary loss (`MTP_HORIZONS`, `MTP_WEIGHT`)
- Integrated but experimental upside levers:
  - late fake-quant training for large linear layers (`FAKE_QUANT_BITS`, `FAKE_QUANT_START_STEP`)
  - selective test-time compute during validation (`TTC_ENABLED`, `TTC_RECUR_STEPS`)
- Full-state compact serialization:
  - every tensor in the model state dict is either quantized or stored explicitly
  - post-train roundtrip validation reloads the artifact and reruns final validation
  - default export mode is `QUANT_MODE=int8`; `mixed` and `int4` are available but optional

## Why this is safer than the original research sketch

- No late-created parameters. All trainable tensors exist before DDP/optimizer setup.
- No partial export. Embeddings, control tensors, norms, and auxiliary parameters are all included in the artifact path.
- No train/eval mismatch in the compact format. The same dequantized tensors are loaded back into the exact same module structure before final validation.
- No fake MoR savings. The model uses explicit shared recurrence without pretending token routing reduced FLOPs when it did not.
- Separate `train`, `eval`, and `selftest` modes:
  - `--mode selftest` runs on CPU and validates model build + compact roundtrip without the dataset
  - `--mode full` is the intended challenge path

## Recommended first RunPod flow

From inside the repo on a RunPod H100 machine:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-18_CodexSafeSharedDepth
bash run_on_runpod.sh
```

If you also export author metadata, the script will emit `submission.json` automatically:

```bash
AUTHOR_NAME="Your Name" \
GITHUB_ID="your-handle" \
bash run_on_runpod.sh
```

After collecting multiple logs, you can summarize variance with:

```bash
python3 check_significance.py run1/train.log run2/train.log run3/train.log run4/train.log run5/train.log
```

## Useful knobs to sweep first

```bash
MODEL_DIM=768 NUM_RECUR_STEPS=8 QUANT_MODE=int8
MODEL_DIM=896 NUM_RECUR_STEPS=6 QUANT_MODE=int8
MODEL_DIM=768 NUM_RECUR_STEPS=8 QUANT_MODE=mixed
MIN_RECUR_STEPS=5 RECUR_WARMUP_STEPS=3000
EVAL_RECUR_STEPS=9
BACKOUT_AFTER_DEPTH=4 BACKOUT_INIT=0.0
FAKE_QUANT_BITS=8 FAKE_QUANT_START_STEP=12000
FAKE_QUANT_BITS=4 FAKE_QUANT_START_STEP=14000 QUANT_MODE=mixed
TTC_ENABLED=1 TTC_RECUR_STEPS=10 TTC_TOP2_MARGIN=0.35 TTC_MIN_TOKEN_FRACTION=0.03
MTP_HORIZONS=0
FFN_MULT=2.5
```

## Experimental notes for later agents

These levers are intentionally integrated even though their payoff is not yet proven on this exact stack. They should be kept only if they win on real H100 A/B runs:

- `FAKE_QUANT_BITS`
  - Why it may help: train-time exposure to the same kind of weight distortion introduced by the final compact artifact.
  - Why it may fail: can reduce optimization quality or slow the compiled path.
  - What to measure:
    - `final_compact_roundtrip_exact val_bpb`
    - step time around and after `FAKE_QUANT_START_STEP`
    - final artifact size under `QUANT_MODE=mixed` or `int4`

- `TTC_ENABLED`
  - Why it may help: the shared-depth model can exploit extra recurrent passes at eval time on uncertain tokens.
  - Why it may fail: extra eval passes may exceed the 10-minute eval budget or help too few tokens to matter.
  - What to measure:
    - eval wallclock
    - delta between plain `EVAL_RECUR_STEPS` and `TTC_RECUR_STEPS`
    - fraction of tokens hitting the TTC gate

- `BACKOUT_AFTER_DEPTH`
  - Why it may help: deeper shared recurrence can oversmooth representations; subtracting an early hidden state can preserve recoverable local signal.
  - Why it may fail: can destabilize the recurrent stack or simply learn a near-zero gate.
  - What to measure:
    - whether `backout.gate` moves away from zero
    - pre-export and roundtrip `val_bpb`
    - interaction with `EVAL_RECUR_STEPS > NUM_RECUR_STEPS`

Points that are still intentionally not integrated because they are not yet worth the complexity at this stage:

- token-routed MoR
- ternary BitNet/QAT
- arithmetic coding
- document-boundary masking
- NorMuon

They may still become worthwhile later, but only after the current stack is benchmarked and a concrete bottleneck is identified.

## Local sanity check

```bash
python3 train_gpt.py --mode selftest
```

That does not require CUDA or the FineWeb shards; it only validates the model graph and compact export roundtrip.
