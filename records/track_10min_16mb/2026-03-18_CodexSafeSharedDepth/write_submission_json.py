from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


FINAL_RE = re.compile(
    r"final_compact_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) "
    r"bytes_model:(?P<bytes_model>\d+) bytes_code:(?P<bytes_code>\d+) bytes_total:(?P<bytes_total>\d+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", default=".")
    parser.add_argument("--author", required=True)
    parser.add_argument("--github-id", required=True)
    parser.add_argument("--name", default="Codex Safe Shared Depth")
    parser.add_argument(
        "--blurb",
        default=(
            "Shared-depth transformer with GQA, SwiGLU, Muon+Adam optimization, "
            "sequence-length warmup, and full compact roundtrip validation."
        ),
    )
    args = parser.parse_args()

    record_dir = Path(args.record_dir).resolve()
    log_path = record_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    final_match = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = FINAL_RE.search(line)
        if match:
            final_match = match
    if final_match is None:
        raise RuntimeError("Could not find final_compact_roundtrip_exact line in train.log")

    payload = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb,
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "val_loss": float(final_match.group("val_loss")),
        "val_bpb": float(final_match.group("val_bpb")),
        "bytes_model_compact_zlib": int(final_match.group("bytes_model")),
        "bytes_code": int(final_match.group("bytes_code")),
        "bytes_total": int(final_match.group("bytes_total")),
    }

    out_path = record_dir / "submission.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
