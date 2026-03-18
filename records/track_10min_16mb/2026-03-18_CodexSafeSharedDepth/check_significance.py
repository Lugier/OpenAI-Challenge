from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


FINAL_RE = re.compile(r"final_compact_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)")


def extract_bpb(log_path: Path) -> float:
    match = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        found = FINAL_RE.search(line)
        if found:
            match = found
    if match is None:
        raise RuntimeError(f"Could not find final_compact_roundtrip_exact line in {log_path}")
    return float(match.group("val_bpb"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--baseline", type=float, default=1.22436570)
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.logs]
    scores = [extract_bpb(path) for path in paths]
    mean = sum(scores) / len(scores)
    if len(scores) < 2:
        raise RuntimeError("Need at least two runs to estimate variance")
    sample_var = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    stderr = math.sqrt(sample_var / len(scores))
    t_stat = (mean - args.baseline) / stderr if stderr > 0 else float("-inf")

    print(f"scores={scores}")
    print(f"mean={mean:.8f}")
    print(f"delta_vs_baseline={mean - args.baseline:.8f}")
    print(f"t_stat={t_stat:.6f}")

    if stats is None:
        print("scipy_not_available=1")
        print("p_value=unavailable")
        return

    p_value = stats.ttest_1samp(scores, args.baseline, alternative="less").pvalue
    print(f"p_value={p_value:.8g}")


if __name__ == "__main__":
    main()
