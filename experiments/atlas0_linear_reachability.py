from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_atlas import run_atlas


def main() -> None:
    result = run_atlas()
    out = ROOT / "results" / "atlas0_linear_reachability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for task, metrics in result["summary"].items():
        print(task)
        print(f"  grown target error  {metrics['target_relative_fro_error']['mean']:.5f} +/- {metrics['target_relative_fro_error']['std']:.5f}")
        print(f"  hull target error   {metrics['hull_relative_fro_error']['mean']:.5f} +/- {metrics['hull_relative_fro_error']['std']:.5f}")
        print(f"  held-out NMSE       {metrics['heldout_nmse']['mean']:.5f} +/- {metrics['heldout_nmse']['std']:.5f}")
        print(f"  occupied features   {metrics['mean_effective_feature_count']['mean']:.2f} +/- {metrics['mean_effective_feature_count']['std']:.2f}")
        print(f"  grown ranks         {metrics['grown_rank_values']}")


if __name__ == "__main__":
    main()
