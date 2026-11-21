"""Write deploy/sample_request.json: the latest 30 cycles of one real test engine.

The deploy pipeline sends this request to the live service as a smoke test.
"""

import argparse
import json
from pathlib import Path

from rul_service.data import SUBSETS, load_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--unit", type=int, default=3, help="test engine number")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("deploy/sample_request.json"))
    args = parser.parse_args()

    _, test, test_rul = load_subset(args.raw_dir, args.subset)
    engine = test[test["unit"] == args.unit].tail(30)
    readings = engine.drop(columns="unit").to_dict("records")
    body = {"engine_id": f"{args.subset}-test-{args.unit}", "readings": readings}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body))
    truth = int(test_rul.loc[test_rul["unit"] == args.unit, "rul"].iloc[0])
    print(f"wrote {args.output} ({len(readings)} readings). True RUL for this engine: {truth}")


if __name__ == "__main__":
    main()
