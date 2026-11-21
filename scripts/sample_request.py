"""Send a real test engine's latest 30 cycles to a running API and compare with the truth."""

import argparse
import os
from pathlib import Path

import httpx

from rul_service.data import load_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--unit", type=int, default=1, help="test engine number")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("RUL_API_KEY"),
        help="sent as the X-API-Key header (defaults to the RUL_API_KEY environment variable)",
    )
    args = parser.parse_args()

    _, test, test_rul = load_subset(Path("data/raw"), args.subset)
    engine = test[test["unit"] == args.unit].tail(30)
    readings = engine.drop(columns="unit").to_dict("records")
    body = {"engine_id": f"{args.subset}-test-{args.unit}", "readings": readings}
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    response = httpx.post(f"{args.url}/v1/predict", json=body, headers=headers, timeout=60)
    print("HTTP", response.status_code)
    print(response.json())
    truth = int(test_rul.loc[test_rul["unit"] == args.unit, "rul"].iloc[0])
    print(f"True RUL from NASA's answer file: {truth}")


if __name__ == "__main__":
    main()
