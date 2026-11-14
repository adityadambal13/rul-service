"""Train the serving bundle on one subset, evaluate it, save it and log the run to MLflow."""

import argparse
from pathlib import Path

import mlflow

from rul_service import config
from rul_service.data import SUBSETS, load_subset
from rul_service.model import evaluate_bundle, fit_bundle, save_bundle

TRACKING_URI = "sqlite:///mlflow.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=config.model_path())
    args = parser.parse_args()

    train, test, test_rul = load_subset(args.raw_dir, args.subset)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("rul-bundle")
    with mlflow.start_run(run_name=f"bundle-{args.subset}") as run:
        bundle = fit_bundle(train, args.subset, extra_metadata={"mlflow_run_id": run.info.run_id})
        metrics = evaluate_bundle(bundle, test, test_rul)
        save_bundle(bundle, args.output)

        mlflow.log_params({"subset": args.subset, "window": bundle.window, "cap": bundle.rul_cap})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.output))

    print(f"saved {args.output}  (version {bundle.version})")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print("Interval coverage should be near 0.80 for an 80% band; far below means too narrow.")


if __name__ == "__main__":
    main()
