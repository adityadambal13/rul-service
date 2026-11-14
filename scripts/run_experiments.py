"""Run every model on every subset and print the results."""

from rul_service.data import SUBSETS
from rul_service.train import MODELS, run


def main() -> None:
    for subset in SUBSETS:
        for model_name in MODELS:
            m = run(subset, model_name)
            print(
                f"{subset} {model_name:11s} "
                f"RMSE={m['test_rmse']:.2f} NASA score={m['test_nasa_score']:.0f}"
            )


if __name__ == "__main__":
    main()
