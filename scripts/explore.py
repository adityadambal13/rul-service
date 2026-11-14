"""First look at the data: engine lifetimes, dead sensors, and degradation trends."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt

from rul_service.data import SENSOR_COLS, SUBSETS, load_subset
from rul_service.features import RegimeScaler

RAW = Path("data/raw")
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def main() -> None:
    for subset in SUBSETS:
        train, test, rul = load_subset(RAW, subset)
        life = train.groupby("unit")["cycle"].max()
        scaler = RegimeScaler(n_regimes=N_REGIMES[subset]).fit(train)
        dropped = [c for c in SENSOR_COLS if c not in scaler.kept_sensors]
        print(
            f"{subset}: {train['unit'].nunique()} train engines, "
            f"life min/median/max = {life.min()}/{int(life.median())}/{life.max()} cycles, "
            f"test RUL max = {rul['rul'].max()}, dropped sensors = {dropped}"
        )

    # Plot a few informative sensors for three engines, FD001
    train, _, _ = load_subset(RAW, "FD001")
    scaler = RegimeScaler(n_regimes=1).fit(train)
    scaled = scaler.transform(train)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, sensor in zip(axes.ravel(), scaler.kept_sensors[:6], strict=False):
        for unit in (1, 2, 3):
            e = scaled[scaled["unit"] == unit]
            ax.plot(e["cycle"], e[sensor], label=f"engine {unit}")
        ax.set_title(sensor)
    axes[0, 0].legend()
    fig.suptitle("FD001: scaled sensor readings vs cycle (run to failure)")
    Path("reports").mkdir(exist_ok=True)
    fig.savefig("reports/fd001_sensors.png", dpi=120, bbox_inches="tight")
    print("saved reports/fd001_sensors.png")


if __name__ == "__main__":
    main()
