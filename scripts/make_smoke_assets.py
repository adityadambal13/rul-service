"""Create a tiny synthetic model bundle and a matching request for CI smoke tests.

Writes smoke/models/rul_bundle.joblib and smoke/request.json. The bundle is trained on
synthetic data, so it proves the container works end to end, not that predictions are good.
"""

import json
from pathlib import Path

import numpy as np

from rul_service.model import fit_bundle, save_bundle
from rul_service.synthetic import make_engine, make_training_frame, readings_from

OUT = Path("smoke")


def main() -> None:
    train = make_training_frame(n_engines=25, seed=0)
    bundle = fit_bundle(train, "FD001", max_iter=40)
    save_bundle(bundle, OUT / "models" / "rul_bundle.joblib")

    engine = make_engine(np.random.default_rng(99), unit=1, n_cycles=150)
    request = {"engine_id": "smoke-engine", "readings": readings_from(engine, last_cycle=100)}
    (OUT / "request.json").write_text(json.dumps(request))
    print(f"wrote {OUT}/models/rul_bundle.joblib and {OUT}/request.json")


if __name__ == "__main__":
    main()
