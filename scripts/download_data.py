"""Download and unpack the NASA C-MAPSS dataset into data/raw."""

import io
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]
RAW_DIR = Path("data/raw")
EXPECTED = [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)]


def extract(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".zip"):
                extract(zf.read(name), dest)
            elif name.endswith(".txt"):
                (dest / Path(name).name).write_bytes(zf.read(name))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                extract(resp.read(), RAW_DIR)
            break
        except Exception as exc:
            print(f"failed: {url}: {exc}")
    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")
    print("OK: all 12 files present in data/raw")


if __name__ == "__main__":
    main()
