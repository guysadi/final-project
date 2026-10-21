"""Download the E-commerce Customer Churn dataset from Kaggle into data/raw/."""
import shutil
from pathlib import Path

import kagglehub

DATASET = "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    src_dir = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, RAW_DIR / f.name)
            print(f"Copied {f.name} -> {RAW_DIR / f.name}")


if __name__ == "__main__":
    main()
