"""Break Crew 1's handoff on purpose, to show the Flow refusing to train.

"A validation you've never seen fail is untested code." Each scenario damages
artifacts/crew1 the way a real handoff goes wrong; `restore` puts the originals back.

    python scripts/break_handoff.py unit_change     # CashbackAmount dollars → cents (the H&V incident)
    python -m src.flow --skip-crew1                 # → stops at Gate 1, no model trained
    python scripts/break_handoff.py restore
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

BACKUP_DIR = paths.ROOT / ".break_backup"
TARGETS = [paths.CLEAN_DATA, paths.CONTRACT]


def _backup() -> None:
    if BACKUP_DIR.exists():
        return  # keep the pristine copy from the first break
    BACKUP_DIR.mkdir()
    for p in TARGETS:
        shutil.copy2(p, BACKUP_DIR / p.name)


def _edit_data(fn) -> None:
    df = pd.read_csv(paths.CLEAN_DATA)
    fn(df).to_csv(paths.CLEAN_DATA, index=False)


def _edit_contract(fn) -> None:
    contract = json.loads(paths.CONTRACT.read_text())
    fn(contract)
    paths.CONTRACT.write_text(json.dumps(contract, indent=2))


def unit_change():
    """CashbackAmount silently converted from dollars to cents."""
    _edit_data(lambda df: df.assign(CashbackAmount=(df["CashbackAmount"] * 100).round(2)))


def rename_column():
    """Tenure renamed to match a 'new naming convention'."""
    _edit_data(lambda df: df.rename(columns={"Tenure": "tenure_months"}))


def drop_contract_column():
    """A row deleted from the contract: data has a column the contract no longer describes."""
    _edit_contract(lambda c: c["columns"].pop("CouponUsed"))


def bad_category():
    """Someone abbreviates a category value."""
    _edit_data(lambda df: df.assign(Gender=df["Gender"].replace({"Female": "F"})))


def remove_feature_source():
    """OrderCount dropped from both data and contract. They still agree, so Gate 1 passes —
    but the model needs OrderCount, so Gate 2 stops the Flow before modeling."""
    _edit_data(lambda df: df.drop(columns="OrderCount"))
    _edit_contract(lambda c: c["columns"].pop("OrderCount"))


SCENARIOS = {f.__name__: f for f in
             (unit_change, rename_column, drop_contract_column, bad_category, remove_feature_source)}


def restore() -> None:
    if not BACKUP_DIR.exists():
        print("Nothing to restore — artifacts are untouched.")
        return
    for p in TARGETS:
        shutil.copy2(BACKUP_DIR / p.name, p)
    shutil.rmtree(BACKUP_DIR)
    print("Restored the original Crew 1 artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Break Crew 1's handoff on purpose.")
    parser.add_argument("scenario", choices=[*SCENARIOS, "restore"])
    name = parser.parse_args().scenario
    if name == "restore":
        restore()
        return
    _backup()
    SCENARIOS[name]()
    print(f"Broke the handoff: {name} — {SCENARIOS[name].__doc__.strip()}")
    print("Now run:  python -m src.flow --skip-crew1")
    print("Undo with: python scripts/break_handoff.py restore")


if __name__ == "__main__":
    main()
