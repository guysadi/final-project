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
from src.break_scenarios import SCENARIOS as BREAKS  # noqa: E402

BACKUP_DIR = paths.ROOT / ".break_backup"
TARGETS = [paths.CLEAN_DATA, paths.CONTRACT]


def _backup() -> None:
    if BACKUP_DIR.exists():
        return  # keep the pristine copy from the first break
    BACKUP_DIR.mkdir()
    for p in TARGETS:
        shutil.copy2(p, BACKUP_DIR / p.name)


def _apply(scenario) -> None:
    df = pd.read_csv(paths.CLEAN_DATA)
    contract = json.loads(paths.CONTRACT.read_text())
    df, contract = scenario.apply(df, contract)
    df.to_csv(paths.CLEAN_DATA, index=False)
    paths.CONTRACT.write_text(json.dumps(contract, indent=2))


SCENARIOS = {name: (lambda s=s: _apply(s)) for name, s in BREAKS.items()}


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
    print(f"Broke the handoff: {name} — {BREAKS[name].description}")
    print("Now run:  python -m src.flow --skip-crew1")
    print("Undo with: python scripts/break_handoff.py restore")


if __name__ == "__main__":
    main()
