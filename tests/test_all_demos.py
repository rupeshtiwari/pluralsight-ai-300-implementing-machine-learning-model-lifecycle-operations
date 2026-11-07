from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_20_demos_execute_cleanly():
    scripts = sorted((ROOT / "demos").glob("m*_demo.py"))
    assert len(scripts) == 20
    for s in scripts:
        r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True, check=True)
        assert "PASS]" in r.stdout
