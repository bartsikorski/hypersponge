"""The scripted GUI session (design 18.3): opens a window (Xvfb on Linux without a display)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]


def test_scripted_gui_session(tmp_path):
    report = tmp_path / "report.json"
    cmd = [sys.executable, "-m", "hypersponge", "gui", "--size", "960x540", "--selftest", str(report)]
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        if not shutil.which("xvfb-run"):
            pytest.skip("no display and no xvfb-run")
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1400x900x24"] + cmd
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert report.exists(), r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads(report.read_text(encoding="utf-8"))
    failed = [(s["name"], s.get("error")) for s in rep["steps"] if not s["passed"]]
    assert not failed, failed
    assert not rep["gl_errors"] and not rep["errors"], (rep["gl_errors"], rep["errors"])
    assert rep["passed"] and r.returncode == 0, r.stderr[-3000:]
    steps = {s["name"]: s for s in rep["steps"]}
    expected = ["startup", "wheel_100", "drag", "right_drag", "keys", "style_levels", "style_clay", "palette_ember",
                "inverted", "quality_low", "home", "light_background", "default_style", "m_1", "m_2", "max_level_3", "max_level_1",
                "restore_and_home", "deep_300", "home_again", "tab", "f1", "m_key", "f11", "f12", "resize"]
    assert list(steps) == expected
    assert steps["wheel_100"]["depth"] >= 20 and steps["wheel_100"]["covered"] == 0
    assert steps["deep_300"]["depth"] >= 300 and steps["deep_300"]["covered"] == 0
    assert steps["restore_and_home"]["diff_to_start"] < 2.0
    assert Path(steps["f12"]["file"]).exists()
    shots = report.with_name("report_shots")
    for name in expected:
        assert (shots / f"{name}.png").exists(), name
