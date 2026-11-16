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
    expected = ["startup", "drag_3d", "right_drag_3d", "wheel_30", "style_levels_3d", "style_axes_3d",
                "style_clay_3d", "palette_ember_3d", "light_background_3d", "fog_off", "occlusion_0",
                "quality_low_3d", "fov_70", "view_distance_100", "m_1_3d", "m_3_3d", "max_level_1_3d",
                "max_level_3_3d", "restore_3d", "keys_then_home", "dive_3d", "home_3d", "mode_2d", "wheel_100",
                "drag_2d", "right_drag_2d", "keys_2d", "style_levels_2d", "style_clay_2d", "palette_ember_2d",
                "inverted", "quality_low_2d", "home_2d", "light_background_2d", "default_style_2d", "m_1_2d",
                "m_2_2d", "max_level_3_2d", "max_level_1_2d", "restore_2d", "deep_2d", "home_again_2d", "mode_3d",
                "tab", "f1", "m_key", "f11", "f12", "resize"]
    assert list(steps) == expected
    assert steps["wheel_100"]["depth"] >= 20 and steps["wheel_100"]["covered"] == 0
    assert steps["deep_2d"]["depth"] >= 300 and steps["deep_2d"]["covered"] == 0
    assert steps["restore_2d"]["diff_to_start"] < 2.0
    assert steps["startup"]["overview"] and steps["drag_3d"]["rho_change"] < 1e-9
    assert not steps["wheel_30"]["overview"] and steps["wheel_30"]["depth"] >= 3
    assert steps["keys_then_home"]["diff_to_start"] < 2.0 and steps["restore_3d"]["diff_to_start"] < 2.0
    assert steps["dive_3d"]["depth"] >= 60 and steps["mode_3d"]["diff_to_start"] < 2.0
    assert Path(steps["f12"]["file"]).exists()
    shots = report.with_name("report_shots")
    for name in expected:
        assert (shots / f"{name}.png").exists(), name
