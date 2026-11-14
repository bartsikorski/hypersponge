"""The command line (design 18.1)."""
import numpy as np
import pytest
from PIL import Image

from hypersponge.cli import main


@pytest.mark.parametrize("argv", [
    ["render", "-o", "x.png", "--mode", "3d"],                 # not built yet
    ["render", "-o", "x.png", "--style", "axes"],              # not a 2D style
    ["render", "-o", "x.png", "--style", "nope"],
    ["render", "-o", "x.png", "--palette", "nope"],
    ["render", "-o", "x.png", "--m", "3"],
    ["render", "-o", "x.png", "--size", "10x10"],
    ["render", "-o", "x.png", "--size", "big"],
    ["render", "-o", "x.png", "--height", "-1"],
    ["render", "-o", "x.png", "--center", "0.5"],
    ["render"],                                                # -o is required
    ["gui", "--mode", "4d"],
    ["--size", "12"],                                          # options first mean gui
    ["frobnicate"],
])
def test_bad_arguments_exit_2(argv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(argv) == 2
    assert not (tmp_path / "x.png").exists()


def test_help_exits_0(capsys):
    assert main(["--help"]) == 0
    assert "render" in capsys.readouterr().out


@pytest.mark.gpu
def test_render_whole_and_deep(ctx, tmp_path):
    out = tmp_path / "whole.png"
    assert main(["render", "-o", str(out), "--size", "160x90", "--samples", "2"]) == 0
    a = np.asarray(Image.open(out))
    assert a.shape == (90, 160, 3) and a.std() > 10
    out2 = tmp_path / "deep.png"
    assert main(["render", "-o", str(out2), "--size", "160x90", "--samples", "2", "--dive", "1500",
                 "--cursor", "37.5,-20.5", "--style", "levels", "--palette", "ice", "--light"]) == 0
    b = np.asarray(Image.open(out2))
    assert b.shape == (90, 160, 3) and b.std() > 5


@pytest.mark.gpu
def test_render_center_height_dust(ctx, tmp_path):
    out = tmp_path / "dust.png"
    assert main(["render", "-o", str(out), "--size", "120x120", "--samples", "1", "--m", "1",
                 "--center", "0.25,0.25", "--height", "0.4", "--style", "clay", "--invert"]) == 0
    assert np.asarray(Image.open(out)).std() > 5


@pytest.mark.gpu
def test_info_passes(ctx, capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("PASS") and "WRONG" not in out
