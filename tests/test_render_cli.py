"""The command line (design 18.1)."""
import numpy as np
import pytest
from PIL import Image

from hypersponge.cli import main


@pytest.mark.parametrize("argv", [
    ["render", "-o", "x.png", "--mode", "2d", "--style", "axes"],   # not a 2D style
    ["render", "-o", "x.png", "--style", "wdepth"],            # 4D only
    ["render", "-o", "x.png", "--center", "0.5,0.5"],          # 2D only
    ["render", "-o", "x.png", "--m", "4"],
    ["render", "-o", "x.png", "--mode", "4d", "--m", "5"],
    ["render", "-o", "x.png", "--mode", "4d", "--preset", "nope"],
    ["render", "-o", "x.png", "--preset", "dust"],             # 4D only
    ["render", "-o", "x.png", "--fov", "5"],
    ["bench", "--size", "1x1"],
    ["render", "-o", "x.png", "--style", "nope"],
    ["render", "-o", "x.png", "--palette", "nope"],
    ["render", "-o", "x.png", "--mode", "2d", "--m", "3"],
    ["render", "-o", "x.png", "--size", "10x10"],
    ["render", "-o", "x.png", "--size", "big"],
    ["render", "-o", "x.png", "--mode", "2d", "--height", "-1"],
    ["render", "-o", "x.png", "--mode", "2d", "--center", "0.5"],
    ["render"],                                                # -o is required
    ["gui", "--mode", "5d"],
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
    assert main(["render", "-o", str(out), "--mode", "2d", "--size", "160x90", "--samples", "2"]) == 0
    a = np.asarray(Image.open(out))
    assert a.shape == (90, 160, 3) and a.std() > 10
    out2 = tmp_path / "deep.png"
    assert main(["render", "-o", str(out2), "--mode", "2d", "--size", "160x90", "--samples", "2", "--dive", "1500",
                 "--cursor", "37.5,-20.5", "--style", "levels", "--palette", "ice", "--light"]) == 0
    b = np.asarray(Image.open(out2))
    assert b.shape == (90, 160, 3) and b.std() > 5


@pytest.mark.gpu
def test_render_center_height_dust(ctx, tmp_path):
    out = tmp_path / "dust.png"
    assert main(["render", "-o", str(out), "--mode", "2d", "--size", "120x120", "--samples", "1", "--m", "1",
                 "--center", "0.25,0.25", "--height", "0.4", "--style", "clay", "--invert"]) == 0
    assert np.asarray(Image.open(out)).std() > 5


@pytest.mark.gpu
def test_info_passes(ctx, capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("PASS") and "WRONG" not in out


@pytest.mark.gpu
def test_render_3d_overview_and_dive(ctx, tmp_path, capsys):
    out = tmp_path / "sponge.png"
    assert main(["render", "-o", str(out), "--size", "160x90", "--samples", "2"]) == 0
    a = np.asarray(Image.open(out))
    assert a.shape == (90, 160, 3) and a.std() > 10
    assert "overview" in capsys.readouterr().out
    out2 = tmp_path / "deep.png"
    assert main(["render", "-o", str(out2), "--size", "160x90", "--samples", "1", "--dive", "200",
                 "--cursor", "10,-4", "--style", "axes", "--no-fog", "--fov", "70", "--m", "2"]) == 0
    assert np.asarray(Image.open(out2)).std() > 5
    assert "immersed" in capsys.readouterr().out


@pytest.mark.gpu
def test_bench_runs(ctx, capsys):
    assert main(["bench", "--size", "64x36", "--frames", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("64x36") == 10 and "3D overview" in out and "4D generic" in out


@pytest.mark.gpu
def test_render_4d_presets(ctx, tmp_path, capsys):
    pics = {}
    for preset in ("diagonal", "face-diagonal", "dust"):
        out = tmp_path / f"{preset}.png"
        assert main(["render", "-o", str(out), "--mode", "4d", "--preset", preset, "--size", "120x90",
                     "--samples", "1"]) == 0
        pics[preset] = np.asarray(Image.open(out)).astype(int)
        assert pics[preset].std() > 5
    assert np.abs(pics["diagonal"] - pics["face-diagonal"]).mean() > 3
    out = tmp_path / "deep.png"
    assert main(["render", "-o", str(out), "--mode", "4d", "--preset", "generic", "--style", "wdepth",
                 "--dive", "120", "--size", "120x90", "--samples", "1"]) == 0
    assert "immersed" in capsys.readouterr().out
