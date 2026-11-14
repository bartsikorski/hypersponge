"""Interface pieces that need no window: the inset geometry, the status text, input handling."""
import pytest

from hypersponge.core.camera import Camera2D
from hypersponge.core.rule import Rule
from hypersponge.ui import inset
from hypersponge.ui.app import App, InputSnapshot
from hypersponge.ui.panel import depth_text


def test_inset_square_and_view():
    sq0, sq1, v0, v1, dot = inset.geometry_2d((0.5, 0.5), 0.5, 1.0, (0.0, 0.0))
    assert sq1[0] - sq0[0] == pytest.approx(inset.SCALE_2D) and sq1[1] - sq0[1] == pytest.approx(inset.SCALE_2D)
    assert v1[0] - v0[0] == pytest.approx(0.5 * inset.SCALE_2D) and dot is None
    assert (v0[0] + v1[0]) / 2 == pytest.approx(inset.BOX / 2)
    # a wide view is shrunk so that it fits
    sq0, sq1, v0, v1, dot = inset.geometry_2d((0.5, 0.5), 1.25, 16 / 9, (0.0, 0.0))
    assert v1[0] - v0[0] == pytest.approx(inset.FIT) and 0 <= v0[0] and v1[0] <= inset.BOX
    # a tiny view is a dot at its centre, y down
    sq0, sq1, v0, v1, dot = inset.geometry_2d((0.25, 0.75), 1e-6, 1.6, (10.0, 20.0))
    assert dot == pytest.approx((10 + 100 - 0.25 * inset.SCALE_2D, 20 + 100 - 0.25 * inset.SCALE_2D))


def test_depth_text():
    assert depth_text(0) == "3^0"
    assert depth_text(41) == "3^-41 (10^-19.6)"


def make_app():
    app = App("2d", (800, 600))
    app.fb = (800, 600)
    for cam in app.cams.values():
        cam.resize(800, 600)
    return app


def test_drag_pans_and_ends_on_release():
    app = make_app()
    c0 = app.camera.center_world().copy()
    app.handle_input(InputSnapshot(mouse=(400, 300), down=(True, False, False), clicked=(True, False, False)), 0.016)
    app.handle_input(InputSnapshot(mouse=(500, 300), down=(True, False, False)), 0.016)
    c1 = app.camera.center_world()
    assert c1[0] < c0[0] and abs(c1[1] - c0[1]) < 1e-12           # the picture follows the mouse
    app.handle_input(InputSnapshot(mouse=(500, 300)), 0.016)
    assert app.drag is None and not app.hint


def test_drag_does_not_start_over_the_panel_but_wheel_zooms_elsewhere():
    app = make_app()
    k0 = app.camera.key()
    app.handle_input(InputSnapshot(mouse=(10, 10), down=(True, False, False), clicked=(True, False, False),
                                   ui_mouse=True), 0.016)
    assert app.drag is None and app.camera.key() == k0
    app.handle_input(InputSnapshot(mouse=(400, 300), wheel=1.0), 0.016)
    assert app.camera.world_height() == pytest.approx(1.25 * 0.8)


def test_keys_toggle_and_typing_blocks_them():
    app = make_app()
    app.handle_input(InputSnapshot(pressed={"tab", "f1", "m"}), 0.016)
    assert not app.ui_visible and app.show_help and not app.show_inset
    app.handle_input(InputSnapshot(pressed={"tab"}, keys={"d"}, typing=True), 0.5)
    assert not app.ui_visible and app.camera.center_world()[0] == pytest.approx(0.5)
    app.handle_input(InputSnapshot(keys={"d"}), 0.1)                # one view height per second
    assert app.camera.center_world()[0] == pytest.approx(0.5 + 0.1 * 1.25)
    app.handle_input(InputSnapshot(keys={"d"}, shift=True), 0.05)   # four with Shift
    assert app.camera.center_world()[0] == pytest.approx(0.5 + 0.3 * 1.25)
    app.handle_input(InputSnapshot(pressed={"home"}), 0.016)
    assert app.camera.center_world()[0] == pytest.approx(0.5)


def test_rule_changes():
    app = make_app()
    app.set_rule(m=1)
    assert app.rule == Rule(2, 1)
    app.set_rule(max_level=3)
    assert app.rule == Rule(2, 1, 3)
    app.set_rule(m=2, max_level=None)
    assert app.rule == Rule(2, 2) and isinstance(app.camera, Camera2D)
    app.set_mode("3d")                                              # not built yet: ignored
    assert app.mode == "2d"
