"""The controls window (F1), design section 5.2.  Stage 1 lists the 2D controls."""
from __future__ import annotations

from imgui_bundle import hello_imgui, imgui

CONTROLS_2D = [
    ("Left or right drag", "Pan: the picture follows the mouse"),
    ("Wheel", "Zoom by 1.25 per notch, keeping the point under the cursor"),
    ("W A S D", "Pan by one view height per second; Shift is 4 times faster"),
    ("Home", "Reset the view"),
    ("Tab", "Hide or show the panel"),
    ("F1", "Show or hide this window"),
    ("M", "Hide or show the inset map"),
    ("F11", "Full screen (fills the monitor's work area)"),
    ("F12", "Save the picture as hypersponge_YYYYMMDD_HHMMSS.png in the current folder"),
]

ABOUT = ("Zoom has no limit: positions are kept as exact integers plus a small float, so the "
         "picture stays sharp at any depth. The status bar shows the depth as a power of 3.")


def draw_help(app) -> None:
    vp = imgui.get_main_viewport()
    imgui.set_next_window_pos(vp.get_center(), imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(hello_imgui.em_size(34), 0), imgui.Cond_.first_use_ever)
    visible, still_open = imgui.begin("Controls", True, imgui.WindowFlags_.no_collapse)
    if visible:
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_stretch_prop
        if imgui.begin_table("controls", 2, flags):
            imgui.table_setup_column("Input", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("2D carpet")
            imgui.table_headers_row()
            for key, what in CONTROLS_2D:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(key)
                imgui.table_next_column()
                imgui.text_wrapped(what)
            imgui.end_table()
        imgui.spacing()
        imgui.text_wrapped(ABOUT)
        imgui.text_disabled("3D and 4D controls arrive with those modes.")
    imgui.end()
    if still_open is False:
        app.show_help = False
