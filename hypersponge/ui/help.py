"""The controls window (F1), design section 5.2."""
from __future__ import annotations

from imgui_bundle import hello_imgui, imgui

CONTROLS = [               # (input, 2D, 3D)
    ("Left-drag", "Pan: the picture follows the mouse",
     "Overview: turn the sponge about its centre. Immersed: look around"),
    ("Right-drag", "Pan", "Slide sideways and up or down; the point under the cursor follows the mouse"),
    ("Wheel", "Zoom by 1.25 per notch, keeping the point under the cursor",
     "Move 20% of the way to the surface under the cursor; wheel back undoes it"),
    ("W S A D", "Pan by one view height per second", "Fly forward, back, left, right"),
    ("R F", "", "Fly up, down"),
    ("Shift", "4 times faster keys", "4 times faster keys"),
    ("Q E", "", "Roll"),
    ("Home", "Reset the view", "Back to the overview"),
    ("Tab", "Hide or show the panel", ""),
    ("F1", "Show or hide this window", ""),
    ("M", "Hide or show the inset map", ""),
    ("F11", "Full screen (fills the monitor's work area)", ""),
    ("F12", "Save the picture as hypersponge_YYYYMMDD_HHMMSS.png in the current folder (Pictures when "
            "started from the icon)", ""),
]

ABOUT = ("Zoom has no limit: positions are kept as exact integers plus a small float, so the "
         "picture stays sharp at any depth. The status bar shows the depth as a power of 3. In 3D "
         "the camera never enters the solid: every move goes at most half way to the nearest "
         "surface ahead.")


def draw_help(app) -> None:
    vp = imgui.get_main_viewport()
    imgui.set_next_window_pos(vp.get_center(), imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(hello_imgui.em_size(46), 0), imgui.Cond_.first_use_ever)
    visible, still_open = imgui.begin("Controls", True, imgui.WindowFlags_.no_collapse)
    if visible:
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_stretch_prop
        if imgui.begin_table("controls", 3, flags):
            imgui.table_setup_column("Input", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("2D carpet")
            imgui.table_setup_column("3D sponge")
            imgui.table_headers_row()
            for key, two, three in CONTROLS:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(key)
                imgui.table_next_column()
                imgui.text_wrapped(two)
                imgui.table_next_column()
                imgui.text_wrapped(three or (two if key in ("Tab", "F1", "M", "F11", "F12") else ""))
            imgui.end_table()
        imgui.spacing()
        imgui.text_wrapped(ABOUT)
        imgui.text_disabled("4D controls arrive with the 4D mode.")
    imgui.end()
    if still_open is False:
        app.show_help = False
