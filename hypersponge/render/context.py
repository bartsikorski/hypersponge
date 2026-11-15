"""OpenGL contexts: the window's (made by hello_imgui) and a headless one for stills and tests."""
from __future__ import annotations

import sys

GL_REQUIRE = 330          # OpenGL 3.3 core is all the shaders need


def make_headless_context():
    """A standalone OpenGL 3.3 context (no window).  On Linux it falls back to EGL."""
    import moderngl
    errors = []
    # (glcontext has no EGL backend on Windows: trying it there only hides the real error)
    for kw in ({},) + (() if sys.platform == "win32" else ({"backend": "egl"},)):
        try:
            return moderngl.create_standalone_context(require=GL_REQUIRE, **kw)
        except Exception as e:      # pragma: no cover - platform dependent
            errors.append(f"{kw}: {e}")
    raise RuntimeError("Could not create an OpenGL 3.3 context:\n" + "\n".join(errors))


def describe(ctx) -> str:
    info = ctx.info
    return f"{info.get('GL_RENDERER', '?')} | OpenGL {info.get('GL_VERSION', '?')} | {info.get('GL_VENDOR', '?')}"
