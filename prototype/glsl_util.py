"""Shader loading and uniform upload shared by the prototype's GPU checks."""
from pathlib import Path
import numpy as np
from sponge_ref import MAXLEV

HERE = Path(__file__).resolve().parent / "shaders"
VERT = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"


def source(name: str) -> str:
    return "#version 330\n" + (HERE / "common.glsl").read_text() + "\n" + (HERE / name).read_text()


def program(ctx, name: str):
    import moderngl
    prog = ctx.program(vertex_shader=VERT, fragment_shader=source(name))
    vao = ctx.vertex_array(prog, [(ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4")), "2f", "p")])
    return prog, vao


def set_table(prog, tb):
    lo, hi, w, blo, bhi = tb.arrays(np.float32)

    def pad(a, dtype="f4"):
        o = np.zeros((MAXLEV,) + a.shape[1:], dtype=dtype)
        o[:len(a)] = a
        return o.tobytes()
    prog["uCount"] = len(w)
    prog["uLo"].write(pad(lo)); prog["uHi"].write(pad(hi)); prog["uW"].write(pad(w))
    if "uLevelNo" in prog:
        prog["uLevelNo"].write(pad(tb.level.astype("i4"), "i4"))
    prog["uBoxLo"] = tuple(blo); prog["uBoxHi"] = tuple(bhi)
    prog["uOff"] = tuple(tb.off); prog["uM"] = tb.m


def set_if(prog, **kw):
    for k, v in kw.items():
        if k in prog:
            if isinstance(v, np.ndarray):
                prog[k].write(v.astype("f4").tobytes())
            else:
                prog[k] = v
