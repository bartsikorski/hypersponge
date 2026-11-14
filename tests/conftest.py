import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="rewrite tests/golden/*.png from this machine (look at them before committing)")


@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="session")
def ctx():
    try:
        from hypersponge.render.context import make_headless_context
        return make_headless_context()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"no OpenGL 3.3 context: {e}")


@pytest.fixture(scope="session")
def renderer(ctx):
    from hypersponge.render.renderer import Renderer
    return Renderer(ctx, dynamic=False)
