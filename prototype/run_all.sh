#!/bin/sh
# Runs every prototype check.  Needs uv; the GPU checks need any OpenGL 3.3 (Mesa llvmpipe is fine).
set -e
cd "$(dirname "$0")"
PY="uv run --quiet --with numpy --with moderngl --with pillow python -u"
for t in check_estimator check_table check_levels check_camera check_camera_rules check_camera2d; do
    echo "== $t"; $PY $t.py
done
echo "== check_shaders"; xvfb-run -a -s "-screen 0 800x600x24" $PY check_shaders.py
echo "ALL PROTOTYPE CHECKS PASSED"
