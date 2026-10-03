import os
import sys
import json
import subprocess

import pytest

from retinaface.commons import backend_utils
from retinaface.commons.logger import Logger

logger = Logger("tests/test_backends.py")

# top level modules of each backend engine
BACKEND_MODULES = {
    backend_utils.TENSORFLOW: "tensorflow",
    backend_utils.PYTORCH: "torch",
    backend_utils.ONNX: "onnxruntime",
}

# the backend engine is decided once per process, and retinaface may already be imported
# by other tests. so, each backend is tested in a fresh python process.
SCRIPT = """
import sys
import json
from retinaface import RetinaFace
from retinaface.commons import backend_utils

faces = RetinaFace.detect_faces("tests/dataset/img3.jpg")
print(json.dumps({
    "backend": backend_utils.get_backend_engine(),
    "num_faces": len(faces),
    "modules": sorted(name.split(".")[0] for name in sys.modules),
}))
"""


@pytest.mark.parametrize("backend", backend_utils.BACKENDS)
def test_only_enforced_backend_is_imported(backend: str):
    if not backend_utils.is_backend_available(backend):
        pytest.skip(f"{backend} is not installed")

    env = {**os.environ, backend_utils.BACKEND_ENGINE_ENV_VAR: backend}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # the last line is the json result, previous ones may be logs
    resp = json.loads(result.stdout.strip().splitlines()[-1])

    assert resp["backend"] == backend
    assert resp["num_faces"] > 0

    modules = set(resp["modules"])
    assert BACKEND_MODULES[backend] in modules
    for other_backend, module in BACKEND_MODULES.items():
        if other_backend != backend:
            assert module not in modules, f"{module} is imported although backend is {backend}"

    logger.info(f"✅ only {backend} is imported when {backend} backend is enforced")
