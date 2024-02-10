import json
from retinaface import RetinaFace
from retinaface.commons.logger import Logger

logger = Logger("tests/test_package.py")


def test_version():
    with open("./package_info.json", "r", encoding="utf-8") as f:
        package_info = json.load(f)

    assert RetinaFace.__version__ == package_info["version"]
    logger.info("✅ versions are matching in both package_info.json and retinaface/__init__.py")
