import numpy as np
from retinaface import RetinaFace
from retinaface.commons.logger import Logger

logger = Logger("tests/test_actions.py")

THRESHOLD = 1000


def test_detect_first():
    """
    Test the default behavior. Detect first and align second causes
        so many black pixels
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/img11.jpg")
    num_black_pixels = np.sum(np.all(faces[0] == 0, axis=2))
    assert num_black_pixels > THRESHOLD
    logger.info("✅ Disabled align_first test for single face photo done")


def test_align_first():
    """
    Test align first behavior. Align first and detect second do not cause
        so many black pixels in contrast to default behavior
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/img11.jpg", align_first=True)
    num_black_pixels = np.sum(np.all(faces[0] == 0, axis=2))
    assert num_black_pixels < THRESHOLD
    logger.info("✅ Enabled align_first test for single face photo  done")


def test_align_first_for_group_photo():
    """
    Align first will not work if the given image has many faces and
        it will cause so many black pixels
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/couple.jpg", align_first=True)
    for face in faces:
        num_black_pixels = np.sum(np.all(face == 0, axis=2))
        assert num_black_pixels > THRESHOLD

    logger.info("✅ Enabled align_first test for group photo done")


def test_default_behavior_for_group_photo():
    """
    Align first will not work in the default behaviour and
        it will cause so many black pixels
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/couple.jpg")
    for face in faces:
        num_black_pixels = np.sum(np.all(face == 0, axis=2))
        assert num_black_pixels > THRESHOLD

    logger.info("✅ Disabled align_first test for group photo done")
