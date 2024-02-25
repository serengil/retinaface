import numpy as np
from retinaface import RetinaFace
from retinaface.commons.logger import Logger

logger = Logger("tests/test_actions.py")

THRESHOLD = 1000

VISUAL_TEST = False


def test_align_first():
    """
    Test align first behavior. Align first and detect second do not cause
        so many black pixels in contrast to default behavior
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/img11.jpg")
    num_black_pixels = np.sum(np.all(faces[0] == 0, axis=2))
    assert num_black_pixels < THRESHOLD
    logger.info("✅ Enabled align_first test for single face photo  done")


def test_align_first_for_group_photo():
    """
    Align first will not work if the given image has many faces and
        it will cause so many black pixels
    """
    faces = RetinaFace.extract_faces(img_path="tests/dataset/couple.jpg")
    for face in faces:
        num_black_pixels = np.sum(np.all(face == 0, axis=2))
        assert num_black_pixels < THRESHOLD
        if VISUAL_TEST is True:
            import matplotlib.pyplot as plt

            plt.imshow(face)
            plt.show()

    logger.info("✅ Enabled align_first test for group photo done")
