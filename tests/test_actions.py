import cv2
import numpy as np
from retinaface import RetinaFace
from retinaface.commons.logger import Logger

logger = Logger("tests/test_actions.py")

do_plotting = False

if do_plotting is True:
    import matplotlib.pyplot as plt


def int_tuple(t):
    return tuple(int(x) for x in t)


def test_analyze_crowded_photo():
    img_path = "tests/dataset/img3.jpg"
    img = cv2.imread(img_path)
    resp = RetinaFace.detect_faces(img_path, threshold=0.1)

    # it has to find 239 faces with threshold 0.1
    assert len(resp.keys()) > 200

    for idx, identity in resp.items():
        confidence = identity["score"]
        logger.debug(f"Confidence score of {idx} is {confidence}")

        rectangle_color = (255, 255, 255)

        landmarks = identity["landmarks"]
        diameter = 1

        cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

        facial_area = identity["facial_area"]

        cv2.rectangle(
            img,
            (facial_area[2], facial_area[3]),
            (facial_area[0], facial_area[1]),
            rectangle_color,
            1,
        )

    if do_plotting is True:
        plt.imshow(img[:, :, ::-1])
        plt.axis("off")
        plt.show()
        # cv2.imwrite("outputs/" + img_path.split("/")[1], img)

    logger.info("✅ Crowded photo analysis test done")


def test_alignment_for_inverse_clock_way():
    # img11.jpg is required to rotate inverse direction of clock
    img_path = "tests/dataset/img11.jpg"
    img = cv2.imread(img_path)
    do_alignment_checks(img, expected_faces=1)
    logger.info("✅ Alignment for inverse clock way test done")


def test_alignment_for_clock_way():
    # img11.jpg is required to rotate inverse direction of clock
    img_path = "tests/dataset/img11.jpg"
    img = cv2.imread(img_path)
    mirror_img = cv2.flip(img, 1)
    do_alignment_checks(mirror_img, expected_faces=1)
    logger.info("✅ Alignment for clock way test done")


def do_alignment_checks(img: np.ndarray, expected_faces: int) -> None:
    faces = RetinaFace.extract_faces(img_path=img, align=True, expand_face_area=25)

    # it has one clear face
    assert len(faces) == expected_faces

    for face in faces:
        if do_plotting is True:
            plt.imshow(face)
            plt.axis("off")
            plt.show()

        obj = RetinaFace.detect_faces(face, threshold=0.1)
        landmarks = obj["face_1"]["landmarks"]
        right_eye = landmarks["right_eye"]
        left_eye = landmarks["left_eye"]

        # check eyes are on same horizantal
        assert abs(right_eye[1] - left_eye[1]) < 10


def test_different_expanding_ratios():
    expand_ratios = [0, 25, 50]

    for expand_ratio in expand_ratios:
        faces = RetinaFace.extract_faces(
            img_path="tests/dataset/img11.jpg", align=True, expand_face_area=expand_ratio
        )
        for face in faces:
            if do_plotting is True:
                plt.imshow(face)
                plt.axis("off")
                plt.show()


def test_resize():
    faces = RetinaFace.extract_faces(img_path="tests/dataset/img11.jpg", target_size=(224, 224))
    for face in faces:
        assert face.shape == (224, 224, 3)
        if do_plotting is True:
            plt.imshow(face)
            plt.show()
    logger.info("✅ resize test done")
