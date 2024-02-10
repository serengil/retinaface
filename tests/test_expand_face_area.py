import cv2
from retinaface import RetinaFace
from retinaface.commons import postprocess
from retinaface.commons.logger import Logger

logger = Logger("tests/test_expand_face_area.py")


def test_expand_face_area():
    img_path = "tests/dataset/img11.jpg"
    default_faces = RetinaFace.extract_faces(img_path=img_path, expand_face_area=10)

    img1 = default_faces[0]
    img1 = cv2.resize(img1, (500, 500))

    obj1 = RetinaFace.detect_faces(img1, threshold=0.1)

    expanded_faces = RetinaFace.extract_faces(img_path=img_path, expand_face_area=50)

    img2 = expanded_faces[0]
    img2 = cv2.resize(img2, (500, 500))

    obj2 = RetinaFace.detect_faces(img2, threshold=0.1)

    landmarks1 = obj1["face_1"]["landmarks"]
    landmarks2 = obj2["face_1"]["landmarks"]

    distance1 = postprocess.find_euclidean_distance(landmarks1["right_eye"], landmarks1["left_eye"])
    distance2 = postprocess.find_euclidean_distance(landmarks2["right_eye"], landmarks2["left_eye"])

    # 2nd one's expand ratio is higher. so, it should be smaller.
    assert distance2 < distance1

    logger.info("✅ Test expand face area is done")
