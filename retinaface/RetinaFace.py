import os
import warnings
import logging
from typing import Union, Any, Optional, Dict, Tuple

import numpy as np
import tensorflow as tf

from retinaface.model import retinaface_model
from retinaface.commons import preprocess, postprocess
from retinaface.commons.logger import Logger

logger = Logger(module="retinaface/RetinaFace.py")

# pylint: disable=global-variable-undefined, no-name-in-module, unused-import, too-many-locals, redefined-outer-name, too-many-statements, too-many-arguments

# ---------------------------

# configurations
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Limit the amount of reserved VRAM so that other scripts can be run in the same GPU as well
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# ---------------------------


def build_model() -> Any:
    """
    Builds retinaface model once and store it into memory
    """
    # pylint: disable=invalid-name
    global model  # singleton design pattern

    if not "model" in globals():
        model = tf.function(
            retinaface_model.build_model(),
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),),
        )

    return model


def detect_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    allow_upscaling: bool = True,
) -> Dict[str, Any]:
    """
    Detect the facial area for a given image
    Args:
        img_path (str or numpy array): given image
        threshold (float): threshold for detection
        model (Model): pre-trained model can be given
        allow_upscaling (bool): allowing up-scaling
    Returns:
        detected faces as:
        {
            "face_1": {
                "score": 0.9993440508842468,
                "facial_area": [155, 81, 434, 443],
                "landmarks": {
                    "right_eye": [257.82974, 209.64787],
                    "left_eye": [374.93427, 251.78687],
                    "nose": [303.4773, 299.91144],
                    "mouth_right": [228.37329, 338.73193],
                    "mouth_left": [320.21982, 374.58798]
                }
            }
        }
    """
    resp = {}
    img = preprocess.get_image(img_path)

    # ---------------------------

    if model is None:
        model = build_model()

    # ---------------------------

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    # ---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        # _key = f"stride{s}"
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return resp

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

    # nms = cpu_nms_wrapper(nms_threshold)
    # keep = nms(pre_det)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    for idx, face in enumerate(det):
        label = "face_" + str(idx + 1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp


def extract_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    align: bool = True,
    allow_upscaling: bool = True,
    expand_face_area: int = 0,
    align_first: bool = False,
) -> list:
    """
    Extract detected and aligned faces
    Args:
        img_path (str or numpy): given image
        threshold (float): detection threshold
        model (Model): pre-trained model can be passed to the function
        align (bool): enable or disable alignment
        allow_upscaling (bool): allowing up-scaling
        expand_face_area (int): expand detected facial area with a percentage
        align_first (bool): set this True to align first and detect second
            this can be applied only if input image has just one face
    """
    resp = []

    # ---------------------------

    img = preprocess.get_image(img_path)

    # ---------------------------

    obj = detect_faces(
        img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling
    )

    if align_first is True and len(obj) > 1:
        logger.warn(
            f"Even though align_first is set to True, there are {len(obj)} faces in input image."
            "Align first functionality can be applied only if there is single face in the input"
        )

    if isinstance(obj, dict):
        for _, identity in obj.items():
            facial_area = identity["facial_area"]

            x = facial_area[0]
            y = facial_area[1]
            w = facial_area[2]
            h = facial_area[3]

            # expand the facial area to be extracted and stay within img.shape limits
            x1 = max(0, x - int((w * expand_face_area) / 100))  # expand left
            y1 = max(0, y - int((h * expand_face_area) / 100))  # expand top
            x2 = min(img.shape[1], w + int((w * expand_face_area) / 100))  # expand right
            y2 = min(img.shape[0], h + int((h * expand_face_area) / 100))  # expand bottom

            if align_first is False or (align_first is True and len(obj) > 1):
                facial_img = img[y1:y2, x1:x2]
            else:
                facial_img = img.copy()

            if align is True:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                # mouth_right = landmarks["mouth_right"]
                # mouth_left = landmarks["mouth_left"]
                facial_img, rotate_angle, rotate_direction = postprocess.alignment_procedure(
                                                                facial_img, right_eye, left_eye, nose
                                                            )

            if align_first is True and len(obj) == 1:
                facial_area = rotate_facial_area(
                    facial_area, rotate_angle, rotate_direction, img.shape
                    )
                # expand the facial area to be extracted and stay within img.shape limits
                x1 = max(0, facial_area[0] - int((facial_area[2] * expand_face_area) / 100))
                y1 = max(0, facial_area[1] - int((facial_area[3] * expand_face_area) / 100))
                x2 = min(img.shape[1], facial_area[2] + int((facial_area[2] * expand_face_area) / 100))
                y2 = min(img.shape[0], facial_area[3] + int((facial_area[3] * expand_face_area) / 100))
                facial_img = facial_img[y1:y2, x1:x2]

            resp.append(facial_img[:, :, ::-1])

    return resp

def rotate_facial_area(facial_area: Tuple[int, int, int, int], angle: float, direction:
                        int, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.

    Args:
        facial_area (tuple of int): Representing the coordinates (x1, y1, x2, y2) of the facial area.
        angle (float): Angle of rotation in degrees.
        direction (int): Direction of rotation (-1 for clockwise, 1 for counterclockwise).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        tuple of int: Representing the new coordinates (x1, y1, x2, y2) of the rotated facial area.
    """
    # Angle in radians
    angle = angle * np.pi / 180

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - size[1] / 2
    y = (facial_area[1] + facial_area[3]) / 2 - size[0] / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + size[1] / 2
    y_new = y_new + size[0] / 2

    # Calculate the new facial area
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    return (int(x1), int(y1), int(x2), int(y2))
