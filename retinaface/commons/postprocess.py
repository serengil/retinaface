import math
from typing import Union, Tuple
import numpy as np
from PIL import Image


# pylint: disable=unused-argument


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> float:
    """
    Find euclidean distance between 2 vectors
    Args:
        source_representation (numpy array or list)
        test_representation (numpy array or list)
    Returns
        distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(
    img: np.ndarray, left_eye: tuple, right_eye: tuple, nose: tuple
) -> Tuple[np.ndarray, float, int]:
    """
    Alignma given face with respect to the left and right eye coordinates.
    Left eye is the eye appearing on the left (right eye of the person). Left top point is (0, 0)
    Args:
        img (numpy array): given image
        left_eye (tuple): left eye coordinates.
            Left eye is appearing on the left of image (right eye of the person)
        right_eye (tuple): right eye coordinates.
            Right eye is appearing on the right of image (left eye of the person)
        nose (tuple): coordinates of nose
    """

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------
    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0],
        # floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173
        # which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    else:
        angle = 0.0  # Dummy value for undefined angle

    # -----------------------

    return img, angle, direction


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
        angle (float): Angle of rotation in degrees.
        direction (int): Direction of rotation (-1 for clockwise, 1 for counterclockwise).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_facial_area (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) of the rotated facial area.
    """
    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)


def bbox_pred(boxes, box_deltas):
    """
    This function is copied from the following code snippet:
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def landmark_pred(boxes, landmark_deltas):
    """
    This function copied from the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred


def clip_boxes(boxes, im_shape):
    """
    This function copied from rcnn module of retinaface-tf2 project
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/processing/bbox_transform.py
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def anchors_plane(height, width, stride, base_anchors):
    """
    This function is mainly based on the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/anchors.pyx
    """
    A = base_anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(
        base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1)
    )
    return all_anchors


def cpu_nms(dets, threshold):
    """
    This function is mainly based on the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx
    Fast R-CNN by Ross Girshick
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep
