from typing import List

import numpy as np
import onnxruntime as ort

from retinaface.commons import weight_utils

# pylint: disable=too-few-public-methods

# the 2nd one is a backup source, used when downloading from the 1st one fails
WEIGHTS_URLS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/retinaface.onnx",
]


class RetinaFace:
    """
    ONNX implementation of RetinaFace with ResNet50 backbone.
    The onnx graph is exported from retinaface/model/retinaface_pth_model.py
        with dynamic batch, height and width dimensions.

    Calling it expects a float array in (N, H, W, 3) shape as the tensorflow model does and
        returns outputs in the same order and (N, H, W, C) layout of the tensorflow model:
        [cls_prob_stride32, bbox_pred_stride32, landmark_pred_stride32,
         cls_prob_stride16, bbox_pred_stride16, landmark_pred_stride16,
         cls_prob_stride8, bbox_pred_stride8, landmark_pred_stride8]
    """

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = [output.name for output in session.get_outputs()]

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        return self.session.run(self.output_names, {self.input_name: x})


def load_weights() -> ort.InferenceSession:
    """
    Loading pre-trained RetinaFace model in onnx format
    Returns:
        session (ort.InferenceSession): onnx runtime session of the pre-trained model
    """
    exact_file = weight_utils.download_weights_if_necessary(
        file_name="retinaface.onnx", source_urls=WEIGHTS_URLS
    )

    # prefer gpu when onnxruntime-gpu is installed, fall back to cpu otherwise
    available = ort.get_available_providers()
    providers = [
        provider
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in available
    ]
    return ort.InferenceSession(exact_file, providers=providers)


def build_model() -> RetinaFace:
    """
    Build RetinaFace model in onnx
    """
    return RetinaFace(load_weights())
