import os
from pathlib import Path
from typing import List

import gdown
import torch
from torch import nn
import torch.nn.functional as F

from retinaface.commons.logger import Logger

logger = Logger(module="retinaface/model/retinaface_pth_model.py")

# pylint: disable=too-many-instance-attributes

# every batch normalization layer of the original model uses this epsilon
BN_EPS = 1.9999999494757503e-05


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=BN_EPS)


def _conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias=True):
    # tensorflow model applies explicit zero padding before valid convolutions
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


class ResidualUnit(nn.Module):
    """
    Pre-activation bottleneck unit of ResNet50 v2.
    Module names follow tensorflow layer names (e.g. stage1_unit1_bn1 -> stage1.unit1.bn1)
    """

    def __init__(self, in_channels: int, mid_channels: int, stride: int, shortcut: bool):
        super().__init__()
        out_channels = mid_channels * 4
        self.bn1 = _bn(in_channels)
        self.conv1 = _conv(in_channels, mid_channels, 1, bias=False)
        self.bn2 = _bn(mid_channels)
        self.conv2 = _conv(mid_channels, mid_channels, 3, stride=stride, bias=False)
        self.bn3 = _bn(mid_channels)
        self.conv3 = _conv(mid_channels, out_channels, 1, bias=False)
        self.sc = (
            _conv(in_channels, out_channels, 1, stride=stride, bias=False) if shortcut else None
        )

    def forward(self, x: torch.Tensor):
        """
        Returns unit output and its relu2 activation, which feeds the ssh lateral connections
        """
        relu1 = F.relu(self.bn1(x))
        relu2 = F.relu(self.bn2(self.conv1(relu1)))
        relu3 = F.relu(self.bn3(self.conv2(relu2)))
        out = self.conv3(relu3)
        shortcut = self.sc(relu1) if self.sc is not None else x
        return out + shortcut, relu2


class ContextModule(nn.Module):
    """
    SSH detection module. Module names follow tensorflow layer names
    (e.g. ssh_m1_det_context_conv3_1 -> ssh_m1_det.context_conv3_1)
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        half = channels // 2
        self.conv1 = _conv(channels, channels, 3)
        self.conv1_bn = _bn(channels)
        self.context_conv1 = _conv(channels, half, 3)
        self.context_conv1_bn = _bn(half)
        self.context_conv2 = _conv(half, half, 3)
        self.context_conv2_bn = _bn(half)
        self.context_conv3_1 = _conv(half, half, 3)
        self.context_conv3_1_bn = _bn(half)
        self.context_conv3_2 = _conv(half, half, 3)
        self.context_conv3_2_bn = _bn(half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1_bn(self.conv1(x))
        context1 = F.relu(self.context_conv1_bn(self.context_conv1(x)))
        context2 = self.context_conv2_bn(self.context_conv2(context1))
        context3 = F.relu(self.context_conv3_1_bn(self.context_conv3_1(context1)))
        context3 = self.context_conv3_2_bn(self.context_conv3_2(context3))
        return F.relu(torch.cat([conv1, context2, context3], dim=1))


class Head(nn.Module):
    """
    Classification, bounding box and landmark heads for a single stride
    """

    def __init__(self, channels: int = 512, num_anchors: int = 2):
        super().__init__()
        self.cls_score = nn.Conv2d(channels, num_anchors * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.landmark_pred = nn.Conv2d(channels, num_anchors * 10, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        score = self.cls_score(x)
        # channels are ordered as [bg_a1, bg_a2, face_a1, face_a2]
        # softmax is applied over background & face scores of each anchor
        bg, face = score[:, 0:2], score[:, 2:4]
        prob = torch.softmax(torch.stack([bg, face], dim=0), dim=0)
        prob = torch.cat([prob[0], prob[1]], dim=1)
        return [prob, self.bbox_pred(x), self.landmark_pred(x)]


def _crop_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Center crop x to the spatial size of target
    """
    h, w = target.shape[2], target.shape[3]
    dy = (x.shape[2] - h) // 2
    dx = (x.shape[3] - w) // 2
    return x[:, :, dy : dy + h, dx : dx + w]


class RetinaFace(nn.Module):
    """
    PyTorch implementation of RetinaFace with ResNet50 backbone.
    It is a port of retinaface/model/retinaface_model.py

    forward expects a float tensor in (N, H, W, 3) shape as the tensorflow model does and
        returns outputs in the same order and (N, H, W, C) layout of the tensorflow model:
        [cls_prob_stride32, bbox_pred_stride32, landmark_pred_stride32,
         cls_prob_stride16, bbox_pred_stride16, landmark_pred_stride16,
         cls_prob_stride8, bbox_pred_stride8, landmark_pred_stride8]
    """

    def __init__(self):
        super().__init__()
        self.bn_data = _bn(3)
        self.conv0 = _conv(3, 64, 7, stride=2, bias=False)
        self.bn0 = _bn(64)

        in_channels = 64
        for stage_idx, (mid_channels, units) in enumerate(
            [(64, 3), (128, 4), (256, 6), (512, 3)], start=1
        ):
            stage = nn.ModuleDict()
            for unit_idx in range(1, units + 1):
                first = unit_idx == 1
                stage[f"unit{unit_idx}"] = ResidualUnit(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    stride=2 if first and stage_idx > 1 else 1,
                    shortcut=first,
                )
                in_channels = mid_channels * 4
            setattr(self, f"stage{stage_idx}", stage)

        self.bn1 = _bn(2048)

        # feature pyramid
        self.ssh_c3_lateral = _conv(2048, 256, 1)
        self.ssh_c3_lateral_bn = _bn(256)
        self.ssh_c2_lateral = _conv(512, 256, 1)
        self.ssh_c2_lateral_bn = _bn(256)
        self.ssh_c2_aggr = _conv(256, 256, 3)
        self.ssh_c2_aggr_bn = _bn(256)
        self.ssh_m1_red_conv = _conv(256, 256, 1)
        self.ssh_m1_red_conv_bn = _bn(256)
        self.ssh_c1_aggr = _conv(256, 256, 3)
        self.ssh_c1_aggr_bn = _bn(256)

        # ssh context modules
        self.ssh_m3_det = ContextModule()
        self.ssh_m2_det = ContextModule()
        self.ssh_m1_det = ContextModule()

        # heads
        self.face_rpn_stride32 = Head()
        self.face_rpn_stride16 = Head()
        self.face_rpn_stride8 = Head()

    @staticmethod
    def _run_stage(stage: nn.ModuleDict, x: torch.Tensor):
        first_relu2 = None
        for unit in stage.values():
            x, relu2 = unit(x)
            if first_relu2 is None:
                first_relu2 = relu2
        return x, first_relu2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x = self.bn_data(x)
        x = F.relu(self.bn0(self.conv0(x)))
        # zero padding is used in the original model. it is safe because inputs are non-negative
        x = F.max_pool2d(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=2)

        x, _ = self._run_stage(self.stage1, x)
        x, _ = self._run_stage(self.stage2, x)
        x, stage3_unit1_relu2 = self._run_stage(self.stage3, x)
        x, stage4_unit1_relu2 = self._run_stage(self.stage4, x)
        x = F.relu(self.bn1(x))

        c3 = F.relu(self.ssh_c3_lateral_bn(self.ssh_c3_lateral(x)))
        c2 = F.relu(self.ssh_c2_lateral_bn(self.ssh_c2_lateral(stage4_unit1_relu2)))
        c1 = F.relu(self.ssh_m1_red_conv_bn(self.ssh_m1_red_conv(stage3_unit1_relu2)))

        c3_up = _crop_like(F.interpolate(c3, scale_factor=2, mode="nearest"), c2)
        c2 = F.relu(self.ssh_c2_aggr_bn(self.ssh_c2_aggr(c2 + c3_up)))

        c2_up = _crop_like(F.interpolate(c2, scale_factor=2, mode="nearest"), c1)
        c1 = F.relu(self.ssh_c1_aggr_bn(self.ssh_c1_aggr(c1 + c2_up)))

        outputs = []
        for features, context, head in [
            (c3, self.ssh_m3_det, self.face_rpn_stride32),
            (c2, self.ssh_m2_det, self.face_rpn_stride16),
            (c1, self.ssh_m1_det, self.face_rpn_stride8),
        ]:
            outputs.extend(head(context(features)))

        # NCHW -> NHWC to be compatible with tensorflow model's outputs
        return [output.permute(0, 2, 3, 1) for output in outputs]


def load_weights(model: RetinaFace) -> RetinaFace:
    """
    Loading pre-trained weights for the RetinaFace model
    Args:
        model (RetinaFace): retinaface model structure with random weights
    Returns:
        model (RetinaFace): retinaface model with its structure and pre-trained weights
    """
    home = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))
    exact_file = home + "/.deepface/weights/retinaface.pth"
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.pth"

    # -----------------------------

    if not os.path.exists(home + "/.deepface"):
        os.mkdir(home + "/.deepface")
        logger.info(f"Directory {home}/.deepface created")

    if not os.path.exists(home + "/.deepface/weights"):
        os.mkdir(home + "/.deepface/weights")
        logger.info(f"Directory {home}/.deepface/weights created")

    # -----------------------------

    if os.path.isfile(exact_file) is not True:
        logger.info(f"retinaface.pth will be downloaded from the url {url}")
        gdown.download(url, exact_file, quiet=False)

    # -----------------------------

    # gdown should download the pretrained weights here.
    # If it does not still exist, then throw an exception.
    if os.path.isfile(exact_file) is not True:
        raise ValueError(
            "Pre-trained weight could not be loaded!"
            + " You might try to download the pre-trained weights from the url "
            + url
            + f" and copy it to the {exact_file} manually."
        )

    model.load_state_dict(torch.load(exact_file, map_location="cpu", weights_only=True))
    return model


def build_model() -> RetinaFace:
    """
    Build RetinaFace model in pytorch
    """
    model = RetinaFace()
    model = load_weights(model)
    model.eval()
    return model
