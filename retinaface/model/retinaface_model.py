import os
from pathlib import Path
import gdown
import tensorflow as tf
from retinaface.commons.logger import Logger

logger = Logger(module="retinaface/model/retinaface_model.py")

# pylint: disable=too-many-statements, no-name-in-module

# configurations

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model
    from keras.layers import (
        Input,
        BatchNormalization,
        ZeroPadding2D,
        Conv2D,
        ReLU,
        MaxPool2D,
        Add,
        UpSampling2D,
        concatenate,
        Softmax,
    )

else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        BatchNormalization,
        ZeroPadding2D,
        Conv2D,
        ReLU,
        MaxPool2D,
        Add,
        UpSampling2D,
        concatenate,
        Softmax,
    )


def load_weights(model: Model):
    """
    Loading pre-trained weights for the RetinaFace model
    Args:
        model (Model): retinaface model structure with randon weights
    Returns:
        model (Model): retinaface model with its structure and pre-trained weights

    """
    home = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))

    exact_file = home + "/.deepface/weights/retinaface.h5"
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"

    # -----------------------------

    if not os.path.exists(home + "/.deepface"):
        os.mkdir(home + "/.deepface")
        logger.info(f"Directory {home}/.deepface created")

    if not os.path.exists(home + "/.deepface/weights"):
        os.mkdir(home + "/.deepface/weights")
        logger.info(f"Directory {home}/.deepface/weights created")

    # -----------------------------

    if os.path.isfile(exact_file) is not True:
        logger.info(f"retinaface.h5 will be downloaded from the url {url}")
        gdown.download(url, exact_file, quiet=False)

    # -----------------------------

    # gdown should download the pretrained weights here.
    # If it does not still exist, then throw an exception.
    if os.path.isfile(exact_file) is not True:
        raise ValueError(
            "Pre-trained weight could not be loaded!"
            + " You might try to download the pre-trained weights from the url "
            + url
            + " and copy it to the ",
            exact_file,
            "manually.",
        )

    model.load_weights(exact_file)

    return model


def build_model() -> Model:
    """
    Build RetinaFace model
    """
    data = Input(dtype=tf.float32, shape=(None, None, 3), name="data")

    bn_data = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn_data", trainable=False)(
        data
    )

    conv0_pad = ZeroPadding2D(padding=tuple([3, 3]))(bn_data)

    conv0 = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        name="conv0",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(conv0_pad)

    bn0 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn0", trainable=False)(conv0)

    relu0 = ReLU(name="relu0")(bn0)

    pooling0_pad = ZeroPadding2D(padding=tuple([1, 1]))(relu0)

    pooling0 = MaxPool2D((3, 3), (2, 2), padding="valid", name="pooling0")(pooling0_pad)

    stage1_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn1", trainable=False
    )(pooling0)

    stage1_unit1_relu1 = ReLU(name="stage1_unit1_relu1")(stage1_unit1_bn1)

    stage1_unit1_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    stage1_unit1_sc = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_sc",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    stage1_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn2", trainable=False
    )(stage1_unit1_conv1)

    stage1_unit1_relu2 = ReLU(name="stage1_unit1_relu2")(stage1_unit1_bn2)

    stage1_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit1_relu2)

    stage1_unit1_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit1_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_conv2_pad)

    stage1_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn3", trainable=False
    )(stage1_unit1_conv2)

    stage1_unit1_relu3 = ReLU(name="stage1_unit1_relu3")(stage1_unit1_bn3)

    stage1_unit1_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu3)

    plus0_v1 = Add()([stage1_unit1_conv3, stage1_unit1_sc])

    stage1_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn1", trainable=False
    )(plus0_v1)

    stage1_unit2_relu1 = ReLU(name="stage1_unit2_relu1")(stage1_unit2_bn1)

    stage1_unit2_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu1)

    stage1_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn2", trainable=False
    )(stage1_unit2_conv1)

    stage1_unit2_relu2 = ReLU(name="stage1_unit2_relu2")(stage1_unit2_bn2)

    stage1_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit2_relu2)

    stage1_unit2_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_conv2_pad)

    stage1_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn3", trainable=False
    )(stage1_unit2_conv2)

    stage1_unit2_relu3 = ReLU(name="stage1_unit2_relu3")(stage1_unit2_bn3)

    stage1_unit2_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu3)

    plus1_v2 = Add()([stage1_unit2_conv3, plus0_v1])

    stage1_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn1", trainable=False
    )(plus1_v2)

    stage1_unit3_relu1 = ReLU(name="stage1_unit3_relu1")(stage1_unit3_bn1)

    stage1_unit3_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu1)

    stage1_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn2", trainable=False
    )(stage1_unit3_conv1)

    stage1_unit3_relu2 = ReLU(name="stage1_unit3_relu2")(stage1_unit3_bn2)

    stage1_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit3_relu2)

    stage1_unit3_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_conv2_pad)

    stage1_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn3", trainable=False
    )(stage1_unit3_conv2)

    stage1_unit3_relu3 = ReLU(name="stage1_unit3_relu3")(stage1_unit3_bn3)

    stage1_unit3_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu3)

    plus2 = Add()([stage1_unit3_conv3, plus1_v2])

    stage2_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn1", trainable=False
    )(plus2)

    stage2_unit1_relu1 = ReLU(name="stage2_unit1_relu1")(stage2_unit1_bn1)

    stage2_unit1_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_sc = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn2", trainable=False
    )(stage2_unit1_conv1)

    stage2_unit1_relu2 = ReLU(name="stage2_unit1_relu2")(stage2_unit1_bn2)

    stage2_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit1_relu2)

    stage2_unit1_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_conv2_pad)

    stage2_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn3", trainable=False
    )(stage2_unit1_conv2)

    stage2_unit1_relu3 = ReLU(name="stage2_unit1_relu3")(stage2_unit1_bn3)

    stage2_unit1_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu3)

    plus3 = Add()([stage2_unit1_conv3, stage2_unit1_sc])

    stage2_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn1", trainable=False
    )(plus3)

    stage2_unit2_relu1 = ReLU(name="stage2_unit2_relu1")(stage2_unit2_bn1)

    stage2_unit2_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu1)

    stage2_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn2", trainable=False
    )(stage2_unit2_conv1)

    stage2_unit2_relu2 = ReLU(name="stage2_unit2_relu2")(stage2_unit2_bn2)

    stage2_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit2_relu2)

    stage2_unit2_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_conv2_pad)

    stage2_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn3", trainable=False
    )(stage2_unit2_conv2)

    stage2_unit2_relu3 = ReLU(name="stage2_unit2_relu3")(stage2_unit2_bn3)

    stage2_unit2_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu3)

    plus4 = Add()([stage2_unit2_conv3, plus3])

    stage2_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn1", trainable=False
    )(plus4)

    stage2_unit3_relu1 = ReLU(name="stage2_unit3_relu1")(stage2_unit3_bn1)

    stage2_unit3_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu1)

    stage2_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn2", trainable=False
    )(stage2_unit3_conv1)

    stage2_unit3_relu2 = ReLU(name="stage2_unit3_relu2")(stage2_unit3_bn2)

    stage2_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit3_relu2)

    stage2_unit3_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_conv2_pad)

    stage2_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn3", trainable=False
    )(stage2_unit3_conv2)

    stage2_unit3_relu3 = ReLU(name="stage2_unit3_relu3")(stage2_unit3_bn3)

    stage2_unit3_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu3)

    plus5 = Add()([stage2_unit3_conv3, plus4])

    stage2_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn1", trainable=False
    )(plus5)

    stage2_unit4_relu1 = ReLU(name="stage2_unit4_relu1")(stage2_unit4_bn1)

    stage2_unit4_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu1)

    stage2_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn2", trainable=False
    )(stage2_unit4_conv1)

    stage2_unit4_relu2 = ReLU(name="stage2_unit4_relu2")(stage2_unit4_bn2)

    stage2_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit4_relu2)

    stage2_unit4_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_conv2_pad)

    stage2_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn3", trainable=False
    )(stage2_unit4_conv2)

    stage2_unit4_relu3 = ReLU(name="stage2_unit4_relu3")(stage2_unit4_bn3)

    stage2_unit4_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu3)

    plus6 = Add()([stage2_unit4_conv3, plus5])

    stage3_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn1", trainable=False
    )(plus6)

    stage3_unit1_relu1 = ReLU(name="stage3_unit1_relu1")(stage3_unit1_bn1)

    stage3_unit1_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_sc = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn2", trainable=False
    )(stage3_unit1_conv1)

    stage3_unit1_relu2 = ReLU(name="stage3_unit1_relu2")(stage3_unit1_bn2)

    stage3_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit1_relu2)

    stage3_unit1_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_conv2_pad)

    ssh_m1_red_conv = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_m1_red_conv",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage3_unit1_relu2)

    stage3_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn3", trainable=False
    )(stage3_unit1_conv2)

    ssh_m1_red_conv_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_red_conv_bn", trainable=False
    )(ssh_m1_red_conv)

    stage3_unit1_relu3 = ReLU(name="stage3_unit1_relu3")(stage3_unit1_bn3)

    ssh_m1_red_conv_relu = ReLU(name="ssh_m1_red_conv_relu")(ssh_m1_red_conv_bn)

    stage3_unit1_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu3)

    plus7 = Add()([stage3_unit1_conv3, stage3_unit1_sc])

    stage3_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn1", trainable=False
    )(plus7)

    stage3_unit2_relu1 = ReLU(name="stage3_unit2_relu1")(stage3_unit2_bn1)

    stage3_unit2_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu1)

    stage3_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn2", trainable=False
    )(stage3_unit2_conv1)

    stage3_unit2_relu2 = ReLU(name="stage3_unit2_relu2")(stage3_unit2_bn2)

    stage3_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit2_relu2)

    stage3_unit2_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_conv2_pad)

    stage3_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn3", trainable=False
    )(stage3_unit2_conv2)

    stage3_unit2_relu3 = ReLU(name="stage3_unit2_relu3")(stage3_unit2_bn3)

    stage3_unit2_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu3)

    plus8 = Add()([stage3_unit2_conv3, plus7])

    stage3_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn1", trainable=False
    )(plus8)

    stage3_unit3_relu1 = ReLU(name="stage3_unit3_relu1")(stage3_unit3_bn1)

    stage3_unit3_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu1)

    stage3_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn2", trainable=False
    )(stage3_unit3_conv1)

    stage3_unit3_relu2 = ReLU(name="stage3_unit3_relu2")(stage3_unit3_bn2)

    stage3_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit3_relu2)

    stage3_unit3_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_conv2_pad)

    stage3_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn3", trainable=False
    )(stage3_unit3_conv2)

    stage3_unit3_relu3 = ReLU(name="stage3_unit3_relu3")(stage3_unit3_bn3)

    stage3_unit3_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu3)

    plus9 = Add()([stage3_unit3_conv3, plus8])

    stage3_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn1", trainable=False
    )(plus9)

    stage3_unit4_relu1 = ReLU(name="stage3_unit4_relu1")(stage3_unit4_bn1)

    stage3_unit4_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu1)

    stage3_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn2", trainable=False
    )(stage3_unit4_conv1)

    stage3_unit4_relu2 = ReLU(name="stage3_unit4_relu2")(stage3_unit4_bn2)

    stage3_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit4_relu2)

    stage3_unit4_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_conv2_pad)

    stage3_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn3", trainable=False
    )(stage3_unit4_conv2)

    stage3_unit4_relu3 = ReLU(name="stage3_unit4_relu3")(stage3_unit4_bn3)

    stage3_unit4_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu3)

    plus10 = Add()([stage3_unit4_conv3, plus9])

    stage3_unit5_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn1", trainable=False
    )(plus10)

    stage3_unit5_relu1 = ReLU(name="stage3_unit5_relu1")(stage3_unit5_bn1)

    stage3_unit5_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit5_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu1)

    stage3_unit5_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn2", trainable=False
    )(stage3_unit5_conv1)

    stage3_unit5_relu2 = ReLU(name="stage3_unit5_relu2")(stage3_unit5_bn2)

    stage3_unit5_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit5_relu2)

    stage3_unit5_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit5_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_conv2_pad)

    stage3_unit5_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn3", trainable=False
    )(stage3_unit5_conv2)

    stage3_unit5_relu3 = ReLU(name="stage3_unit5_relu3")(stage3_unit5_bn3)

    stage3_unit5_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit5_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu3)

    plus11 = Add()([stage3_unit5_conv3, plus10])

    stage3_unit6_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn1", trainable=False
    )(plus11)

    stage3_unit6_relu1 = ReLU(name="stage3_unit6_relu1")(stage3_unit6_bn1)

    stage3_unit6_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit6_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu1)

    stage3_unit6_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn2", trainable=False
    )(stage3_unit6_conv1)

    stage3_unit6_relu2 = ReLU(name="stage3_unit6_relu2")(stage3_unit6_bn2)

    stage3_unit6_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit6_relu2)

    stage3_unit6_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit6_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_conv2_pad)

    stage3_unit6_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn3", trainable=False
    )(stage3_unit6_conv2)

    stage3_unit6_relu3 = ReLU(name="stage3_unit6_relu3")(stage3_unit6_bn3)

    stage3_unit6_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit6_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu3)

    plus12 = Add()([stage3_unit6_conv3, plus11])

    stage4_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn1", trainable=False
    )(plus12)

    stage4_unit1_relu1 = ReLU(name="stage4_unit1_relu1")(stage4_unit1_bn1)

    stage4_unit1_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_sc = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn2", trainable=False
    )(stage4_unit1_conv1)

    stage4_unit1_relu2 = ReLU(name="stage4_unit1_relu2")(stage4_unit1_bn2)

    stage4_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit1_relu2)

    stage4_unit1_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_conv2_pad)

    ssh_c2_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c2_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage4_unit1_relu2)

    stage4_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn3", trainable=False
    )(stage4_unit1_conv2)

    ssh_c2_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_lateral_bn", trainable=False
    )(ssh_c2_lateral)

    stage4_unit1_relu3 = ReLU(name="stage4_unit1_relu3")(stage4_unit1_bn3)

    ssh_c2_lateral_relu = ReLU(name="ssh_c2_lateral_relu")(ssh_c2_lateral_bn)

    stage4_unit1_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu3)

    plus13 = Add()([stage4_unit1_conv3, stage4_unit1_sc])

    stage4_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn1", trainable=False
    )(plus13)

    stage4_unit2_relu1 = ReLU(name="stage4_unit2_relu1")(stage4_unit2_bn1)

    stage4_unit2_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu1)

    stage4_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn2", trainable=False
    )(stage4_unit2_conv1)

    stage4_unit2_relu2 = ReLU(name="stage4_unit2_relu2")(stage4_unit2_bn2)

    stage4_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit2_relu2)

    stage4_unit2_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_conv2_pad)

    stage4_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn3", trainable=False
    )(stage4_unit2_conv2)

    stage4_unit2_relu3 = ReLU(name="stage4_unit2_relu3")(stage4_unit2_bn3)

    stage4_unit2_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu3)

    plus14 = Add()([stage4_unit2_conv3, plus13])

    stage4_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn1", trainable=False
    )(plus14)

    stage4_unit3_relu1 = ReLU(name="stage4_unit3_relu1")(stage4_unit3_bn1)

    stage4_unit3_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu1)

    stage4_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn2", trainable=False
    )(stage4_unit3_conv1)

    stage4_unit3_relu2 = ReLU(name="stage4_unit3_relu2")(stage4_unit3_bn2)

    stage4_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit3_relu2)

    stage4_unit3_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_conv2_pad)

    stage4_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn3", trainable=False
    )(stage4_unit3_conv2)

    stage4_unit3_relu3 = ReLU(name="stage4_unit3_relu3")(stage4_unit3_bn3)

    stage4_unit3_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu3)

    plus15 = Add()([stage4_unit3_conv3, plus14])

    bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn1", trainable=False)(plus15)

    relu1 = ReLU(name="relu1")(bn1)

    ssh_c3_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c3_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(relu1)

    ssh_c3_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c3_lateral_bn", trainable=False
    )(ssh_c3_lateral)

    ssh_c3_lateral_relu = ReLU(name="ssh_c3_lateral_relu")(ssh_c3_lateral_bn)

    ssh_m3_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m3_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_conv1_pad)

    ssh_m3_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv1_pad)

    ssh_c3_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_c3_up")(
        ssh_c3_lateral_relu
    )

    ssh_m3_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_conv1_bn", trainable=False
    )(ssh_m3_det_conv1)

    ssh_m3_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv1_bn", trainable=False
    )(ssh_m3_det_context_conv1)

    x1_shape = tf.shape(ssh_c3_up)
    x2_shape = tf.shape(ssh_c2_lateral_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop0 = tf.slice(ssh_c3_up, offsets, size, "crop0")

    ssh_m3_det_context_conv1_relu = ReLU(name="ssh_m3_det_context_conv1_relu")(
        ssh_m3_det_context_conv1_bn
    )

    plus0_v2 = Add()([ssh_c2_lateral_relu, crop0])

    ssh_m3_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv2_pad)

    ssh_m3_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_1_pad)

    ssh_c2_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus0_v2)

    ssh_c2_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c2_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c2_aggr_pad)

    ssh_m3_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv2_bn", trainable=False
    )(ssh_m3_det_context_conv2)

    ssh_m3_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_1_bn", trainable=False
    )(ssh_m3_det_context_conv3_1)

    ssh_c2_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_aggr_bn", trainable=False
    )(ssh_c2_aggr)

    ssh_m3_det_context_conv3_1_relu = ReLU(name="ssh_m3_det_context_conv3_1_relu")(
        ssh_m3_det_context_conv3_1_bn
    )

    ssh_c2_aggr_relu = ReLU(name="ssh_c2_aggr_relu")(ssh_c2_aggr_bn)

    ssh_m3_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv3_1_relu
    )

    ssh_m3_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_2_pad)

    ssh_m2_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m2_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_conv1_pad)

    ssh_m2_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv1_pad)

    ssh_m2_red_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_m2_red_up")(
        ssh_c2_aggr_relu
    )

    ssh_m3_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_2_bn", trainable=False
    )(ssh_m3_det_context_conv3_2)

    ssh_m2_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_conv1_bn", trainable=False
    )(ssh_m2_det_conv1)

    ssh_m2_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv1_bn", trainable=False
    )(ssh_m2_det_context_conv1)

    x1_shape = tf.shape(ssh_m2_red_up)
    x2_shape = tf.shape(ssh_m1_red_conv_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop1 = tf.slice(ssh_m2_red_up, offsets, size, "crop1")

    ssh_m3_det_concat = concatenate(
        [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn],
        3,
        name="ssh_m3_det_concat",
    )

    ssh_m2_det_context_conv1_relu = ReLU(name="ssh_m2_det_context_conv1_relu")(
        ssh_m2_det_context_conv1_bn
    )

    plus1_v1 = Add()([ssh_m1_red_conv_relu, crop1])

    ssh_m3_det_concat_relu = ReLU(name="ssh_m3_det_concat_relu")(ssh_m3_det_concat)

    ssh_m2_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv2_pad)

    ssh_m2_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_1_pad)

    ssh_c1_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus1_v1)

    ssh_c1_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c1_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c1_aggr_pad)

    face_rpn_cls_score_stride32 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride32"
    )

    face_rpn_bbox_pred_stride32 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    face_rpn_landmark_pred_stride32 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    ssh_m2_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv2_bn", trainable=False
    )(ssh_m2_det_context_conv2)

    ssh_m2_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_1_bn", trainable=False
    )(ssh_m2_det_context_conv3_1)

    ssh_c1_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c1_aggr_bn", trainable=False
    )(ssh_c1_aggr)

    ssh_m2_det_context_conv3_1_relu = ReLU(name="ssh_m2_det_context_conv3_1_relu")(
        ssh_m2_det_context_conv3_1_bn
    )

    ssh_c1_aggr_relu = ReLU(name="ssh_c1_aggr_relu")(ssh_c1_aggr_bn)

    face_rpn_cls_prob_stride32 = Softmax(name="face_rpn_cls_prob_stride32")(
        face_rpn_cls_score_reshape_stride32
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride32)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride32[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride32[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride32[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride32[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride32"
    )

    ssh_m2_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv3_1_relu
    )

    ssh_m2_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_2_pad)

    ssh_m1_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m1_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_conv1_pad)

    ssh_m1_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv1_pad)

    ssh_m2_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_2_bn", trainable=False
    )(ssh_m2_det_context_conv3_2)

    ssh_m1_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_conv1_bn", trainable=False
    )(ssh_m1_det_conv1)

    ssh_m1_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv1_bn", trainable=False
    )(ssh_m1_det_context_conv1)

    ssh_m2_det_concat = concatenate(
        [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn],
        3,
        name="ssh_m2_det_concat",
    )

    ssh_m1_det_context_conv1_relu = ReLU(name="ssh_m1_det_context_conv1_relu")(
        ssh_m1_det_context_conv1_bn
    )

    ssh_m2_det_concat_relu = ReLU(name="ssh_m2_det_concat_relu")(ssh_m2_det_concat)

    ssh_m1_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv2_pad)

    ssh_m1_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_1_pad)

    face_rpn_cls_score_stride16 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 0], face_rpn_cls_score_stride16[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 2], face_rpn_cls_score_stride16[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride16"
    )

    face_rpn_bbox_pred_stride16 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    face_rpn_landmark_pred_stride16 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    ssh_m1_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv2_bn", trainable=False
    )(ssh_m1_det_context_conv2)

    ssh_m1_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_1_bn", trainable=False
    )(ssh_m1_det_context_conv3_1)

    ssh_m1_det_context_conv3_1_relu = ReLU(name="ssh_m1_det_context_conv3_1_relu")(
        ssh_m1_det_context_conv3_1_bn
    )

    face_rpn_cls_prob_stride16 = Softmax(name="face_rpn_cls_prob_stride16")(
        face_rpn_cls_score_reshape_stride16
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride16)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride16[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride16[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride16[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride16[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride16"
    )

    ssh_m1_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv3_1_relu
    )

    ssh_m1_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_2_pad)

    ssh_m1_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_2_bn", trainable=False
    )(ssh_m1_det_context_conv3_2)

    ssh_m1_det_concat = concatenate(
        [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn],
        3,
        name="ssh_m1_det_concat",
    )

    ssh_m1_det_concat_relu = ReLU(name="ssh_m1_det_concat_relu")(ssh_m1_det_concat)
    face_rpn_cls_score_stride8 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 0], face_rpn_cls_score_stride8[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 2], face_rpn_cls_score_stride8[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride8"
    )

    face_rpn_bbox_pred_stride8 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_landmark_pred_stride8 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_cls_prob_stride8 = Softmax(name="face_rpn_cls_prob_stride8")(
        face_rpn_cls_score_reshape_stride8
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride8)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride8[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride8[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride8[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride8[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride8"
    )

    model = Model(
        inputs=data,
        outputs=[
            face_rpn_cls_prob_reshape_stride32,
            face_rpn_bbox_pred_stride32,
            face_rpn_landmark_pred_stride32,
            face_rpn_cls_prob_reshape_stride16,
            face_rpn_bbox_pred_stride16,
            face_rpn_landmark_pred_stride16,
            face_rpn_cls_prob_reshape_stride8,
            face_rpn_bbox_pred_stride8,
            face_rpn_landmark_pred_stride8,
        ],
    )
    model = load_weights(model)

    return model
