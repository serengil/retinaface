# RetinaFace

[![Stars](https://img.shields.io/github/stars/serengil/retinaface)](https://github.com/serengil/retinaface)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/retinaface/blob/master/LICENSE)

RetinaFace is the face detection module of [insightface](https://github.com/deepinsight/insightface) project. The original implementation is mainly based on mxnet. Then, its tensorflow based [re-implementation](https://github.com/StanislasBertrand/RetinaFace-tf2) is published by [Stanislas Bertrand](https://github.com/StanislasBertrand).

This repo is heavily inspired from the study of Stanislas Bertrand. Its source code is simplified and it is transformed to pip compatible but the main structure of the reference model and its pre-trained weights are same.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/retinaface/master/tests/outputs/img1.jpg" width="90%" height="90%"></p>

Notice that face recognition module of insightface project is called as [ArcFace](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/).

## Installation

The easiest way to install retinaface is to download it from [pypi](https://pypi.org/project/retina-face/).

```
pip install retina-face
```

**Face Detection**

RetinaFace offers a face detection function. It expects an exact path of an image as input.

```python
from retina-face import RetinaFace
resp = RetinaFace.detect_faces("img1.jpg")
```

Then it returns the facial area coordinates and some landmarks (eyes, nose and mouth) with a confidence score.

```json
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
```

**Alignment**

A modern face recognition [pipeline](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: detect, [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [represent](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). Experiments show that alignment increases the face recognition accuracy almost 1%. Here, retinaface can find the facial landmarks including eye coordinates. In this way, it can apply alignment to detected faces with its extract faces function.

```python
import matplotlib.pyplot as plt
faces = RetinaFace.extract_faces(img_path = "img.jpg")
for face in faces:
  plt.imshow(img)
  plt.show()
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/retinaface/master/tests/outputs/img11-2.jpg" width="90%" height="90%"></p>

**FAQ and troubleshooting**

Pre-trained weights of the retinaface model is going to be downloaded from Google Drive once. Download limit of my Google Drive account might be exceeded sometimes. In this case, you will have an exception like **"too many users have viewed or downloaded this file recently. Please try accessing the file again later"**. Still, you can access the pre-trained weights on Google Drive. Please, download it [here](https://drive.google.com/uc?id=1K3Eq2k1b9dpKkucZjPAiCCnNzfCMosK4) and copy to the HOME/.deepface/weights folder manually.

You can find out your HOME_FOLDER with python as shown below.

```python
from pathlib import Path
home = str(Path.home())
print("HOME_FOLDER is ",home)
```

## Support

There are many ways to support a project. Starring⭐️ the repo is just one🙏

## Acknowledgements

This work is mainly based on the [insightface](https://github.com/deepinsight/insightface) project and [retinaface](https://arxiv.org/pdf/1905.00641.pdf) paper; and it is heavily inspired from the re-implementation of [retinaface-tf2](https://github.com/StanislasBertrand/RetinaFace-tf2) by [Stanislas Bertrand](https://github.com/StanislasBertrand). Finally, Bertrand's [implemenation](https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx) uses [Fast R-CNN](https://arxiv.org/abs/1504.08083) written by [Ross Girshick](https://github.com/rbgirshick/fast-rcnn) in the background. All of those reference studies licensed are under MIT license.

## Licence

This project is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/retinaface/blob/master/LICENSE) for more details.
