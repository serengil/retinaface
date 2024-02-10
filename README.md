# RetinaFace

<div align="center">

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/retina-face?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20downloads)](https://pepy.tech/project/retina-face)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/retina-face?color=green&label=conda%20downloads)](https://anaconda.org/conda-forge/retina-face)
[![Stars](https://img.shields.io/github/stars/serengil/retinaface?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/serengil/retinaface/stargazers)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/retinaface/blob/master/LICENSE)
[![Tests](https://github.com/serengil/retinaface/actions/workflows/tests.yml/badge.svg)](https://github.com/serengil/retinaface/actions/workflows/tests.yml)

[![Blog](https://img.shields.io/:blog-sefiks.com-blue.svg?style=flat&logo=wordpress)](https://sefiks.com)
[![YouTube](https://img.shields.io/:youtube-@sefiks-red.svg?style=flat&logo=youtube)](https://www.youtube.com/@sefiks?sub_confirmation=1)
[![Twitter](https://img.shields.io/:follow-@serengil-blue.svg?style=flat&logo=twitter)](https://twitter.com/intent/user?screen_name=serengil)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dserengil%26type%3Dpatrons&style=flat)](https://www.patreon.com/serengil?repo=retinaface)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/serengil?logo=GitHub&color=lightgray)](https://github.com/sponsors/serengil)

[![DOI](http://img.shields.io/:DOI-10.1109/ASYU50717.2020.9259802-blue.svg?style=flat)](https://doi.org/10.1109/ASYU50717.2020.9259802)
[![DOI](http://img.shields.io/:DOI-10.1109/ICEET53442.2021.9659697-blue.svg?style=flat)](https://doi.org/10.1109/ICEET53442.2021.9659697)

</div>

RetinaFace is a deep learning based cutting-edge facial detector for Python coming with facial landmarks. Its detection performance is amazing even in the crowd as shown in the following illustration.

RetinaFace is the face detection module of [insightface](https://github.com/deepinsight/insightface) project. The original implementation is mainly based on mxnet. Then, its tensorflow based [re-implementation](https://github.com/StanislasBertrand/RetinaFace-tf2) is published by [Stanislas Bertrand](https://github.com/StanislasBertrand). So, this repo is heavily inspired from the study of Stanislas Bertrand. Its source code is simplified and it is transformed to pip compatible but the main structure of the reference model and its pre-trained weights are same.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/retinaface/master/tests/outputs/img3.jpg" width="90%" height="90%">
<br><em>The Yellow Angels - Fenerbahce Women's Volleyball Team</em>
</p>

## Installation [![PyPI](https://img.shields.io/pypi/v/retina-face.svg)](https://pypi.org/project/retina-face/) [![Conda](https://img.shields.io/conda/vn/conda-forge/retina-face.svg)](https://anaconda.org/conda-forge/retina-face)

The easiest way to install retinaface is to download it from [PyPI](https://pypi.org/project/retina-face/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install retina-face
```

RetinaFace is also available at [`Conda`](https://anaconda.org/conda-forge/retina-face). You can alternatively install the package via conda.

```shell
$ conda install -c conda-forge retina-face
```

Then, you will be able to import the library and use its functionalities.

```python
from retinaface import RetinaFace
```

**Face Detection** - [`Demo`](https://youtu.be/Wm1DucuQk70)

RetinaFace offers a face detection function. It expects an exact path of an image as input.

```python
resp = RetinaFace.detect_faces("img1.jpg")
```

Then, it will return the facial area coordinates and some landmarks (eyes, nose and mouth) with a confidence score.

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

**Alignment** - [`Tutorial`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [`Demo`](https://youtu.be/WA9i68g4meI)

A modern face recognition [pipeline](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: detect, [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [normalize](https://sefiks.com/2020/11/20/facial-landmarks-for-face-recognition-with-dlib/), [represent](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). Experiments show that alignment increases the face recognition accuracy almost 1%. Here, retinaface can find the facial landmarks including eye coordinates. In this way, it can apply alignment to detected faces with its extracting faces function.

```python
import matplotlib.pyplot as plt
faces = RetinaFace.extract_faces(img_path = "img.jpg", align = True)
for face in faces:
  plt.imshow(face)
  plt.show()
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/retinaface/master/tests/outputs/alignment-procedure.png" width="80%" height="80%"></p>

If you prefer to prioritize alignment before detection, you may opt to set the `align_first` parameter to True. By following this approach, you will eliminate the black pixel areas that arise as a result of alignment following detection. This functionality is applicable only when the provided image contains a single face.

**Face Recognition** - [`Demo`](https://youtu.be/WnUVYQP4h44)

Notice that face recognition module of insightface project is [ArcFace](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), and face detection module is RetinaFace. ArcFace and RetinaFace pair is wrapped in [deepface](https://github.com/serengil/deepface) library for Python. Consider to use deepface if you need an end-to-end face recognition pipeline.

```python
#!pip install deepface
from deepface import DeepFace
obj = DeepFace.verify("img1.jpg", "img2.jpg"
          , model_name = 'ArcFace', detector_backend = 'retinaface')
print(obj["verified"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/retinaface/master/tests/outputs/retinaface-arcface.png" width="100%" height="100%"></p>

Notice that ArcFace got 99.40% accuracy on [LFW data set](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/) whereas human beings just have 97.53% confidence.

## Contribution [![Tests](https://github.com/serengil/retinaface/actions/workflows/tests.yml/badge.svg)](https://github.com/serengil/retinaface/actions/workflows/tests.yml)

Pull requests are more than welcome! You should run the unit tests and linting locally before creating a PR. Commands `make test` and `make lint` will help you to run it locally. Once a PR created, GitHub test workflow will be run automatically and unit test results will be available in [GitHub actions](https://github.com/serengil/retinaface/actions) before approval.

## Support

There are many ways to support a project. Starring⭐️ the repo is just one 🙏

You can also support this work on [Patreon](https://www.patreon.com/serengil?repo=retinaface)

<a href="https://www.patreon.com/serengil?repo=retinaface">
<img src="https://raw.githubusercontent.com/serengil/retinaface/master/icons/patreon.png" width="30%" height="30%">
</a>

## Acknowledgements

This work is mainly based on the [insightface](https://github.com/deepinsight/insightface) project and [retinaface](https://arxiv.org/pdf/1905.00641.pdf) paper; and it is heavily inspired from the re-implementation of [retinaface-tf2](https://github.com/StanislasBertrand/RetinaFace-tf2) by [Stanislas Bertrand](https://github.com/StanislasBertrand). Finally, Bertrand's [implementation](https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx) uses [Fast R-CNN](https://arxiv.org/abs/1504.08083) written by [Ross Girshick](https://github.com/rbgirshick/fast-rcnn) in the background. All of those reference studies are licensed under MIT license.

## Citation

If you are using RetinaFace in your research, please consider to cite its [original research paper](https://arxiv.org/abs/1905.00641). Besides, if you are using this re-implementation of retinaface, please consider to cite the following research papers as well. Here are examples of BibTeX entries:

```BibTeX
@inproceedings{serengil2020lightface,
  title        = {LightFace: A Hybrid Deep Face Recognition Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages        = {23-27},
  year         = {2020},
  doi          = {10.1109/ASYU50717.2020.9259802},
  url          = {https://doi.org/10.1109/ASYU50717.2020.9259802},
  organization = {IEEE}
}
```

```BibTeX
@inproceedings{serengil2021lightface,
  title        = {HyperExtended LightFace: A Facial Attribute Analysis Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2021 International Conference on Engineering and Emerging Technologies (ICEET)},
  pages        = {1-4},
  year         = {2021},
  doi          = {10.1109/ICEET53442.2021.9659697},
  url          = {https://doi.org/10.1109/ICEET53442.2021.9659697},
  organization = {IEEE}
}
```

Finally, if you use this RetinaFace re-implementation in your GitHub projects, please add `retina-face` dependency in the requirements.txt.

## Licence

This project is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/retinaface/blob/master/LICENSE) for more details.
