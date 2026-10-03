import json
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def parse_requirements(file_name):
    """Read a requirements file, skipping its blank lines and comments"""
    with open(file_name, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


# requirements.txt still includes tensorflow not to break `pip install retina-face` for
# existing users. switch it to requirements_base.txt in the next major release.
requirements = parse_requirements("requirements.txt")

# retinaface runs on tensorflow, pytorch or onnx. `pip install retina-face[tensorflow]`,
# `pip install retina-face[pytorch]` and `pip install retina-face[onnx]` install the backend
# engine you want, without dragging the others in.
tensorflow_requirements = parse_requirements("requirements_tf.txt")
pytorch_requirements = parse_requirements("requirements_pth.txt")
onnx_requirements = parse_requirements("requirements_onnx.txt")

with open("package_info.json", "r", encoding="utf-8") as f:
    package_info = json.load(f)


setuptools.setup(
    name="retina-face",  # pip install retina-face
    version=package_info["version"],
    author="Sefik Ilkin Serengil",
    author_email="serengil@gmail.com",
    description="RetinaFace: Deep Face Detection Framework in TensorFlow, PyTorch and ONNX for Python",
    data_files=[
        (
            "",
            [
                "README.md",
                "requirements.txt",
                "requirements_base.txt",
                "requirements_tf.txt",
                "requirements_pth.txt",
                "requirements_onnx.txt",
                "package_info.json",
            ],
        )
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serengil/retinaface",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5.5",
    install_requires=requirements,
    extras_require={
        "tensorflow": tensorflow_requirements,
        "pytorch": pytorch_requirements,
        "onnx": onnx_requirements,
    },
)
