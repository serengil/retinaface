import json
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

with open("package_info.json", "r", encoding="utf-8") as f:
    package_info = json.load(f)


setuptools.setup(
    name="retina-face",  # pip install retina-face
    version=package_info["version"],
    author="Sefik Ilkin Serengil",
    author_email="serengil@gmail.com",
    description="RetinaFace: Deep Face Detection Framework in TensorFlow for Python",
    data_files=[("", ["README.md", "requirements.txt", "package_info.json"])],
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
)
