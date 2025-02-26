import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="IDOL", # Replace with your own username
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires = [
        "wandb==0.16.1",
        "lightning==2.0.6",
        "pytorch-lightning==2.0.0",
        "torch==2.0.1", # 1.8.1
        "disentanglement-lib==1.4",
        "torchvision",
        "torchaudio",
        "h5py",
        "ipdb",
        "opencv-python",
        "pymunk",
    ],
    tests_require=[
        "pytest"
    ],
)