from setuptools import find_namespace_packages, setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="anai-opensource",
    packages=find_namespace_packages(
        exclude=[
            "build.*",
            "examples.*",
            "test/*",
            "dist/*",
            "dask-worker-space/*",
            "anai.egg-info",
            "anai_info",
        ],
        include=["anai.*", "anai"],
    ),
    version="0.1.2",
    license="Apache License 2.0",
    description="Automated ML",
    url="https://github.com/Revca-ANAI/ANAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Revca-ANAI",
    author_email="info@anai.io",
    keywords=["ANAI", "AutoML", "Python"],
    install_requires=[
        open("requirements.txt", "r").read().splitlines(),
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.9",
    ],
)
