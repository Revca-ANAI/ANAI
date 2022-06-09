from setuptools import find_namespace_packages, setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="anai",
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
    version="prerelease-7",
    license="Apache License 2.0",
    description="Automated ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arsh Anwar",
    author_email="arsh.anwar@revca.io",
    keywords=["ANAI", "AutoML", "Python"],
    install_requires=[
        open("requirements.txt", "r").read().splitlines(),
        "missingpy @ git+https://github.com/d4rk-lucif3r/missingpy.git@0.2.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.9",
    ],
)
