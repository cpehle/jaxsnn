import os
from os import path

import setuptools
from setuptools import setup


pwd = path.abspath(path.dirname(__file__))

with open(path.join(pwd, "requirements.txt")) as fp:
    install_requires = fp.read()

with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

if os.name == "nt":
    compile_args = ["/std:c++17"]
else:
    compile_args = ["-O3"]


setup(
    install_requires=install_requires,
    setup_requires=["setuptools", "wheel"],
    name="jaxsnn",
    version="0.0.1",
    description="",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="http://github.com/cpehle/jaxsnn",
    author="Christian Pehle",
    author_email="christian.pehle@gmail.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning spiking neural networks",
)
