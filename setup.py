from setuptools import setup, find_packages


version = "0.1.0"

requirements = [
    "numpy==1.26.4"
]

info = {
    "name": "utils",
    "version": version,
    "author": "Felix Frohnert",
    "description": "Utils to process data",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/FelixFrohnertDB/pole-learning",
    "license": "Apache 2.0",
    "provides": ["utils"],
    "install_requires": requirements,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["Machine Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)