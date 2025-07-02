from setuptools import setup, find_packages


version = "0.1.0"

requirements = [
    "catboost==1.2.7",
    "matplotlib==3.10.3",
    "numpy==2.3.1",
    "pandas==2.3.0",
    "scikit_learn==1.7.0",
    "scipy==1.16.0",
    "seaborn==0.13.2",
    "tqdm==4.66.5"]

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