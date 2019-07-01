Whisker Contact Quickstart
==========================

Welcome to the Whisker Contact quickstart! This document provides some basic information on how to get started on running the whisker contact detector on your own data as soon as possible. It assumes some basic working knowledge of Python, mostly of package management and virtual environment management.

Installation
------------

`Python >3.6` is required to run the scripts in this repo. The most recent version of Python is recommended, unless some packages required do not yet support it. It is always recommended that you use a specific virtual environment (conda environment) to install the packages required for this whisker contact analysis. [TODO: add instructions for Python and Conda]

A few basic Python modules must also be installed:

```
opencv, scipy, numpy, matplotlib, moviepy, tqdm, losswise, pillow, scikit-image, fbs
```

These modules can be installed through your Python package manager. If you're runnning Anaconda, try `conda install [PACKAGE]` before `pip install [PACKAGE]` [TODO: add specific commands/environment yaml file]. Many of these packages might already be installed if you have Conda or have done previous data science work.

A few more involved packages must also be installed, namely `keras` and `tensorflow`.

### Installing Tensorflow and Keras

If your system has no GPU, installation is easy:

```
pip install tensorflow keras
```

If your system has a GPU that you would like to use with Tensorflow, installation is much more involved. First, make sure that a recent driver version for your NVIDIA GPU is installed [TODO: add info about this]. Then, you have a few options.

If you're on Ubuntu, an easy way to install all required CUDA/cuDNN/tensorflow-gpu/etc without too much hassle is by using [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software).

If you're not on Ubuntu, or want to have more control over the install process, follow [Tensorflow's guide](https://tensorflow.org/install/gpu). Make sure the versions of the software that is installed is compatible. A helpful table is provided on the Tensorflow install guide.

Once the appropriate version of Tensorflow is installed, finish it up with a `pip install keras`.

Congrats! Your enviornment is set up.

Labeling Training Set
---------------------

Make sure that you have a folder with all relevant files from a recording session. For now, `trackedFeaturesRaw.csv` and `runAnalyzed.csv` are required (code will be updated).