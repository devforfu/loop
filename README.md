<p align="center">
<img src="./assets/loop.gif"/>
</p>

A simple implementation a Deep Learning models' training loop built on top of [`pytorch`](https://pytorch.org) with maximal compatibility with that framework in mind. Learning and entertainment alongside with robustness and clean code.

## Intro

The [`pytorch`](https://pytorch.org) framework provides a very clean and straightforward interface to build (Deep) Machine Learning models and read the datasets from a persistent storage. So let's use the best features of this great tool and write a set of thin and transparent wrappers on top of it to build a general-purpose training/validation loop that will be able to accept [`Dataset`](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset) and [`Module`](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module) instances, and run training process using modern Deep Learning training techniques.

## Roadmap

The features and improvements to be implemeted:

- [x] Basic implementation of training loop for CNN-based image classification models
- [ ] Simplify training loop instantiation
- [ ] Make possible to pass plain PyTorch classes and objects directly into the loop and utils
- [ ] More callbacks (early stopping, model saver, [`visdom`](https://github.com/facebookresearch/visdom) integration, etc.)
- [ ] CNN regression
- [ ] Smoke tests and sanity checks to verify the correctness of training process 
- [ ] Adding more examples and applications (Jupyter notebooks)
- [ ] Continuous integration
- [ ] Benchmarking on "classical" image datasets
- [ ] Basic set of image augmentations
- [ ] Basic RNN support
- [ ] Basic GAN support

## Dependencies

- psutil
- numpy
- pandas
- torch
- torchvision
- (dev only) pytest

## If you need something mature and robust

Please check the following projects (especially, the last one) if you would like to have something that is more suitable for production usage with less manual work and debugging:

  1. [Ignite](https://pytorch.org/ignite/) — an official high-level interface for PyTorch

  2. [Torchsample](https://github.com/ncullen93/torchsample) — a Keras-like wrapper with callbacks, augmentation, and handy utils

  3. [Skorch](https://github.com/dnouri/skorch) — a scikit-learn compatible neural network library

  4. [fastai](https://docs.fast.ai/) — a powerful end-to-end solution to train Deep Learning models of various complexity with high accuracy and computation speed

## Outro

The repository started as an author's attempt to write some simple solution to train an image classifier with modern Deep Learning training techniques as described in [this post](https://towardsdatascience.com/deep-learning-model-training-loop-e41055a24b73).
