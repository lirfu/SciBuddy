# Lirfu's Python Utils
This repository contains a variety of utils I use in day-to-day work. Between the deep learning frameworks, I use `torch` so some utils will require this dependency.

Hope you find something useful or interesting inside!

## Camera utils `lirfu_pyutils.camera`
Utilities for using the computer/laptop camera in your computer vision applications.

## Data utils `lirfu_pyutils.data`
Loaders and converters for images, pointclouds, meshes and more. Also tools for changing the dimension order of image tensors and image concatenation.

## Data structures `lirfu_pyutils.datastructures`
Generic data structure solutions for very specific use-cases.

## Logging utils `lirfu_pyutils.logging`
A flexible implementation of logging utilities for your generic Python needs. Construct your own logging pipeline by encapsulating the leaf loggers with various decorators. The global `LOG` is a singleton logger instance that can be accessed anywhere by just importing it (no need for passing loggers as a method argument).

## Optimization utils `lirfu_pyutils.optim`
Loss tracking and model checkpointing to reduce code clutter in optimization loops.

## Project utils `lirfu_pyutils.project`
Experiment tracking and hyperparameter searches for easier project management and development. Also other utils such as GIF generator.

## Matplotlib plotting utils `lirfu_pyutils.pyplot`
Various utils for plotting various results using matplotlib. Also, plotting contexts for cleaner and more reliable plotting code.

## Tools `lirfu_pyutils.tools`
General tools such as a timer and memory management.

## Torch utils `lirfu_pyutils.torch`
Torch specific tools, mainly regarding the experiment reproducibility (seeding of random generators).

## Vision utils `lirfu_pyutils.vision`
A bunch of convolution kernels an various other tools that are in the domain of computer vision.