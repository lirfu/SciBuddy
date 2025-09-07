# Scientist's Buddy
This repository contains a variety of utils for day-to-day work of a Computer Vision/Machine Learning scientist.

Hope you find something useful or interesting inside!

## Camera utils `scibuddy.camera`
Utilities for using the computer/laptop camera in your computer vision applications.

## Data utils `scibuddy.data`
Loaders and converters for images, pointclouds, meshes and more. Also tools for changing the dimension order of image tensors and image concatenation.

## Data structures `scibuddy.datastructures`
Generic data structure solutions for very specific use-cases.

## Logging utils `scibuddy.logging`
A flexible implementation of logging utilities for your generic Python needs. Construct your own logging pipeline by encapsulating the leaf loggers with various decorators. The global `LOG` is a singleton logger instance that can be accessed anywhere by just importing it (no need for passing loggers as a method argument).

## Optimization utils `scibuddy.optim`
Loss tracking and model checkpointing to reduce code clutter in optimization loops.

## Project utils `scibuddy.project`
Experiment tracking and hyperparameter searches for easier project management and development. Also other utils such as GIF generator.

## Matplotlib plotting utils `scibuddy.pyplot`
Various utils for plotting various results using matplotlib. Also, plotting contexts for cleaner and more reliable plotting code.

## Tools `scibuddy.tools`
General tools such as a timer and memory management.

## Torch utils `scibuddy.torch`
Torch specific tools, mainly regarding the experiment reproducibility (seeding of random generators).

## Vision utils `scibuddy.vision`
A bunch of convolution kernels an various other tools that are in the domain of computer vision.