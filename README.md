# f-trak
A face detection program that tracks face movement in screen space.

## Motivation
The desire from this has come from another project I am currently developing where knowing the location of a user's head in a 2D reference space is required. I thought it would be cool to split it out into its own side project. 

## Plan
The end goal is to release this as a Rust crate that tracks facial movement and can provide a bounding box containing the user's face in screen space. I intend to make use of a pretrained face detection neural network. A prototype has been written in python, based on an example by [Dr. Adrian Rosebrock](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/), which serves as the design for the first (possibly only) iteration.

Perhaps a better face detector could be built, but for my purposes I only require a simple detector. __Until v1.0 is released, f-trak will not be open to outside PRs__.
