# f-trak
[`f-trak`](https://crates.io/crates/f-trak) is a neural network based face detection program that tracks face movement in screen space.

__Please consider this program as in an experimental/alpha stage of development.__

## Motivation
The desire from this has come from another project I am currently developing where knowing the location of a user's head in a 2D reference space is required. I thought it would be cool to split it out into its own side project. 

## Design
The original goal was to release this as a Rust crate that tracks facial movement and can provide a bounding box containing the user's face in screen space. `f-trak` makes use of a pretrained face detection neural network and `opencv's` `Deep Neural Network` module to find a face in an image frame captured from a video device.

A prototype was written in python, based on an example by [Dr. Adrian Rosebrock](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/), which serves as the design for the current iteration.

Perhaps a better face detector could be built, but for the original requirement only a simple detector was necessary. 

## Set up
f-trak is entirely dependent on the [`opencv-rust`](https://github.com/twistedfall/opencv-rust) crate. Please follow the set up procedure in their documentation.
I've tested f-trak using OpenCV 4.1.2 so I'd recommend using that version, I plan to update the crate in future to support the latest version. Watch this space!

## How to use
f-trak is designed to be run on a separate thread and polled for the current location of a detected face. See the f-trak-test directory for an example application.
