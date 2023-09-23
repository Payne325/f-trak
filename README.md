# f-trak
[`f-trak`](https://crates.io/crates/f-trak) is a neural network based face detection program that tracks face movement in screen space. I originally built this as a cool means of controlling a player character in a POC game I made a while back called [`bongosero`](https://github.com/Payne325/bongosero). So it's only intended purpose is to report a single bbox back representing the portion of a camera frame containing a face.

## Design
`f-trak` makes use of a pretrained face detection neural network and `opencv's` `Deep Neural Network` module to find a face in an image frame captured from a video device.

A prototype was written in python, based on an example by [Dr. Adrian Rosebrock](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/), which serves as the design for the current iteration. A sample of this code is provided in this repository.

## Set up
`f-trak` is entirely dependent on the [`opencv-rust`](https://github.com/twistedfall/opencv-rust) crate. Please follow the set up procedure in their documentation.
 I found setting up for Linux a painless experience, but Windows is a tiny bit fiddly. 
 
### Windows Setup

It's worth noting that when compiling for windows the following environment variables must be set.

`OPENCV_DIR` `"$\opencvLocation\build\x64\vc15\lib"`

`OPENCV_INCLUDE_PATHS` `"$\opencvLocation\build\include"`

`OPENCV_LINK_PATHS` `"$\opencvLocation\build\x64\vc15\lib"`

`OPENCV_LINK_LIBS` `"opencv_world412"`

`Path` `"$\opencvLocation\build\x64\vc15\bin"`

Other environment variables may be needed as the documentation describes.
You'll also need to install llvm, the [`opencv-rust`](https://github.com/twistedfall/opencv-rust#windows-package) crate readme documentation explains further.
## How to use
`f-trak` is designed to be run on a separate thread and polled for the current location of a detected face. See the `f-trak-test` directory for an example application.
