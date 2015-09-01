# neural-style

This is a torch implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

The paper presents an algorithm for combining the content of one image with the style of another image using
convolutional neural networks. Here's an example that maps the artistic style of
[The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night)
onto a night-time photograph of the Stanford campus:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night.jpg" height="200px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/hoovertowernight.jpg" height="200px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/starry_stanford_big.png" width="706px">

## Setup:

Dependencies:
* [torch7](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)

Optional dependencies:
* CUDA 6.5+
* [cudnn.torch](https://github.com/soumith/cudnn.torch)

After installing dependencies, you'll need to run the following script to download the VGG model:
```
sh models/download_models.sh
```

## Usage
Basic usage:
```
th neural_style.lua -style_image <image.jpg> -content_image <image.jpg>
```

Options:
* `-image_size`: Maximum side length (in pixels) of of the generated image. Default is 512.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to -1.

Optimization options:
* `-content_weight`: How much to weight the content reconstruction term. Default is 8.0.
* `-style_weight`: How much to weight the style reconstruction term. Default is 1.0.
* `-num_iterations`: Default is 500.
* `-learning_rate`: Default is 10.0.
* `-momentum`: Default is 0.9.

Output options:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-print_iter`: Print progress every `print_iter` iterations. Set to 0 to disable printing.
* `-save_iter`: Save the image every `save_iter` iterations. Set to 0 to disable saving intermediate results.

Other options:
* `-proto_file`: Path to the `deploy.txt` file for the VGG Caffe model.
* `-model_file`: Path to the `.caffemodel` file for the VGG Caffe model.
* `-backend`: `nn` or `cudnn`. Default is `nn`. `cudnn` requires
  [cudnn.torch](https://github.com/soumith/cudnn.torch).


## Speed
On a GTX Titan X, running 1000 iterations of gradient descent with `-image_size=512` takes about 2 minutes.
In CPU mode on an Intel Core i7-4790k, running the same takes around 40 minutes.
Most of the examples shown here were run for 2000 iterations, but with a bit of parameter tuning most images will
give good results within 1000 iterations.

## Implementation details
Images are initialized with white noise, and optimized using gradient descent with momentum.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, and `conv4_1` layers and content reconstructions using the `conv2_1` layer. The losses at the four style reconstruction layers are weighted using
hand-tuned weights; these are probably not optimal but seem to give good results for a variety of images.
