# neural-style

This is a torch implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

The paper presents an algorithm for combining the content of one image with the style of another image using
convolutional neural networks. Here's an example that maps the artistic style of
[The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night)
onto a night-time photograph of the Stanford campus:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night.jpg" height="200px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/hoovertowernight.jpg" height="200px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/starry_stanford_big_2.png" width="706px">

Applying the style of different images to the same content image gives interesting results.
Here we reproduce Figure 2 from the paper, which renders a photograph of the Tubingen in Germany in a
variety of styles:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/tubingen.jpg" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_shipwreck.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_composition_vii.png" height="250px">

Here are the results of applying the style of various pieces of artwork to this photograph of the
golden gate bridge:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/golden_gate.jpg" height="200px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/frida_kahlo.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_kahlo.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/escher_sphere.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_escher.png" height="160px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/woman-with-hat-matisse.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_matisse.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/the_scream.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_scream.png" height="160px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_crop.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/seated-nude.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_seated.png" height="160px">

### Content / Style Tradeoff

The algorithm allows the user to trade-off the relative weight of the style and content reconstruction terms,
as shown in this example where we port the style of [Picasso's 1907 self-portrait](http://www.wikiart.org/en/pablo-picasso/self-portrait-1907) onto Brad Pitt:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/picasso_selfport1907.jpg" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/brad_pitt.jpg" height="220px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_5_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_1_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_01_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_0025_style_10.png" height="220px">

### Style Scale

By resizing the style image before extracting style features, we can control the types of artistic
features that are transfered from the style image; you can control this behavior with the `-style_scale` flag.
Below we see three examples of rendering the Golden Gate Bridge in the style of The Starry Night.
From left to right, `-style_scale` is 2.0, 1.0, and 0.5.

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale2.png" height=175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale1.png" height=175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale05.png" height=175px">

### Multiple Style Images
You can use more than one style image to blend multiple artistic styles.

Clockwise from upper left: "The Starry Night" + "The Scream", "The Scream" + "Composition VII",
"Seated Nude" + "Composition VII", and "Seated Nude" + "The Starry Night"

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_scream.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream_composition_vii.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_seated.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude_composition_vii.png" height="250px">



### Style Interpolation
When using multiple style images, you can control the degree to which they are blended:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_3_7.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_5_5.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_7_3.png" height="175px">


## Setup:

Dependencies:
* [torch7](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)

Optional dependencies:
* For CUDA backend:
  * CUDA 6.5+
  * [cunn](https://github.com/torch/cunn)
* For cuDNN backend:
  * [cudnn.torch](https://github.com/soumith/cudnn.torch)
* For OpenCL backend:
  * [cltorch](https://github.com/hughperkins/clnn)
  * [clnn](https://github.com/hughperkins/cltorch)

After installing dependencies, you'll need to run the following script to download the VGG model:
```
sh models/download_models.sh
```
This will download the original [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).
Leon Gatys has graciously provided the modified version of the VGG-19 model that was used in their paper;
this will also be downloaded. By default the original VGG-19 model is used.

If you have a smaller memory GPU then using NIN Imagenet model will be better and gives slightly worse yet comparable results. You can get the details on the model from [BVLC Caffe ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and can download the files from [NIN-Imagenet Download Link](https://drive.google.com/folderview?id=0B0IedYUunOQINEFtUi1QNWVhVVU&usp=drive_web)

You can find detailed installation instructions for Ubuntu in the [installation guide](INSTALL.md).

## Usage
Basic usage:
```
th neural_style.lua -style_image <image.jpg> -content_image <image.jpg>
```

OpenCL usage with NIN Model (This requires you download the NIN Imagenet model files as described above):
```
th neural_style.lua -style_image examples/inputs/picasso_selfport1907.jpg -content_image examples/inputs/brad_pitt.jpg -output_image profile.png -model_file models/nin_imagenet_conv.caffemodel -proto_file models/train_val.prototxt -gpu 0 -backend clnn -num_iterations 1000 -seed 123 -content_layers relu0,relu3,relu7,relu12 -style_layers relu0,relu3,relu7,relu12 -content_weight 10 -style_weight 1000 -image_size 512 -optimizer adam
```

![OpenCL NIN Model Picasso Brad Pitt](/examples/outputs/pitt_picasso_nin_opencl.png)


To use multiple style images, pass a comma-separated list like this:

`-style_image starry_night.jpg,the_scream.jpg`.

**Options**:
* `-image_size`: Maximum side length (in pixels) of of the generated image. Default is 512.
* `-style_blend_weights`: The weight for blending the style of multiple style images, as a
  comma-separated list, such as `-style_blend_weights 3,7`. By default all style images
  are equally weighted.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to -1.

**Optimization options**:
* `-content_weight`: How much to weight the content reconstruction term. Default is 5e0.
* `-style_weight`: How much to weight the style reconstruction term. Default is 1e2.
* `-tv_weight`: Weight of total-variation (TV) regularization; this helps to smooth the image.
  Default is 1e-3. Set to 0 to disable TV regularization.
* `-num_iterations`: Default is 1000.
* `-init`: Method for generating the generated image; one of `random` or `image`.
  Default is `random` which uses a noise initialization as in the paper; `image`
  initializes with the content image.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `lbfgs`.
  L-BFGS tends to give better results, but uses more memory. Switching to ADAM will reduce memory usage;
  when using ADAM you will probably need to play with other parameters to get good results, especially
  the style weight, content weight, and learning rate; you may also want to normalize gradients when
  using ADAM.
* `-learning_rate`: Learning rate to use with the ADAM optimizer. Default is 1e1.
* `-normalize_gradients`: If this flag is present, style and content gradients from each layer will be
  L1 normalized. Idea from [andersbll/neural_artistic_style](https://github.com/andersbll/neural_artistic_style).

**Output options**:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-print_iter`: Print progress every `print_iter` iterations. Set to 0 to disable printing.
* `-save_iter`: Save the image every `save_iter` iterations. Set to 0 to disable saving intermediate results.

**Layer options**:
* `-content_layers`: Comma-separated list of layer names to use for content reconstruction.
  Default is `relu4_2`.
* `-style_layers`: Comman-separated list of layer names to use for style reconstruction.
  Default is `relu1_1,relu2_1,relu3_1,relu4_1,relu5_1`.

**Other options**:
* `-style_scale`: Scale at which to extract features from the style image. Default is 1.0.
* `-proto_file`: Path to the `deploy.txt` file for the VGG Caffe model.
* `-model_file`: Path to the `.caffemodel` file for the VGG Caffe model.
  Default is the original VGG-19 model; you can also try the normalized VGG-19 model used in the paper.
* `-pooling`: The type of pooling layers to use; one of `max` or `avg`. Default is `max`.
  The VGG-19 models uses max pooling layers, but the paper mentions that replacing these layers with average
  pooling layers can improve the results. I haven't been able to get good results using average pooling, but
  the option is here.
* `-backend`: `nn`, `cudnn`, or `clnn`. Default is `nn`. `cudnn` requires
  [cudnn.torch](https://github.com/soumith/cudnn.torch) and may reduce memory usage.
  `clnn` requires [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)
* `-cudnn_autotune`: When using the cuDNN backend, pass this flag to use the built-in cuDNN autotuner to select
  the best convolution algorithms for your architecture. This will make the first iteration a bit slower and can
  take a bit more memory, but may significantly speed up the cuDNN backend.

## Frequently Asked Questions

**Problem:** Generated image has saturation artifacts:

<img src="https://cloud.githubusercontent.com/assets/1310570/9694690/fa8e8782-5328-11e5-9c91-11f7b215ad19.png">

**Solution:** Update the `image` packge to the latest version: `luarocks install image`

**Problem:** Running without a GPU gives an error message complaining about `cutorch` not found

**Solution:**
Pass the flag `-gpu -1` when running in CPU-only mode

**Problem:** The program runs out of memory and dies

**Solution:** Try reducing the image size: `-image_size 256` (or lower). Note that different image sizes will likely
require non-default values for `-style_weight` and `-content_weight` for optimal results.
If you are running on a GPU, you can also try running with `-backend cudnn` to reduce memory usage.

**Problem:** Get the following error message:

`models/VGG_ILSVRC_19_layers_deploy.prototxt.cpu.lua:7: attempt to call method 'ceil' (a nil value)`

**Solution:** Update `nn` package to the latest version: `luarocks install nn`

**Problem:** Get an error message complaining about `paths.extname`

**Solution:** Update `torch.paths` package to the latest version: `luarocks install paths`

**Problem:** NIN Imagenet model is not giving good results. 

**Solution:** Make sure the correct `-proto_file` is selected. Also make sure the correct parameters for `-content_layers` and `-style_layers` are set. (See OpenCL usage example above.)

**Problem:** `-backend cudnn` is slower than default NN backend

**Solution:** Add the flag `-cudnn_autotune`; this will use the built-in cuDNN autotuner to select the best convolution algorithms.

## Memory Usage
By default, `neural-style` uses the `nn` backend for convolutions and L-BFGS for optimization.
These give good results, but can both use a lot of memory. You can reduce memory usage with the following:

* **Use cuDNN**: Add the flag `-backend cudnn` to use the cuDNN backend. This will only work in GPU mode.
* **Use ADAM**: Add the flag `-optimizer adam` to use ADAM instead of L-BFGS. This should significantly
  reduce memory usage, but may require tuning of other parameters for good results; in particular you should
  play with the learning rate, content weight, style weight, and also consider using gradient normalization.
  This should work in both CPU and GPU modes.
* **Reduce image size**: If the above tricks are not enough, you can reduce the size of the generated image;
  pass the flag `-image_size 256` to generate an image at half the default size.
  
With the default settings, `neural-style` uses about 3.5GB of GPU memory on my system;
switching to ADAM and cuDNN reduces the GPU memory footprint to about 1GB.

## Speed
Speed can vary a lot depending on the backend and the optimizer.
Here are some times for running 500 iterations with `-image_size=512` on a GTX Titan X with different settings:
* `-backend nn -optimizer lbfgs`: 62 seconds
* `-backend nn -optimizer adam`: 49 seconds
* `-backend cudnn -optimizer lbfgs`: 79 seconds
* `-backend cudnn -cudnn_autotune -optimizer lbfgs`: 58 seconds
* `-backend cudnn -cudnn_autotune -optimizer adam`: 44 seconds
* `-backend clnn -optimizer lbfgs`: 169 seconds
* `-backend clnn -optimizer adam`: 106 seconds 

## Implementation details
Images are initialized with white noise and optimized using L-BFGS.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` layers
and content reconstructions using the `conv4_2` layer. As in the paper, the five style reconstruction losses have
equal weights.
