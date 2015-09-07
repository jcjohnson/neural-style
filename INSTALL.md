#neural-style Installation

This guide will walk you through the setup for `neural-style` on Ubuntu.

## Step 1: Install torch7

First we need to install torch, following the installation instructions
[here](http://torch.ch/docs/getting-started.html#_):

```
# in a terminal, run the commands
cd ~/
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
```

The first script installs all dependencies for torch and may take a while.
The second script actually installs lua and torch.
The second script also edits your `.bashrc` file so that torch is added to your `PATH` variable;
we need to source it to refresh our environment variables:

```
source ~/.bashrc
```

To check that your torch installation is working, run the command `th` to enter the interactive shell.
To quit just type `exit`.


## Step 2: Install loadcaffe

`loadcaffe` depends on [Google's Protocol Buffer library](https://developers.google.com/protocol-buffers/?hl=en)
so we'll need to install that first:

```
sudo apt-get install libprotobuf-dev protobuf-compiler
```

Now we can instal `loadcaffe`:

```
luarocks install loadcaffe
```

## Step 3: Install neural-style

First we clone `neural-style` from GitHub:

```
cd ~/
git clone https://github.com/jcjohnson/neural-style.git
cd neural-style
```

Next we need to download the pretrained neural network models:

```
sh models/download_models.sh
```

You should now be able to run `neural-style` in CPU mode like this:

```
th neural_style.lua -gpu -1 -print_iter -1
```

If everything is working properly you should see output like this:

```
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:505] Reading dangerously large protocol message.  If the message turns out to be larger than 1073741824 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 574671192
Successfully loaded models/VGG_ILSVRC_19_layers.caffemodel
conv1_1: 64 3 3 3
conv1_2: 64 64 3 3
conv2_1: 128 64 3 3
conv2_2: 128 128 3 3
conv3_1: 256 128 3 3
conv3_2: 256 256 3 3
conv3_3: 256 256 3 3
conv3_4: 256 256 3 3
conv4_1: 512 256 3 3
conv4_2: 512 512 3 3
conv4_3: 512 512 3 3
conv4_4: 512 512 3 3
conv5_1: 512 512 3 3
conv5_2: 512 512 3 3
conv5_3: 512 512 3 3
conv5_4: 512 512 3 3
fc6: 1 1 25088 4096
fc7: 1 1 4096 4096
fc8: 1 1 4096 1000
WARNING: Skipping content loss	
Iteration 1 / 1000	
  Content 1 loss: 2091178.593750	
  Style 1 loss: 30021.292114	
  Style 2 loss: 700349.560547	
  Style 3 loss: 153033.203125	
  Style 4 loss: 12404635.156250	
  Style 5 loss: 656.860304	
  Total loss: 15379874.666090	
Iteration 2 / 1000	
  Content 1 loss: 2091177.343750	
  Style 1 loss: 30021.292114	
  Style 2 loss: 700349.560547	
  Style 3 loss: 153033.203125	
  Style 4 loss: 12404633.593750	
  Style 5 loss: 656.860304	
  Total loss: 15379871.853590	
```

## (Optional) Step 4: Install CUDA

If you have a [CUDA-capable GPU from NVIDIA](https://developer.nvidia.com/cuda-gpus) then you can
speed up `neural-style` with CUDA.

First download and unpack the local CUDA installer from NVIDIA; note that there are different
installers for each recent version of Ubuntu:

```
# For Ubuntu 14.10
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1410-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410-7-0-local_7.0-28_amd64.deb
```

```
# For Ubuntu 14.04
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
```

```
# For Ubuntu 12.04
http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1204-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1204-7-0-local_7.0-28_amd64.deb
```

Now update the repository cache and install CUDA. Note that this will also install a graphics driver from NVIDIA.

```
sudo apt-get update
sudo apt-get install cuda
```

At this point you may need to reboot your machine to load the new graphics driver.
After rebooting, you should be able to see the status of your graphics card(s) by running
the command `nvidia-smi`; it should give output that looks something like this:

```
Sun Sep  6 14:02:59 2015       
+------------------------------------------------------+                       
| NVIDIA-SMI 346.96     Driver Version: 346.96         |                       
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:01:00.0      On |                  N/A |
| 22%   49C    P8    18W / 250W |   1091MiB / 12287MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:04:00.0     Off |                  N/A |
| 29%   44C    P8    27W / 189W |     15MiB /  6143MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:05:00.0     Off |                  N/A |
| 30%   45C    P8    33W / 189W |     15MiB /  6143MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1277    G   /usr/bin/X                                     631MiB |
|    0      2290    G   compiz                                         256MiB |
|    0      2489    G   ...s-passed-by-fd --v8-snapshot-passed-by-fd   174MiB |
+-----------------------------------------------------------------------------+
```

## (Optional) Step 5: Install CUDA backend for torch

This is easy:

```
luarocks install cutorch
luarocks install cunn
```

You can check that the installation worked by running the following:

```
th -e "require 'cutorch'; require 'cunn'; print(cutorch)"
```

This should produce output like the this:

```
{
  getStream : function: 0x40d40ce8
  getDeviceCount : function: 0x40d413d8
  setHeapTracking : function: 0x40d41a78
  setRNGState : function: 0x40d41a00
  getBlasHandle : function: 0x40d40ae0
  reserveBlasHandles : function: 0x40d40980
  setDefaultStream : function: 0x40d40f08
  getMemoryUsage : function: 0x40d41480
  getNumStreams : function: 0x40d40c48
  manualSeed : function: 0x40d41960
  synchronize : function: 0x40d40ee0
  reserveStreams : function: 0x40d40bf8
  getDevice : function: 0x40d415b8
  seed : function: 0x40d414d0
  deviceReset : function: 0x40d41608
  streamWaitFor : function: 0x40d40a00
  withDevice : function: 0x40d41630
  initialSeed : function: 0x40d41938
  CudaHostAllocator : torch.Allocator
  test : function: 0x40ce5368
  getState : function: 0x40d41a50
  streamBarrier : function: 0x40d40b58
  setStream : function: 0x40d40c98
  streamBarrierMultiDevice : function: 0x40d41538
  streamWaitForMultiDevice : function: 0x40d40b08
  createCudaHostTensor : function: 0x40d41670
  setBlasHandle : function: 0x40d40a90
  streamSynchronize : function: 0x40d41590
  seedAll : function: 0x40d414f8
  setDevice : function: 0x40d414a8
  getNumBlasHandles : function: 0x40d409d8
  getDeviceProperties : function: 0x40d41430
  getRNGState : function: 0x40d419d8
  manualSeedAll : function: 0x40d419b0
  _state : userdata: 0x022fe750
}
```

You should now be able to run `neural-style` in GPU mode:

```
th neural_style.lua -gpu 0 -print_iter 1
```

### (Optional) Step 6: Install cuDNN

cuDNN is a library from NVIDIA that efficiently implements many of the operations (like convolutions and pooling)
that are commonly used in deep learning.

After registering as a developer with NVIDIA, you can [download cuDNN here](https://developer.nvidia.com/cudnn).

After dowloading, you can unpack and install cuDNN like this:

```
tar -xzvf cudnn-6.5-linux-x64-v2.tgz
cd cudnn-6.5-linux-x64-v2/
sudo cp libcudnn* /usr/local/cuda-7.0/lib64
sudo cp cudnn.h /usr/local/cuda-7.0/include
```

Next we need to install the torch bindings for cuDNN:

```
luarocks install cudnn
```

You should now be able to run `neural-style` with cuDNN like this:

```
th neural_style.lua -gpu 0 -backend cudnn
```

Note that the cuDNN backend can only be used for GPU mode.
