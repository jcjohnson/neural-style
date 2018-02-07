FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y \
	git \
    curl \
    wget \
    sudo \						
    libprotobuf-dev \
    protobuf-compiler

# Install torch
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch; bash install-deps; ./install.sh -b

# Export environment variables manually (https://github.com/Kaixhin/dockerfiles/blob/master/cuda-torch/cuda_v8.0/Dockerfile)
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/torch/install/lib/?.so;/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH

# Install CUDA and cuDNN backend
RUN luarocks install loadcaffe && \
    luarocks install cutorch && \
    luarocks install cunn && \
    luarocks install cudnn 

# Download neural-style
RUN git clone https://github.com/jcjohnson/neural-style.git neural-style

# create volume for the images
RUN mkdir /images
VOLUME /images

# Download models
# To use the NIN network, copy the caffemodel into the build directory and uncomment the line below

# RUN cd neural-style; sh models/download_models.sh
# COPY nin_imagenet_conv.caffemodel /neural-style/models/

# Prepare execution environment
WORKDIR /neural-style/



#RUN ls -al /root/torch/install/lib/lua/5.1/
CMD th neural_style.lua -gpu 0 -print_iter 1

# set neural_style as entrypoint
# ENTRYPOINT [ "th",  "neural_style.lua", "-gpu", "0", "-backend", "cudnn", "-cudnn_autotune" ]
# ENTRYPOINT [ "th",  "neural_style.lua" ]

# docker build -t ahbrosha/neural-style-nvidia . && paplay /usr/share/sounds/gnome/default/alerts/sonar.ogg && notify-send -u urgent "Neural-Style"
# docker run --runtime=nvidia --rm -v $(pwd):/images ahbrosha/neural-style-nvidia th neural_style.lua -gpu 0 -backend cudnn -cudnn_autotune -style_image examples/inputs/picasso_selfport1907.jpg -content_image examples/inputs/brad_pitt.jpg -output_image /images/profile.png -model_file models/nin_imagenet_conv.caffemodel -proto_file models/train_val.prototxt -num_iterations 1000 -seed 123 -content_layers relu0,relu3,relu7,relu12 -style_layers relu0,relu3,relu7,relu12 -content_weight 10 -style_weight 1000 -image_size 512 -optimizer adam
