# docker build -t neural-style .
# mkdir /tmp/neural-style-host
# docker run --rm -i -t -v /tmp/neural-style-host:/host neural-style
# cd ~/neural-style
# th neural_style.lua -gpu -1 -style_image examples/inputs/starry_night.jpg -content_image examples/inputs/hoovertowernight.jpg -num_iterations 500 -image_size 760 -output_image /host/out.png

FROM ubuntu:15.10

# install dependencies.
RUN cd ~ && \
	apt-get update && apt-get upgrade -y && \
	apt-get install -y curl wget sudo libprotobuf-dev protobuf-compiler && \
	curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash && \
	git clone https://github.com/torch/distro.git ~/torch --recursive && \
	cd ~/torch && ./install.sh && . ~/.profile && \
	luarocks install loadcaffe && \
	apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install neural-style from the local directory.
ADD . /root/neural-style/
RUN cd ~/neural-style && \
	sh models/download_models.sh

VOLUME ["/host"]
