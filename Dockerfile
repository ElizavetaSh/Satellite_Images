ARG BASE_IMG=nvidia/cuda
ARG TAG=11.6.2-base-ubuntu20.04
FROM $BASE_IMG:$TAG
USER root
ARG CUDA_SUB_VERSION=cu116
ARG TORCH_V=1.12.1+$CUDA_SUB_VERSION
ARG TORCHVISION_V=0.13.1+$CUDA_SUB_VERSION
RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk \
    libsm6 libxext6 libxrender-dev python3-pip build-essential python3-dev cython3 python3-setuptools \
    python3-wheel python3-numpy python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4 \
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev libwebp-dev libbz2-dev libopenjp2-7-dev \
    libjpeg-turbo8-dev libjxr-dev liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev \
    libzopfli-dev libgif-dev libtiff-dev git
RUN apt-get install openssh-server -y 
RUN /usr/bin/python3 -m pip install --upgrade pip pytest
COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir torch==$TORCH_V torchvision==$TORCHVISION_V torch-optimizer==0.1 $(cat requirements.txt) -f https://download.pytorch.org/whl/$CUDA_SUB_VERSION/torch_stable.html
RUN rm requirements.txt
RUN apt-get install libgdal-dev -y 
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip3 install gdal==3.0.4
WORKDIR /app
COPY . /app