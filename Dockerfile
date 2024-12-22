FROM nvcr.io/nvidia/tensorrt:24.12-py3

WORKDIR /code

RUN apt-get update -y
RUN apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install --upgrade cmake

RUN git clone --single-branch --branch rel-1.20.1 --recursive https://github.com/Microsoft/onnxruntime onnxruntime
RUN /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh
RUN /bin/sh onnxruntime/dockerfiles/scripts/checkout_submodules.sh ${TRT_VERSION:0:3}
RUN cd onnxruntime && \
    /bin/sh build.sh --allow_running_as_root --build_shared_lib \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu \
    --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu \
    --config Release --skip_tests --skip_submodule_sync && \
    cd -
RUN cd onnxruntime/build/Linux/Release && \
    make install && \
    cd -

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip
RUN unzip opencv.zip && rm opencv.zip
RUN mkdir -p opencv-4.10.0/build && \
    cd opencv-4.10.0/build && \
    cmake -DCMAKE_BUILD_TYPE=RELRASE -DOPENCV_GENERATE_PKGCONFIG=ON .. && \
    cmake --build . && \
    cd -
RUN cd opencv-4.10.0/build && \
    make install && \
    cd -

RUN wget https://download.pytorch.org/libtorch/nightly/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0.dev20241208%2Bcu126.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.6.0.dev20241208+cu126.zip && rm libtorch-cxx11-abi-shared-with-deps-2.6.0.dev20241208+cu126.zip
