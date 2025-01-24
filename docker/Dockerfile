FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS clang-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget \
    && rm -rf /var/lib/apt/lists/*
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-16 main"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        llvm-16 \
        clang-16 \
    && rm -rf /var/lib/apt/lists/*



FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cmake-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget \
    && rm -rf /var/lib/apt/lists/*
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
    && rm -rf /var/lib/apt/lists/*



FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS bazel-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bazel-6.1.0 \
    && rm -rf /var/lib/apt/lists/*



FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS trt-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*
ARG TRT_VERSION=8.6.1.6
ARG VER="${TRT_VERSION}-1+cuda11.8"
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer8=${VER} libnvonnxparsers8=${VER} libnvparsers8=${VER} libnvinfer-plugin8=${VER} \
        libnvinfer-dev=${VER} libnvonnxparsers-dev=${VER} libnvparsers-dev=${VER} libnvinfer-plugin-dev=${VER} \
        python3-libnvinfer=${VER} libnvinfer-dispatch8=${VER} libnvinfer-dispatch-dev=${VER} libnvinfer-lean8=${VER} \
        libnvinfer-lean-dev=${VER} libnvinfer-vc-plugin8=${VER} libnvinfer-vc-plugin-dev=${VER} \
        libnvinfer-headers-dev=${VER} libnvinfer-headers-plugin-dev=${VER} \
    && rm -rf /var/lib/apt/lists/*



FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS tensorflow-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3-dev \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=clang-builder /usr/lib/ /usr/lib/
COPY --from=clang-builder /usr/bin/ /usr/bin/
COPY --from=clang-builder /usr/bin/clang-16 /usr/bin/clang
COPY --from=clang-builder /usr/bin/clang++-16 /usr/bin/clang++
COPY --from=bazel-builder /usr/bin/bazel-6.1.0 /usr/bin/bazel
COPY --from=trt-builder /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
COPY --from=trt-builder /lib/x86_64-linux-gnu/ /lib/x86_64-linux-gnu/
COPY --from=trt-builder /usr/include/x86_64-linux-gnu/ /usr/include/x86_64-linux-gnu/
RUN ldconfig

# Build TensorFlow
WORKDIR /
ARG TENSORFLOW_VERSION=v2.14.0
RUN git clone -b ${TENSORFLOW_VERSION} --depth 1 https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
ENV TF_NEED_CUDA=1
ENV TF_NEED_TENSORRT=1
ENV TF_CUDA_CLANG=1
ENV TF_NEED_ROCM=0
ENV TF_CUDA_COMPUTE_CAPABILITIES="7.0,8.0"
RUN ./configure
RUN bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so

# Copy TensorFlow C++ Library
ARG LIB_DIR=/usr/local/lib/
RUN cd bazel-bin/tensorflow && \
    ln -s libtensorflow_framework.so.2.14.0 libtensorflow_framework.so && \
    ln -s libtensorflow_framework.so.2.14.0 libtensorflow_framework.so.2 && \
    cp libtensorflow_cc.so ${LIB_DIR} && \
    cp libtensorflow_cc.so.2 ${LIB_DIR} && \
    cp libtensorflow_cc.so.2.14.0 ${LIB_DIR} && \
    cp libtensorflow_framework.so ${LIB_DIR} && \
    cp libtensorflow_framework.so.2 ${LIB_DIR} && \
    cp libtensorflow_framework.so.2.14.0 ${LIB_DIR}

# Copy TensorFlow C++ Header Files
ARG HEADER_DIR=/usr/local/include/
RUN mkdir -p ${HEADER_DIR}
RUN find tensorflow/core -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \;
RUN find tensorflow/cc -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \;
RUN find tensorflow/c -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \;
RUN find tensorflow/tsl -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \;
RUN find third_party/eigen3 -follow -type f -exec cp --parents {} ${HEADER_DIR} \;
RUN cd bazel-bin && \
    find tensorflow -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \;
RUN cd bazel-tensorflow/external/com_google_protobuf/src && \
    find google -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR} \; && \
    find google -follow -type f -name "*.inc" -exec cp --parents {} ${HEADER_DIR} \;
RUN cd bazel-tensorflow/external/com_google_absl && \
    find absl -follow -type f -exec cp --parents {} ${HEADER_DIR} \;
RUN cd bazel-tensorflow/external/ml_dtypes && \
    find include -follow -type f -exec cp --parents {} ${HEADER_DIR} \;
RUN cd bazel-tensorflow/external/eigen_archive && \
    find Eigen -follow -type f -exec cp --parents {} ${HEADER_DIR} \; && \
    find unsupported -follow -type f -exec cp --parents {} ${HEADER_DIR} \;


FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        ca-certificates \
        git \
        python3-dev \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.27
COPY --from=clang-builder /usr/lib/ /usr/lib/
COPY --from=clang-builder /usr/bin/ /usr/bin/
COPY --from=clang-builder /usr/bin/clang-16 /usr/bin/clang
COPY --from=clang-builder /usr/bin/clang++-16 /usr/bin/clang++
COPY --from=cmake-builder /usr/bin/cmake /usr/bin/cmake
COPY --from=cmake-builder /usr/share/cmake-${CMAKE_VERSION}/ /usr/share/cmake-${CMAKE_VERSION}/
COPY --from=trt-builder /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
COPY --from=trt-builder /lib/x86_64-linux-gnu/ /lib/x86_64-linux-lib
COPY --from=trt-builder /usr/include/x86_64-linux-gnu/ /usr/include/x86_64-linux-gnu/
COPY --from=tensorflow-builder /usr/local/lib/ /usr/local/lib/
COPY --from=tensorflow-builder /usr/local/include/ /usr/local/include/
RUN ldconfig

COPY FindTensorFlow.cmake /usr/share/cmake-${CMAKE_VERSION}/Modules/

RUN pip install --upgrade pip
RUN pip install tensorflow
RUN pip install --upgrade tensorflow-hub

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++
