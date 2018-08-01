FROM ubuntu:18.04

# base layer - python3.7
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-suggests --no-install-recommends -y \
        wget ca-certificates locales git python3.7 python3.7-dev make gcc g++ && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    ln -s ./python3.7 /usr/bin/python3 && \
    apt-get download python3-distutils && \
    dpkg -x python3-distutils* / && \
    rm python3-distutils* && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# nvidia driver layer
ENV NVIDIA_DRIVER_VERSION 384.130
RUN apt-get update && \
    apt-get install -y kmod && \
    mkdir -p /opt/nvidia && cd /opt/nvidia/ && \
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run -O /opt/nvidia/driver.run && \
    chmod +x /opt/nvidia/driver.run && \
    /opt/nvidia/driver.run -a -s --no-nvidia-modprobe --no-kernel-module --no-unified-memory --no-x-check --no-opengl-files && \
    rm -rf /opt/nvidia && \
    apt-get remove -y kmod && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# cuda layer
ENV CUDA_VERSION 9.2.88
ENV CUDA_VERSION_DASH 9-2
ENV CUDA_VERSION_MAJOR 9.2
RUN apt-get update && \
    apt-get install -y gnupg && \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb && \
    rm cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install --no-install-suggests --no-install-recommends \
        cuda-cublas-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-cufft-${CUDA_VERSION_DASH} \
        cuda-curand-${CUDA_VERSION_DASH} \
        cuda-cusolver-${CUDA_VERSION_DASH} \
        cuda-cusparse-${CUDA_VERSION_DASH} && \
    sed -i 's#"$#:/usr/local/cuda-${CUDA_VERSION_MAJOR}/bin"#' /etc/environment && \
    apt-get remove -y gnupg && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda-${CUDA_VERSION_MAJOR}/bin

# cudnn layer
ADD libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb /
RUN dpkg -i /libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb && rm libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb

# pip deps
RUN pip3 install --no-cache-dir numpy

# build and install tensorflow
ENV BAZEL_VERSION 0.15.0
ADD libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb /
ADD 0001-Port-to-Python-3.7.patch /
ADD 0002-Update-Cython.patch /
RUN apt-get update && \
    dpkg -i /libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb && \
    rm libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        unzip \
        cuda-command-line-tools-${CUDA_VERSION_DASH} \
        cuda-cublas-dev-${CUDA_VERSION_DASH} \
        cuda-cudart-dev-${CUDA_VERSION_DASH} \
        cuda-cufft-dev-${CUDA_VERSION_DASH} \
        cuda-curand-dev-${CUDA_VERSION_DASH} \
        cuda-cusolver-dev-${CUDA_VERSION_DASH} \
        cuda-cusparse-dev-${CUDA_VERSION_DASH} && \
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x ./bazel-*.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    rm ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    git clone --single-branch --depth 1 --branch r1.9 https://github.com/tensorflow/tensorflow && \
    cd tensorflow && \
    git apply /0001-Port-to-Python-3.7.patch && \
    git apply /0002-Update-Cython.patch && \
    rm /*.patch && \
    echo 'export PYTHON_BIN_PATH="/usr/bin/python3"' > tools/python_bin_path.sh && \
    echo 'import /tensorflow/.tf_configure.bazelrc' > .bazelrc && \
    echo 'build --action_env PYTHON_BIN_PATH="/usr/bin/python3"\n\
build --action_env PYTHON_LIB_PATH="/usr/lib/python3/dist-packages"\n\
build --python_path="/usr/bin/python3"\n\
build --define with_jemalloc=true\n\
build:gcp --define with_gcp_support=false\n\
build:hdfs --define with_hdfs_support=false\n\
build:s3 --define with_s3_support=false\n\
build:kafka --define with_kafka_support=false\n\
build:xla --define with_xla_support=false\n\
build:gdr --define with_gdr_support=false\n\
build:verbs --define with_verbs_support=false\n\
build --action_env TF_NEED_OPENCL_SYCL="0"\n\
build --action_env TF_NEED_CUDA="1"\n\
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-9.2"\n\
build --action_env TF_CUDA_VERSION="9.2"\n\
build --action_env CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"\n\
build --action_env TF_CUDNN_VERSION="7"\n\
build --action_env TF_NCCL_VERSION="1"\n\
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="3.7,6.1,7.0"\n\
build --action_env TF_CUDA_CLANG="0"\n\
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/gcc"\n\
build --config=cuda\n\
test --config=cuda\n\
build --define grpc_no_ares=true\n\
build:opt --copt=-march=native\n\
build:opt --host_copt=-march=native\n\
build:opt --define with_default_optimizations=true\n\
build --strip=always' > .tf_configure.bazelrc && \
    cat .tf_configure.bazelrc && \
    ln -s python3 /usr/bin/python && \
    bazel build --config opt '@protobuf_archive//:src/google/protobuf/any.h' && \
    find bazel-tensorflow/external/protobuf_archive/python/google/protobuf/pyext -name '*.cc' -exec sed -i "s/PyUnicode_AsUTF8AndSize(/(char*)PyUnicode_AsUTF8AndSize(/g" {} \; && \
    bazel build --config=cuda --config=opt --config=monolithic //tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp && \
    pip --no-cache-dir install /tmp/tensorflow-*.whl && \
    rm /tmp/tensorflow-*.whl && \
    bazel shutdown && \
    cd .. && \
    rm -rf tensorflow && \
    rm -rf /root/.bazel /root/.cache/bazel && \
    apt-get remove -y \
        unzip \
        cuda-command-line-tools-${CUDA_VERSION_DASH} \
        cuda-cublas-dev-${CUDA_VERSION_DASH} \
        cuda-cudart-dev-${CUDA_VERSION_DASH} \
        cuda-cufft-dev-${CUDA_VERSION_DASH} \
        cuda-curand-dev-${CUDA_VERSION_DASH} \
        cuda-cusolver-dev-${CUDA_VERSION_DASH} \
        cuda-cusparse-dev-${CUDA_VERSION_DASH} && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install FractalAI deps
ENV NPY_NUM_BUILD_JOBS 8
RUN apt-get update && \
    apt-get install -y cmake pkg-config flex bison curl libpng16-16 libpng-dev libjpeg-turbo8  \
        libjpeg-turbo8-dev zlib1g-dev libhdf5-100 libhdf5-dev libopenblas-base libopenblas-dev gfortran \
        libfreetype6 libfreetype6-dev && \
    pip3 install --no-cache-dir cython && \
    pip3 install --no-cache-dir \
        git+https://github.com/openai/gym \
        git+https://github.com/ray-project/ray#subdirectory=python \
        git+https://github.com/Guillem-db/atari-py \
        networkx jupyter keras h5py Pillow-simd PyOpenGL matplotlib && \
    python3 -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot" && \
    pip3 uninstall -y cython && \
    apt-get remove -y cmake cmake pkg-config flex bison curl libpng-dev \
        libjpeg-turbo8-dev zlib1g-dev libhdf5-dev libopenblas-dev gfortran \
        libfreetype6-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# FractalAI
ADD . /fractalai
RUN rm /fractalai/*.patch /fractalai/*.deb && \
    pip3 install -e /fractalai && \
    apt-get remove -y gcc g++ make git && \
    apt-get autoremove -y

#Jupyter notebook
RUN mkdir /root/.jupyter && \
    echo 'c.NotebookApp.token = "mallorca"' > /root/.jupyter/jupyter_notebook_config.py
CMD jupyter notebook --allow-root --port 8080 --ip 0.0.0.0
