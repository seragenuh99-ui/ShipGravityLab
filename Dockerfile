# 1. 基础镜像：升级到 Ubuntu 22.04 以匹配现代 GLIBC 版本，并支持 4060 显卡
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. 环境设置
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装工具 (Ubuntu 22.04 默认自带 GCC 11/12，非常适合 C++14/17)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    gcc \
    g++ \
    make \
    git \
    python3-pip && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake && \
    rm -rf /var/lib/apt/lists/*

# 4. 设置容器内的工作目录
WORKDIR /app

# 5. 复制源码 (有了 .dockerignore，这里会很干净)
COPY . .

# 6. 编译项目
# 强制指定架构为 89 (RTX 4060)
RUN rm -rf build_linux && mkdir build_linux && cd build_linux && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && \
    make -j$(nproc)

# 7. 启动命令
CMD ["/app/build_linux/ShipGravityLab", "testconfigs/cangduan1-jm.inp"]