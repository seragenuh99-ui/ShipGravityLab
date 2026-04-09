cmake_minimum_required(VERSION 3.18)






# ---------------------------------------------------------
# 1. 项目基础配置 (提前声明语言，CMake 会自动寻找编译器)
# ---------------------------------------------------------
project(ShipGravityLab LANGUAGES CXX CUDA)

# 自动探测 GPU 架构（如果是 4060，保持 89 没问题）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA GPU architectures")
endif()

# ---------------------------------------------------------
# 2. 查找依赖 (Linux 风格：直接用系统路径)
# ---------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# 如果你的代码里用到了 VTK 或 OpenCASCADE，需要取消下面两行的注释：
# find_package(VTK REQUIRED)
# find_package(OpenCASCADE REQUIRED)

# ---------------------------------------------------------
# 3. 生成可执行文件
# ---------------------------------------------------------
add_executable(${PROJECT_NAME} 
    main.cu 
    GpuOperatorOptimized_v2.cu 
    CpuAggregator.cpp 
    MemoryMappedFileReader.h
    OperatorTags.h
)

# ---------------------------------------------------------
# 4. 包含目录与链接库
# ---------------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    # 如果本地还有 fast_float，保留这行
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    # 如果用到了 VTK/OpenCASCADE，记得链接对应的库
)

# ---------------------------------------------------------
# 5. 编译选项优化 (区分 Windows 和 Linux)
# ---------------------------------------------------------
if(MSVC)
    # Windows/Visual Studio 选项
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -O2 -use_fast_math>
    )
else()
    # Linux/GCC 选项 (适合你的 WSL 环境)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++14>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr>
    )
endif()

# 设置源目录宏 (让代码能找到测试数据)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHIPGRAVITYLAB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)