// =========================================================
// ShipGravityLab - GPU优化计算模块 (v2)
// 功能：在GPU上并行计算船舶结构的几何属性和物理属性
// 优化特性：
// 1. 内存池管理：避免重复的GPU内存分配/释放，减少分配时间（从69ms降至0.5ms）
// 2. 批处理传输：优化主机到设备的数据传输
// 3. 性能监控：详细记录各阶段耗时，便于性能分析
// 4. 独立核函数名：避免与旧版本冲突
// 性能数据（CUDA 13.2）：
// =========================================================

#include "OperatorTags.h"
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdio> // 引入 printf 支持

/**
 * @brief CUDA核函数：计算单块板的几何和物理属性（优化版）
 * @param plates 板数据数组（输入输出参数）
 * @param plateCount 板数量
 * @note 每个CUDA线程处理一块板，实现完全并行化
 * 计算流程：面积 → 形心 → 体积 → 重量
 * 支持两种板类型：三角形板（3节点）和四边形板（4节点）
 */
// 极致优化的核函数版本，最大限度减少寄存器使用，使用restrict关键字减少指针别名分析
__global__ void __launch_bounds__(256, 16) calcPlatePropsKernel_ultra_tight(ShipPlateGpuData* __restrict__ plates, int plateCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plateCount) return;

    ShipPlateGpuData* plate = &plates[idx];
    float area, cx, cy, cz;  // 只留最终结果，不提前初始化

    if (plate->nodeCount == 3) {
        // ===== 三角形面积直接 inline，不存 cross 中间量 =====
        const float* a = plate->p1;
        const float* b = plate->p2;
        const float* c = plate->p3;

        float crx = (b[1]-a[1])*(c[2]-a[2]) - (b[2]-a[2])*(c[1]-a[1]);
        float cry = (b[2]-a[2])*(c[0]-a[0]) - (b[0]-a[0])*(c[2]-a[2]);
        float crz = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
        area = 0.5f * sqrtf(crx*crx + cry*cry + crz*crz);

        // 形心直接乘 1/3，不存 inv3 寄存器
        cx = (a[0] + b[0] + c[0]) * 0.3333333f;
        cy = (a[1] + b[1] + c[1]) * 0.3333333f;
        cz = (a[2] + b[2] + c[2]) * 0.3333333f;
    }
    else if (plate->nodeCount == 4) {
        const float* p1 = plate->p1;
        const float* p2 = plate->p2;
        const float* p3 = plate->p3;
        const float* p4 = plate->p4;

        // 直接算两个三角面积，不调用函数（省寄存器）
        // tri1: p1,p2,p3
        float cr1x = (p2[1]-p1[1])*(p3[2]-p1[2]) - (p2[2]-p1[2])*(p3[1]-p1[1]);
        float cr1y = (p2[2]-p1[2])*(p3[0]-p1[0]) - (p2[0]-p1[0])*(p3[2]-p1[2]);
        float cr1z = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]);
        float a1  = 0.5f * sqrtf(cr1x*cr1x + cr1y*cr1y + cr1z*cr1z);

        // tri2: p1,p3,p4
        float cr2x = (p3[1]-p1[1])*(p4[2]-p1[2]) - (p3[2]-p1[2])*(p4[1]-p1[1]);
        float cr2y = (p3[2]-p1[2])*(p4[0]-p1[0]) - (p3[0]-p1[0])*(p4[2]-p1[2]);
        float cr2z = (p3[0]-p1[0])*(p4[1]-p1[1]) - (p3[1]-p1[1])*(p4[0]-p1[0]);
        float a2  = 0.5f * sqrtf(cr2x*cr2x + cr2y*cr2y + cr2z*cr2z);

        area = a1 + a2;

        if (area > 1e-12f) {
            // 不存 w1/w2/inv3，全部当场计算
            float inv = 1.0f / area;
            cx = ( ((p1[0]+p2[0]+p3[0])*a1) + ((p1[0]+p3[0]+p4[0])*a2) ) * inv * 0.3333333f;
            cy = ( ((p1[1]+p2[1]+p3[1])*a1) + ((p1[1]+p3[1]+p4[1])*a2) ) * inv * 0.3333333f;
            cz = ( ((p1[2]+p2[2]+p3[2])*a1) + ((p1[2]+p3[2]+p4[2])*a2) ) * inv * 0.3333333f;
        } else {
            // 退化直接乘 0.25，不存变量
            cx = (p1[0]+p2[0]+p3[0]+p4[0]) * 0.25f;
            cy = (p1[1]+p2[1]+p3[1]+p4[1]) * 0.25f;
            cz = (p1[2]+p2[2]+p3[2]+p4[2]) * 0.25f;
        }
    } else {
        return;
    }

    // 写回
    plate->area = area;
    plate->centroid[0] = cx;
    plate->centroid[1] = cy;
    plate->centroid[2] = cz;

    float vol = area * plate->thickness;
    plate->volume = vol;
    plate->weight = vol * plate->density;
}

/**
 * @brief CUDA核函数：计算单根梁的几何和物理属性
 * @param beams 梁数据数组（输入输出参数）
 * @param beamCount 梁数量
 * @note 每个CUDA线程处理一根梁，实现完全并行化
 */
__global__ void calcBeamPropsKernel(ShipBeamGpuData* __restrict__ beams, int beamCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= beamCount) return;

    ShipBeamGpuData* beam = &beams[idx];
    
    // 计算长度
    float dx = beam->p2[0] - beam->p1[0];
    float dy = beam->p2[1] - beam->p1[1];
    float dz = beam->p2[2] - beam->p1[2];
    float length = sqrtf(dx*dx + dy*dy + dz*dz);
    
    // 计算形心（中点）
    beam->length = length;
    beam->centroid[0] = (beam->p1[0] + beam->p2[0]) * 0.5f;
    beam->centroid[1] = (beam->p1[1] + beam->p2[1]) * 0.5f;
    beam->centroid[2] = (beam->p1[2] + beam->p2[2]) * 0.5f;
    
    // 计算体积和重量
    float volume = length * beam->sectionArea;
    beam->volume = volume;
    beam->weight = volume * beam->density;
}

/**
 * @brief GPU内存池管理结构体
 * @details 预分配固定大小的GPU内存，在程序运行期间重复使用
 * 避免每次计算都分配释放内存，显著减少内存管理开销
 */
struct GpuMemoryPool {
    // GPU内存指针
    ShipPlateGpuData* d_plates = nullptr; // 板数据GPU内存
    ShipBeamGpuData* d_beams = nullptr;   // 梁数据GPU内存
    
    // 内存池容量
    size_t maxPlates = 0; // 最大可容纳的板数量
    size_t maxBeams = 0;  // 最大可容纳的梁数量
    bool initialized = false; // 内存池是否已初始化
    
    // CUDA事件用于精确计时
    cudaEvent_t startEvent; // 开始计时事件
    cudaEvent_t stopEvent;  // 结束计时事件
    
    /**
     * @brief 构造函数：创建CUDA计时事件
     */
    GpuMemoryPool() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }
    
    /**
     * @brief 析构函数：清理所有GPU资源
     */
    ~GpuMemoryPool() {
        cleanup(); // 释放GPU内存
        cudaEventDestroy(startEvent); // 销毁计时事件
        cudaEventDestroy(stopEvent);
    }
    
    /**
     * @brief 初始化内存池，预分配GPU内存
     * @param maxPlateCount 最大板数量容量
     * @param maxBeamCount 最大梁数量容量
     * @return 初始化成功返回true，失败返回false
     * @note 如果内存池已初始化，会先清理再重新分配
     */
    bool initialize(size_t maxPlateCount, size_t maxBeamCount) {
        if (initialized) cleanup(); // 如果已初始化，先清理
        
        cudaError_t err;
        
        // 分配板数据GPU内存
        err = cudaMalloc(&d_plates, maxPlateCount * sizeof(ShipPlateGpuData));
        if (err != cudaSuccess) {
            std::cerr << "[MemoryPool] Failed to allocate plate GPU memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // 分配梁数据GPU内存
        err = cudaMalloc(&d_beams, maxBeamCount * sizeof(ShipBeamGpuData));
        if (err != cudaSuccess) {
            std::cerr << "[MemoryPool] Failed to allocate beam GPU memory: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_plates); // 清理已分配的内存
            return false;
        }
        
        // 更新内存池状态
        maxPlates = maxPlateCount;
        maxBeams = maxBeamCount;
        initialized = true;
        
        std::cout << "[MemoryPool] Initialized with capacity: " 
                  << maxPlateCount << " plates, " << maxBeamCount << " beams" << std::endl;
        return true;
    }
    
    /**
     * @brief 清理内存池，释放所有GPU内存
     * @note 将内存池恢复到未初始化状态
     */
    void cleanup() {
        if (d_plates) cudaFree(d_plates);
        if (d_beams) cudaFree(d_beams);
        d_plates = nullptr;
        d_beams = nullptr;
        maxPlates = 0;
        maxBeams = 0;
        initialized = false;
    }
    
    /**
     * @brief 开始计时
     * @note 记录当前CUDA流的事件时间戳
     */
    void startTimer() {
        cudaEventRecord(startEvent);
    }
    
    /**
     * @brief 停止计时并返回经过的时间
     * @return 从startTimer到stopTimer的经过时间（毫秒）
     * @note 使用CUDA事件计时，比CPU计时更精确，特别是对于GPU操作
     */
    float stopTimer() {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent); // 等待事件完成
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
        return milliseconds;
    }
};

// 全局内存池实例（单例模式）
// 在整个程序运行期间共享，避免重复初始化
static GpuMemoryPool g_memoryPool;

// =========================================================
// 优化版GPU算子调度函数（外部C接口）
// =========================================================

/**
 * @brief 初始化GPU内存池（外部调用接口）
 * @param maxPlateCount 最大板数量容量
 * @param maxBeamCount 最大梁数量容量
 * @note 如果初始化失败，程序会回退到动态内存分配，不影响功能
 */
extern "C" void initGpuMemoryPool(size_t maxPlateCount, size_t maxBeamCount) {
    if (!g_memoryPool.initialize(maxPlateCount, maxBeamCount)) {
        std::cerr << "Warning: GPU memory pool initialization failed, falling back to dynamic allocation" << std::endl;
    }
}

/**
 * @brief 清理GPU内存池（外部调用接口）
 * @note 在程序结束前调用，确保GPU资源正确释放
 */
extern "C" void cleanupGpuMemoryPool() {
    g_memoryPool.cleanup();
}

/**
 * @brief 调度GPU计算板属性（优化版）
 * @param plates 板数据数组（输入输出：主机内存）
 * @param plateCount 板数量
 * @note 完整的GPU计算流程：
 * 1. 分配GPU内存（使用内存池或动态分配）
 * 2. 拷贝数据到GPU（主机→设备）
 * 3. 启动核函数并行计算
 * 4. 拷贝结果回CPU（设备→主机）
 * 5. 释放GPU内存（如果不是内存池）
 * 详细计时各阶段性能，用于性能分析和优化验证
 */
extern "C" void scheduleGpuPlateOpsOptimized(ShipPlateGpuData* plates, int plateCount)
{
    // 定义细分阶段计时器
    cudaEvent_t ev_start, ev_h2d_done, ev_kernel_done, ev_d2h_done;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_h2d_done);
    cudaEventCreate(&ev_kernel_done); cudaEventCreate(&ev_d2h_done);

    ShipPlateGpuData* d_plates = nullptr;
    float mallocTime = 0;

    // ==============================
    // 1. 获取显存（内存池）
    // ==============================
    g_memoryPool.startTimer();
    if (g_memoryPool.initialized && plateCount <= g_memoryPool.maxPlates) {
        d_plates = g_memoryPool.d_plates;
        mallocTime = 0;
    } else {
        cudaMalloc(&d_plates, plateCount * sizeof(ShipPlateGpuData));
        mallocTime = g_memoryPool.stopTimer();
    }

    // ==============================
    // 【优化核心】创建 2 个 CUDA 流
    // ==============================
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 把数据切成两半
    int N = plateCount;
    int half = N / 2;

    // ==============================
    // 【异步拷贝 + 异步计算】
    // ==============================
    cudaEventRecord(ev_start); // 全程开始

    // 前半数据 → 流1 异步拷贝
    cudaMemcpyAsync(d_plates, 
                    plates, 
                    half * sizeof(ShipPlateGpuData), 
                    cudaMemcpyHostToDevice, stream1);

    // 后半数据 → 流2 异步拷贝
    cudaMemcpyAsync(d_plates + half, 
                    plates + half, 
                    (N - half) * sizeof(ShipPlateGpuData), 
                    cudaMemcpyHostToDevice, stream2);

    cudaEventRecord(ev_h2d_done, stream2); // 数据上传结束点

    // 启动核函数（流1）
    int blockSize = 256;
    dim3 grid1((half + blockSize - 1) / blockSize);
    calcPlatePropsKernel_ultra_tight<<<grid1, blockSize, 0, stream1>>>(d_plates, half);

    // 启动核函数（流2）
    dim3 grid2(((N-half) + blockSize - 1) / blockSize);
    calcPlatePropsKernel_ultra_tight<<<grid2, blockSize, 0, stream2>>>(d_plates + half, N-half);

    cudaEventRecord(ev_kernel_done, stream2); // 计算结束点

    // 异步回传
    cudaMemcpyAsync(plates, 
                    d_plates, 
                    half * sizeof(ShipPlateGpuData), 
                    cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(plates + half, 
                    d_plates + half, 
                    (N-half) * sizeof(ShipPlateGpuData), 
                    cudaMemcpyDeviceToHost, stream2);

    cudaEventRecord(ev_d2h_done, stream2); // 回传结束点

    // 等待所有流完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 计算各阶段具体耗时
    float h2d_ms, kernel_ms, d2h_ms, total_ms;
    cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d_done);
    cudaEventElapsedTime(&kernel_ms, ev_h2d_done, ev_kernel_done);
    cudaEventElapsedTime(&d2h_ms, ev_kernel_done, ev_d2h_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_d2h_done);

    // 【修改点】详细分阶段输出
    printf("\n[Plate GPU Async] Detail Timing ----------------\n");
    printf("  Pool/Malloc: %.4f ms\n", mallocTime);
    printf("  H2D Transfer: %.4f ms\n", h2d_ms);
    printf("  Kernel Calc:  %.4f ms\n", kernel_ms);
    printf("  D2H Transfer: %.4f ms\n", d2h_ms);
    printf("  Total GPU:    %.4f ms\n", total_ms);
    printf("------------------------------------------------\n");

    // ==============================
    // 销毁流与事件
    // ==============================
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_h2d_done);
    cudaEventDestroy(ev_kernel_done); cudaEventDestroy(ev_d2h_done);

    // ==============================
    // 释放（如果不是内存池）
    // ==============================
    if (!g_memoryPool.initialized || plateCount > g_memoryPool.maxPlates) {
        cudaFree(d_plates);
    }
}
/**
 * @brief 调度GPU计算梁属性（优化版）
 * @param beams 梁数据数组（输入输出：主机内存）
 * @param beamCount 梁数量
 * @note 与板计算类似的流程，但处理梁数据
 * 性能特点：梁计算比板计算更简单，核函数执行时间更短
 */
extern "C" void scheduleGpuBeamOpsOptimized(ShipBeamGpuData* beams, int beamCount) {
    // 阶段计时事件
    cudaEvent_t ev_start, ev_h2d, ev_ker, ev_d2h;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_ker); cudaEventCreate(&ev_d2h);
    
    float mallocTime = 0;
    ShipBeamGpuData* d_beams = nullptr; // GPU内存指针
    
    // ========== 阶段1：分配GPU内存 ==========
    g_memoryPool.startTimer();
    if (g_memoryPool.initialized && beamCount <= g_memoryPool.maxBeams) {
        d_beams = g_memoryPool.d_beams;
        mallocTime = 0;
    } else {
        cudaMalloc(&d_beams, beamCount * sizeof(ShipBeamGpuData));
        mallocTime = g_memoryPool.stopTimer();
    }
    
    cudaEventRecord(ev_start);

    // ========== 阶段2：拷贝数据到GPU（主机→设备） ==========
    cudaMemcpy(d_beams, beams, beamCount * sizeof(ShipBeamGpuData), cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d);
    
    // ========== 阶段3：启动核函数并行计算 ==========
    int blockSize = 256; 
    int gridSize = (beamCount + blockSize - 1) / blockSize;
    calcBeamPropsKernel<<<gridSize, blockSize>>>(d_beams, beamCount);
    cudaEventRecord(ev_ker);
    
    // ========== 阶段4：拷贝结果回CPU（设备→主机） ==========
    cudaMemcpy(beams, d_beams, beamCount * sizeof(ShipBeamGpuData), cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h);
    
    cudaDeviceSynchronize();

    // 计算分段时长
    float h2d_ms, ker_ms, d2h_ms, total_ms;
    cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d);
    cudaEventElapsedTime(&ker_ms, ev_h2d, ev_ker);
    cudaEventElapsedTime(&d2h_ms, ev_ker, ev_d2h);
    cudaEventElapsedTime(&total_ms, ev_start, ev_d2h);

    // 【修改点】梁计算详细输出
    printf("\n[Beam GPU Sync] Detail Timing -----------------\n");
    printf("  Pool/Malloc:  %.4f ms\n", mallocTime);
    printf("  H2D Transfer: %.4f ms\n", h2d_ms);
    printf("  Kernel Calc:  %.4f ms\n", ker_ms);
    printf("  D2H Transfer: %.4f ms\n", d2h_ms);
    printf("  Total GPU:    %.4f ms\n", total_ms);
    printf("------------------------------------------------\n");
    
    // 销毁事件
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_h2d);
    cudaEventDestroy(ev_ker); cudaEventDestroy(ev_d2h);

    if (!g_memoryPool.initialized || beamCount > g_memoryPool.maxBeams) {
        cudaFree(d_beams);
    }
}

/**
 * @brief 批处理版本：一次性处理所有板和梁
 * @param plates 板数据数组
 * @param plateCount 板数量
 * @param beams 梁数据数组
 * @param beamCount 梁数量
 * @note 便利函数，依次调用板和梁的调度函数
 * 共享同一个内存池，减少总体内存分配开销
 */
extern "C" void scheduleAllGpuOpsOptimized(ShipPlateGpuData* plates, int plateCount, 
                                          ShipBeamGpuData* beams, int beamCount) {
    std::cout << "[Batch GPU] Processing " << plateCount << " plates and " << beamCount << " beams" << std::endl;
    
    // 分别调度板和梁计算，但共享同一个内存池
    if (plateCount > 0) {
        scheduleGpuPlateOpsOptimized(plates, plateCount);
    }
    
    if (beamCount > 0) {
        scheduleGpuBeamOpsOptimized(beams, beamCount);
    }
}

// =========================================================
// 兼容性包装器函数（用于向后兼容）
// =========================================================

/**
 * @brief 调度GPU计算板属性（兼容性包装器）
 * @param plates 板数据数组
 * @param plateCount 板数量
 * @note 这是原始版本的函数名，用于向后兼容
 * 内部调用优化版本 scheduleGpuPlateOpsOptimized
 */
extern "C" void scheduleGpuPlateOps(ShipPlateGpuData* plates, int plateCount) {
    scheduleGpuPlateOpsOptimized(plates, plateCount);
}

/**
 * @brief 调度GPU计算梁属性（兼容性包装器）
 * @param beams 梁数据数组
 * @param beamCount 梁数量
 * @note 这是原始版本的函数名，用于向后兼容
 * 内部调用优化版本 scheduleGpuBeamOpsOptimized
 */
extern "C" void scheduleGpuBeamOps(ShipBeamGpuData* beams, int beamCount) {
    scheduleGpuBeamOpsOptimized(beams, beamCount);
}

/**
 * @brief 性能对比测试函数
 * @param plates 板数据数组
 * @param plateCount 板数量
 * @param beams 梁数据数组
 * @param beamCount 梁数量
 */
extern "C" void benchmarkGpuOps(ShipPlateGpuData* plates, int plateCount,
                                 ShipBeamGpuData* beams, int beamCount) {
    std::cout << "\n=== GPU Performance Benchmark Test ===" << std::endl;
    std::cout << "Data Volume: " << plateCount << " plates, " << beamCount << " beams" << std::endl;

    std::vector<ShipPlateGpuData> platesCopy(plates, plates + plateCount);
    std::vector<ShipBeamGpuData> beamsCopy(beams, beams + beamCount);

    std::cout << "\n[1] Run 1:" << std::endl;
    if (plateCount > 0) scheduleGpuPlateOpsOptimized(platesCopy.data(), plateCount);
    if (beamCount > 0) scheduleGpuBeamOpsOptimized(beamsCopy.data(), beamCount);

    platesCopy.assign(plates, plates + plateCount);
    beamsCopy.assign(beams, beams + beamCount);

    std::cout << "\n[2] Run 2:" << std::endl;
    if (plateCount > 0) scheduleGpuPlateOpsOptimized(platesCopy.data(), plateCount);
    if (beamCount > 0) scheduleGpuBeamOpsOptimized(beamsCopy.data(), beamCount);

    std::cout << "\n=== Test Completed ===" << std::endl;
}