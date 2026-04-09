// =========================================================
// ShipGravityLab - 船舶重心计算系统
// 功能：从 .inp 网格文件读取船舶板/梁结构 → 直接生成GPU数据 → GPU并行计算 → CPU汇总重心
// 优化点：内存映射零拷贝 + 无push_back预分配 + 单遍解析 + 解析即生成GPU数据
// =========================================================

// 1. [必须最先包含] C++ 标准库，解决 std::string, std::vector 等未定义报错
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdio>  // 用于 Linux 下的文件检查

// 2. 项目自定义头文件
#include "MemoryMappedFileReader.h"
#include "OperatorTags.h"

// =========================================================
// GPU 外部函数声明（由CUDA文件实现，这里做声明即可调用）
// =========================================================

/**
 * @brief 初始化GPU内存池
 */
extern "C" void initGpuMemoryPool(size_t maxPlateCount, size_t maxBeamCount);

/**
 * @brief 调度GPU执行板单元属性计算
 */
extern "C" void scheduleGpuPlateOps(ShipPlateGpuData* plates, int plateCount);

/**
 * @brief 调度GPU执行梁单元属性计算
 */
extern "C" void scheduleGpuBeamOps(ShipBeamGpuData* beams, int beamCount);

// =========================================================
// CPU 汇总函数声明
// =========================================================

/**
 * @brief CPU汇总所有单元的重量与形心
 */
void calcTotalGravityCenter(const ShipPlateGpuData* plates, int plateCount,
                            const ShipBeamGpuData* beams, int beamCount,
                            float& totalWeight, float totalCentroid[3]);

// =========================================================
// 主函数：程序入口
// =========================================================
int main(int argc, char* argv[]) {
    // =========================================================
    // 1. 设置输入文件路径
    // =========================================================
    // 修改点：默认路径改为 Linux 相对路径，适配在 build 目录下运行的情况
    std::string inpPath = "testconfigs/cangduan1-jm.inp"; 
    
    // 如果用户在命令行传入了文件路径（例如 ./ShipGravityLab /path/to/file.inp），则覆盖
    if (argc > 1) {
        inpPath = argv[1];
    }
    
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Using inpPath: " << inpPath << std::endl;

    // 增加一步文件存在性检查，防止 Windows 路径在 Linux 下直接崩溃
    FILE* testF = fopen(inpPath.c_str(), "r");
    if (!testF) {
        std::cerr << "!!! Error: Cannot open file: " << inpPath << std::endl;
        std::cerr << "Suggestion: Use './ShipGravityLab ../testconfigs/cangduan1-jm.inp'" << std::endl;
        return 1;
    }
    fclose(testF);
    
    // =========================================================
    // 2. 开始计时：文件读取 + 解析总耗时
    // =========================================================
    auto startRead = std::chrono::high_resolution_clock::now();
    
    // =========================================================
    // 3. 定义数据容器
    // =========================================================
    MemoryMappedFileReader mmapReader;
    std::vector<InpNode> nodes;
    std::vector<PlateEntity> plateEntities;
    std::vector<BeamEntity> beamEntities;
    
    // =========================================================
    // 4. 打开 .inp 文件（内存映射方式）
    // =========================================================
    if (!mmapReader.open(inpPath)) {
        std::cerr << "Failed to open file (mmap error): " << inpPath << std::endl;
        return 1;
    }
    
    // =========================================================
    // 5. 解析文件
    // =========================================================
    auto parseStart = std::chrono::high_resolution_clock::now();
    bool parseSuccess = mmapReader.parse(nodes, plateEntities, beamEntities);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    
    if (!parseSuccess) {
        std::cerr << "Failed to parse file content" << std::endl;
        return 1;
    }
    
    // 关闭内存映射（数据已提取到内部 GPU 容器）
    mmapReader.close();
    
    // =========================================================
    // 6. 打印解析耗时
    // =========================================================
    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - startRead;
    std::chrono::duration<double> parseTime = parseEnd - parseStart;
    
    std::cout << "[Memory Mapped File] Total: " << readTime.count() * 1000 
              << " ms | Parse: " << parseTime.count() * 1000 << " ms" << std::endl;
    std::cout << "Loaded elements. Plate: " << plateEntities.size() 
              << ", Beam: " << beamEntities.size() << std::endl;

    // =========================================================
    // 7. 直接使用解析好的GPU数据
    // =========================================================
    auto& plates = mmapReader.gpuPlates;
    auto& beams  = mmapReader.gpuBeams;

    std::cout << "Prepared data. Plate: " << plates.size() << ", Beam: " << beams.size() << std::endl;

    // =========================================================
    // 8. 打印算子分类信息
    // =========================================================
    std::cout << "\n===== Operator classification =====" << std::endl;
    std::cout << "Plate property ops (GPU): " << plates.size() << std::endl;
    std::cout << "Beam property ops (GPU): " << beams.size() << std::endl;
    std::cout << "Aggregate op (CPU): 1" << std::endl;
    
    // =========================================================
    // 9. 初始化GPU内存池
    // =========================================================
    if (plates.empty() && beams.empty()) {
        std::cout << "No elements to calculate. Finished." << std::endl;
        return 0;
    }
    initGpuMemoryPool(plates.size() * 2, beams.size() * 2);
    
    // =========================================================
    // 10. 开始GPU计算计时
    // =========================================================
    auto startGpu = std::chrono::high_resolution_clock::now();
    
    if (!plates.empty()) {
        scheduleGpuPlateOps(plates.data(), static_cast<int>(plates.size()));
    }
    
    if (!beams.empty()) {
        scheduleGpuBeamOps(beams.data(), static_cast<int>(beams.size()));
    }
    
    auto endGpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuTime = endGpu - startGpu;
    
    std::cout << "\n===== GPU time =====" << std::endl;
    std::cout << "Elapsed: " << gpuTime.count() * 1000 << " ms" << std::endl;
    
    // =========================================================
    // 11. CPU 汇总总重量与总重心
    // =========================================================
    auto startAgg = std::chrono::high_resolution_clock::now();
    
    float totalWeight = 0.0f;
    float totalCentroid[3] = {0.0f, 0.0f, 0.0f};
    
    calcTotalGravityCenter(plates.data(), static_cast<int>(plates.size()),
                          beams.data(), static_cast<int>(beams.size()),
                          totalWeight, totalCentroid);
    
    auto endAgg = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> aggTime = endAgg - startAgg;
    
    // =========================================================
    // 12. 输出最终结果
    // =========================================================
    std::cout << "\n===== Ship section aggregate =====" << std::endl;
    std::cout << "Total weight: " << totalWeight << " kg" << std::endl;
    std::cout << "Centroid: (" << totalCentroid[0] << ", " 
              << totalCentroid[1] << ", " << totalCentroid[2] << ") m" << std::endl;
    
    std::cout << "\n===== CPU aggregate time =====" << std::endl;
    std::cout << "Elapsed: " << aggTime.count() * 1000 << " ms" << std::endl;
    
    // =========================================================
    // 13. 输出程序总耗时
    // =========================================================
    std::chrono::duration<double> totalTime = endAgg - startRead;
    std::cout << "\n===== Total execution time =====" << std::endl;
    std::cout << "Total: " << totalTime.count() * 1000 << " ms" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    return 0;
}