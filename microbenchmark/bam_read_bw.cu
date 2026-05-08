// BaM baseline: Pure I/O random read bandwidth benchmark
// Reproduces CoPilotIO paper Figure 7: random read, 4KB, varying SM count.
//
// Usage: nvm-bam-read-bw [options]
//   --ctrl       /dev/libnvmX      (default: /dev/libnvm0)
//   --sms        N                  (default: 108, number of SMs to use)
//   --warps      N                  (default: 32, warps per SM)
//   --qps        N                  (default: 128, number of NVMe queue pairs)
//   --qd         N                  (default: 1024, queue depth)
//   --io-size    N                  (default: 4096, I/O size in bytes)
//   --gpu        N                  (default: 0)

#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_queue.h>

#include <ctrl.h>
#include <queue.h>
#include <page_cache.h>
#include <util.h>

using error = std::runtime_error;

__global__ void bam_random_read_kernel(
    Controller** ctrls,
    page_cache_d_t* pc,
    uint64_t n_blocks_per_io,
    uint64_t ssd_capacity_blocks,
    uint64_t num_ios_per_thread,
    uint64_t total_threads)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    const uint32_t n_qps = ctrls[0]->n_qps;
    const uint32_t queue = tid % n_qps;
    uint64_t seed = tid * 6364136223846793005ULL + 1442695040888963407ULL;

    for (uint64_t i = 0; i < num_ios_per_thread; i++) {
        const uint64_t pc_idx = (tid + i * total_threads) % pc->n_pages;

        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;

        uint64_t start_lba = (seed % (ssd_capacity_blocks - n_blocks_per_io));
        start_lba = (start_lba / n_blocks_per_io) * n_blocks_per_io;

        read_data(pc, ctrls[0]->d_qps + queue, start_lba, n_blocks_per_io, pc_idx);
    }
}

struct Config {
    std::string ctrl_path = "/dev/libnvm0";
    int gpu = 0;
    uint32_t sms = 108;
    uint32_t warps_per_sm = 32;
    uint32_t qps = 128;
    uint64_t qd = 1024;
    uint64_t io_size = 4096;
    uint64_t bench_ios = 50;
};

static Config parse_args(int argc, char** argv)
{
    Config c;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctrl") && i+1 < argc) c.ctrl_path = argv[++i];
        else if (!strcmp(argv[i], "--sms") && i+1 < argc) c.sms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps") && i+1 < argc) c.warps_per_sm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qps") && i+1 < argc) c.qps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qd") && i+1 < argc) c.qd = strtoull(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "--io-size") && i+1 < argc) c.io_size = strtoull(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "--gpu") && i+1 < argc) c.gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench-ios") && i+1 < argc) c.bench_ios = strtoull(argv[++i], nullptr, 0);
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
    }
    return c;
}

int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    cuda_err_chk(cudaSetDevice(cfg.gpu));

    cudaDeviceProp props{};
    cuda_err_chk(cudaGetDeviceProperties(&props, cfg.gpu));
    const uint32_t hw_sms = props.multiProcessorCount;
    if (cfg.sms > hw_sms) {
        fprintf(stderr, "Warning: requested %u SMs but GPU has %u, clamping\n", cfg.sms, hw_sms);
        cfg.sms = hw_sms;
    }

    std::vector<Controller*> ctrls(1);
    ctrls[0] = new Controller(cfg.ctrl_path.c_str(), 1, cfg.gpu, cfg.qd, cfg.qps);

    const uint64_t blk_size = ctrls[0]->blk_size;
    const uint64_t n_blocks_per_io = cfg.io_size / blk_size;
    const uint64_t ssd_capacity_blocks = ctrls[0]->ns.size;

    const uint32_t threads_per_block = cfg.warps_per_sm * 32;
    const uint32_t grid_dim = cfg.sms;
    const uint64_t total_threads = (uint64_t)grid_dim * threads_per_block;

    const uint64_t n_pc_pages = 1024ULL * 1024;
    page_cache_t pc(cfg.io_size, n_pc_pages, cfg.gpu, *ctrls[0], 64, ctrls);
    page_cache_d_t* d_pc = pc.d_pc_ptr;

    printf("BaM Random Read Benchmark\n");
    printf("  GPU: %s (%u SMs, using %u)\n", props.name, hw_sms, cfg.sms);
    printf("  SSD: %s, blk_size=%lu, capacity=%.1f GiB\n",
           cfg.ctrl_path.c_str(), blk_size,
           (double)(ssd_capacity_blocks * blk_size) / (1024.0*1024*1024));
    printf("  IO size: %lu B, QPs: %u, QD: %lu\n", cfg.io_size, cfg.qps, cfg.qd);
    printf("  Grid: %u blocks x %u threads (%u warps/SM), total %llu threads\n",
           grid_dim, threads_per_block, cfg.warps_per_sm,
           (unsigned long long)total_threads);

    // Warmup
    uint64_t warmup_ios = 10;
    bam_random_read_kernel<<<grid_dim, threads_per_block>>>(
        pc.pdt.d_ctrls, d_pc, n_blocks_per_io, ssd_capacity_blocks,
        warmup_ios, total_threads);
    cuda_err_chk(cudaDeviceSynchronize());

    // Benchmark run
    uint64_t bench_ios = cfg.bench_ios;
    cudaEvent_t t0, t1;
    cuda_err_chk(cudaEventCreate(&t0));
    cuda_err_chk(cudaEventCreate(&t1));

    cuda_err_chk(cudaEventRecord(t0));
    bam_random_read_kernel<<<grid_dim, threads_per_block>>>(
        pc.pdt.d_ctrls, d_pc, n_blocks_per_io, ssd_capacity_blocks,
        bench_ios, total_threads);
    cuda_err_chk(cudaEventRecord(t1));
    cuda_err_chk(cudaEventSynchronize(t1));

    float bench_ms = 0;
    cuda_err_chk(cudaEventElapsedTime(&bench_ms, t0, t1));

    double total_bytes = (double)(total_threads * bench_ios) * cfg.io_size;
    double bw_gibs = (total_bytes / (1024.0*1024*1024)) / (bench_ms / 1000.0);
    double actual_iops = (double)(total_threads * bench_ios) / (bench_ms / 1000.0);

    printf("\nResult:\n");
    printf("  Time: %.2f ms\n", bench_ms);
    printf("  Total data: %.2f GiB\n", total_bytes / (1024.0*1024*1024));
    printf("  Bandwidth: %.2f GiB/s\n", bw_gibs);
    printf("  IOPS: %.2f M\n", actual_iops / 1e6);

    cuda_err_chk(cudaEventDestroy(t0));
    cuda_err_chk(cudaEventDestroy(t1));

    delete ctrls[0];
    return 0;
}
