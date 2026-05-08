// CoPilotIO User Guide — minimal example showing initialization and I/O interfaces.
// Uses 1 SM, 1 warp (32 threads), 1 QP, 1 CPU core, 1 I/O per thread.

#include <cuda.h>
#include <cstdio>
#include <vector>

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_queue.h>

#include <ctrl.h>
#include <queue.h>
#include <copilot.h>
#include <page_cache.h>
#include <util.h>

// ============================================================================
// Synchronous Interface: read_data_copilot()
//   GPU submits I/O → CPU polls CQ and writes notify → GPU polls notify.
//   One call does both submit and poll.
// ============================================================================
__global__ void sync_read_kernel(
    Controller** ctrls, page_cache_d_t* pc,
    uint64_t n_blocks_per_io, uint64_t ssd_capacity_blocks,
    copilot_notify_entry** d_notifies)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    volatile copilot_notify_entry* notify = d_notifies[0];  // QP 0
    uint64_t pc_idx = tid % pc->n_pages;
    uint64_t lba = (tid * n_blocks_per_io) % (ssd_capacity_blocks - n_blocks_per_io);

    // Single call: submit read command + wait for CPU-side CQ notification
    read_data_copilot(pc, ctrls[0]->d_qps, lba, n_blocks_per_io, pc_idx, notify);
}

// ============================================================================
// Asynchronous Interface: submit_read() + poll_copilot()
//   GPU submits a batch of I/Os first, then polls completions.
//   Decouples submission from completion — allows overlap.
// ============================================================================
__global__ void async_read_kernel(
    Controller** ctrls, page_cache_d_t* pc,
    uint64_t n_blocks_per_io, uint64_t ssd_capacity_blocks,
    copilot_notify_entry** d_notifies, copilot_cq_table** d_cq_tables)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    QueuePair* qp = ctrls[0]->d_qps;                       // QP 0
    volatile copilot_notify_entry* notify = d_notifies[0];
    copilot_cq_table* cq_table = d_cq_tables[0];

    // --- Submit phase: enqueue I/O commands, collect CIDs ---
    uint16_t cid;
    uint64_t pc_idx = tid % pc->n_pages;
    uint64_t lba = (tid * n_blocks_per_io) % (ssd_capacity_blocks - n_blocks_per_io);
    cid = submit_read(pc, qp, lba, n_blocks_per_io, pc_idx);

    // --- Poll phase: wait for CPU-side completion notification by CID ---
    poll_copilot(qp, cid, notify, cq_table);
}

int main()
{
    cudaSetDevice(0);

    // ========================================================================
    // 1. Create NVMe controller — SQ and CQ placed on CPU for CoPilotIO
    // ========================================================================
    const uint32_t n_qps = 1;
    const uint64_t qd = 1024;

    std::vector<Controller*> ctrls(1);
    ctrls[0] = new Controller("/dev/libnvm0", /*ns=*/1, /*gpu=*/0, qd, n_qps,
                              /*sq=*/QUEUE_CPU, /*cq=*/QUEUE_CPU);

    const uint64_t io_size = 4096;
    const uint64_t blk_size = ctrls[0]->blk_size;
    const uint64_t n_blocks_per_io = io_size / blk_size;
    const uint64_t ssd_capacity_blocks = ctrls[0]->ns.size;

    // Page cache (DMA buffer pool)
    const uint64_t n_pc_pages = 1024;
    page_cache_t pc(io_size, n_pc_pages, /*gpu=*/0, *ctrls[0], 64, ctrls);

    // ========================================================================
    // 2. Initialize CoPilotIO notify buffers (per-QP, GDRCopy-mapped)
    //    CPU writes notify entries → GPU reads them to detect I/O completion.
    // ========================================================================
    CoPilotNotify notify;
    notify.init();

    copilot_notify_entry* h_notify_ptr = notify.gpu_ptr;
    copilot_notify_entry** d_notify_ptrs;
    cudaMalloc(&d_notify_ptrs, sizeof(copilot_notify_entry*));
    cudaMemcpy(d_notify_ptrs, &h_notify_ptr, sizeof(copilot_notify_entry*),
               cudaMemcpyHostToDevice);

    // ========================================================================
    // 3. Initialize CoPilotIO CQ table (per-QP, needed for async interface)
    //    CPU fills CQ entries by ring position → GPU drains them in order.
    // ========================================================================
    CoPilotCQTable cq_table;
    cq_table.init();

    copilot_cq_table* h_cq_ptr = cq_table.gpu_ptr;
    copilot_cq_table** d_cq_table_ptrs;
    cudaMalloc(&d_cq_table_ptrs, sizeof(copilot_cq_table*));
    cudaMemcpy(d_cq_table_ptrs, &h_cq_ptr, sizeof(copilot_cq_table*),
               cudaMemcpyHostToDevice);

    // ========================================================================
    // 4. Start CoPilotAgent — launches CPU polling thread(s)
    //    Each thread polls CQ, writes notify + cq_table via GDRCopy.
    // ========================================================================
    CoPilotAgent agent;
    agent.start(n_qps, ctrls[0]->h_qps, &notify,
                /*base_core=*/16, /*n_cores=*/1, &cq_table);

    // ========================================================================
    // 5. Launch GPU kernels — 1 SM, 1 warp (32 threads), 1 I/O per thread
    // ========================================================================
    const int grid = 1;
    const int block = 32;

    printf("=== Synchronous read ===\n");
    sync_read_kernel<<<grid, block>>>(
        pc.pdt.d_ctrls, pc.d_pc_ptr,
        n_blocks_per_io, ssd_capacity_blocks,
        d_notify_ptrs);
    cudaDeviceSynchronize();
    printf("Done.\n");

    printf("=== Asynchronous read ===\n");
    async_read_kernel<<<grid, block>>>(
        pc.pdt.d_ctrls, pc.d_pc_ptr,
        n_blocks_per_io, ssd_capacity_blocks,
        d_notify_ptrs, d_cq_table_ptrs);
    cudaDeviceSynchronize();
    printf("Done.\n");

    // ========================================================================
    // 6. Cleanup
    // ========================================================================
    agent.stop();
    notify.destroy();
    cq_table.destroy();
    cudaFree(d_notify_ptrs);
    cudaFree(d_cq_table_ptrs);
    delete ctrls[0];
    return 0;
}
