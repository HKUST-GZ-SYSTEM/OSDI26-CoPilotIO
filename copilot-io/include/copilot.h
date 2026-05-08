#ifndef __COPILOT_H__
#define __COPILOT_H__

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <gdrapi.h>
#include "nvm_types.h"

static inline void copilot_sfence() { __asm__ volatile("sfence" ::: "memory"); }
static inline void copilot_pause()  { __asm__ volatile("pause"  ::: "memory"); }

struct copilot_notify_entry {
    volatile uint32_t ready;
    uint32_t pos;
    uint32_t loc;
    uint32_t cq_head;
};

#define COPILOT_NUM_CIDS 65536
#define COPILOT_NOTIFY_BYTES (COPILOT_NUM_CIDS * sizeof(copilot_notify_entry))

#define COPILOT_CQ_MAX_DEPTH 4096

struct copilot_cq_table_entry {
    uint16_t cid;
    uint16_t sq_head;
};

struct copilot_cq_table {
    volatile uint32_t ready[COPILOT_CQ_MAX_DEPTH];
    copilot_cq_table_entry entries[COPILOT_CQ_MAX_DEPTH];
    volatile uint32_t drain_lock;
    volatile uint32_t gpu_head;
};

class CoPilotNotify {
public:
    copilot_notify_entry* gpu_ptr;
    copilot_notify_entry* cpu_ptr;

private:
    gdr_t gdr;
    gdr_mh_t mh;
    void* raw_gpu_ptr;
    void* map_ptr;
    size_t alloc_size;
    size_t pin_size;

public:
    CoPilotNotify() : gpu_ptr(nullptr), cpu_ptr(nullptr),
                      gdr(nullptr), raw_gpu_ptr(nullptr), map_ptr(nullptr),
                      alloc_size(0), pin_size(0) {}

    void init() {
        pin_size = (COPILOT_NOTIFY_BYTES + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        alloc_size = pin_size + GPU_PAGE_SIZE;

        cudaError_t err = cudaMalloc(&raw_gpu_ptr, alloc_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CoPilotNotify: cudaMalloc(%zu) failed: %s\n",
                    alloc_size, cudaGetErrorString(err));
            exit(1);
        }

        unsigned long raw_addr = (unsigned long)(uintptr_t)raw_gpu_ptr;
        unsigned long aligned_addr = (raw_addr + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        gpu_ptr = (copilot_notify_entry*)(uintptr_t)aligned_addr;
        cudaMemset(gpu_ptr, 0, pin_size);

        gdr = gdr_open();
        if (!gdr) {
            fprintf(stderr, "CoPilotNotify: gdr_open failed (is gdrdrv loaded?)\n");
            exit(1);
        }

        int ret = gdr_pin_buffer(gdr, aligned_addr, pin_size, 0, 0, &mh);
        if (ret) {
            fprintf(stderr, "CoPilotNotify: gdr_pin_buffer failed: %d\n", ret);
            exit(1);
        }

        ret = gdr_map(gdr, mh, &map_ptr, pin_size);
        if (ret) {
            fprintf(stderr, "CoPilotNotify: gdr_map failed: %d\n", ret);
            exit(1);
        }

        gdr_info_t info;
        gdr_get_info(gdr, mh, &info);
        ptrdiff_t off = (ptrdiff_t)(aligned_addr - info.va);
        cpu_ptr = (copilot_notify_entry*)((char*)map_ptr + off);

        memset((void*)cpu_ptr, 0, COPILOT_NOTIFY_BYTES);
    }

    void destroy() {
        if (map_ptr) {
            gdr_unmap(gdr, mh, map_ptr, pin_size);
            map_ptr = nullptr;
        }
        if (gdr) {
            gdr_unpin_buffer(gdr, mh);
            gdr_close(gdr);
            gdr = nullptr;
        }
        if (raw_gpu_ptr) {
            cudaFree(raw_gpu_ptr);
            raw_gpu_ptr = nullptr;
        }
        gpu_ptr = nullptr;
        cpu_ptr = nullptr;
    }

    ~CoPilotNotify() { destroy(); }
};

class CoPilotCQTable {
public:
    copilot_cq_table* gpu_ptr;
    copilot_cq_table* cpu_ptr;

private:
    gdr_t gdr;
    gdr_mh_t mh;
    void* raw_gpu_ptr;
    void* map_ptr;
    size_t alloc_size;
    size_t pin_size;

public:
    CoPilotCQTable() : gpu_ptr(nullptr), cpu_ptr(nullptr),
                       gdr(nullptr), raw_gpu_ptr(nullptr), map_ptr(nullptr),
                       alloc_size(0), pin_size(0) {}

    void init() {
        size_t table_bytes = sizeof(copilot_cq_table);
        pin_size = (table_bytes + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        alloc_size = pin_size + GPU_PAGE_SIZE;

        cudaError_t err = cudaMalloc(&raw_gpu_ptr, alloc_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CoPilotCQTable: cudaMalloc(%zu) failed: %s\n",
                    alloc_size, cudaGetErrorString(err));
            exit(1);
        }

        unsigned long raw_addr = (unsigned long)(uintptr_t)raw_gpu_ptr;
        unsigned long aligned_addr = (raw_addr + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        gpu_ptr = (copilot_cq_table*)(uintptr_t)aligned_addr;
        cudaMemset(gpu_ptr, 0, pin_size);

        gdr = gdr_open();
        if (!gdr) {
            fprintf(stderr, "CoPilotCQTable: gdr_open failed\n");
            exit(1);
        }

        int ret = gdr_pin_buffer(gdr, aligned_addr, pin_size, 0, 0, &mh);
        if (ret) {
            fprintf(stderr, "CoPilotCQTable: gdr_pin_buffer failed: %d\n", ret);
            exit(1);
        }

        ret = gdr_map(gdr, mh, &map_ptr, pin_size);
        if (ret) {
            fprintf(stderr, "CoPilotCQTable: gdr_map failed: %d\n", ret);
            exit(1);
        }

        gdr_info_t info;
        gdr_get_info(gdr, mh, &info);
        ptrdiff_t off = (ptrdiff_t)(aligned_addr - info.va);
        cpu_ptr = (copilot_cq_table*)((char*)map_ptr + off);

        memset((void*)cpu_ptr, 0, sizeof(copilot_cq_table));
    }

    void destroy() {
        if (map_ptr) {
            gdr_unmap(gdr, mh, map_ptr, pin_size);
            map_ptr = nullptr;
        }
        if (gdr) {
            gdr_unpin_buffer(gdr, mh);
            gdr_close(gdr);
            gdr = nullptr;
        }
        if (raw_gpu_ptr) {
            cudaFree(raw_gpu_ptr);
            raw_gpu_ptr = nullptr;
        }
        gpu_ptr = nullptr;
        cpu_ptr = nullptr;
    }

    ~CoPilotCQTable() { destroy(); }
};

struct copilot_cq_info {
    volatile nvm_cpl_t* cq;
    uint32_t qs_minus_1;
    uint32_t qs_log2;
    copilot_notify_entry* cpu_ptr;
    copilot_cq_table* cq_table_cpu_ptr;
    uint32_t head;
    uint64_t completions;
};

struct copilot_thread_args {
    copilot_cq_info* cqs;
    int n_cqs;
    int core_id;
    volatile bool* stop;
};

static void* copilot_poll_thread(void* arg) {
    copilot_thread_args* a = (copilot_thread_args*)arg;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(a->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    struct sched_param param;
    param.sched_priority = 99;
    sched_setscheduler(0, SCHED_FIFO, &param);

    int n_cqs = a->n_cqs;
    copilot_cq_info* cqs = a->cqs;
    uint64_t total_completions = 0;
    uint64_t next_log = 500000;

    while (!*a->stop) {
        bool any_progress = false;

        for (int q = 0; q < n_cqs; q++) {
            copilot_cq_info& ci = cqs[q];
            uint32_t pos = ci.head & ci.qs_minus_1;
            bool expected_phase = (~(ci.head >> ci.qs_log2)) & 1;

            uint32_t dword3 = ci.cq[pos].dword[3];
            bool phase = (dword3 >> 16) & 1;

            if (phase != expected_phase)
                continue;

            uint16_t cid = dword3 & 0xffff;

            copilot_notify_entry e;
            e.pos = pos;
            e.loc = ci.head;
            e.cq_head = ci.head;
            e.ready = 1;
            ci.cpu_ptr[cid] = e;
            copilot_sfence();

            if (ci.cq_table_cpu_ptr) {
                uint16_t sq_head = ci.cq[pos].dword[2] & 0xffff;
                copilot_cq_table_entry te;
                te.cid = cid;
                te.sq_head = sq_head;
                ci.cq_table_cpu_ptr->entries[pos] = te;
                copilot_sfence();
                ci.cq_table_cpu_ptr->ready[pos] = 1;
                copilot_sfence();
            }

            ci.head++;
            ci.completions++;
            total_completions++;
            any_progress = true;
        }

        if (!any_progress)
            copilot_pause();

        if (total_completions >= next_log) {
            next_log = total_completions + 500000;
            uint32_t total_cq_pending = 0;
            for (int q = 0; q < n_cqs; q++) {
                copilot_cq_info& ci = cqs[q];
                uint32_t scan = ci.head;
                for (;;) {
                    uint32_t spos = scan & ci.qs_minus_1;
                    bool exp = (~(scan >> ci.qs_log2)) & 1;
                    uint32_t d3 = ci.cq[spos].dword[3];
                    if (((d3 >> 16) & 1) != exp) break;
                    total_cq_pending++;
                    scan++;
                    if (total_cq_pending > 4096) break;
                }
            }
            printf("[CoPilot core %d] %d CQs, completions=%lu cq_pending=%u\n",
                   a->core_id, n_cqs, (unsigned long)total_completions, total_cq_pending);
        }
    }

    return nullptr;
}

class CoPilotAgent {
    static constexpr int MAX_THREADS = 64;
    static constexpr int MAX_QPS = 128;
    pthread_t threads[MAX_THREADS];
    copilot_thread_args thread_args[MAX_THREADS];
    copilot_cq_info cq_infos[MAX_QPS];
    volatile bool stop_flag;
    int n_threads;

public:
    CoPilotAgent() : stop_flag(false), n_threads(0) {}

    template<typename QP>
    void start(int n_qps, QP** h_qps, CoPilotNotify* notifies, int base_core, int n_cores = 0,
               CoPilotCQTable* cq_tables = nullptr) {
        if (n_cores <= 0) n_cores = n_qps;
        if (n_cores > n_qps) n_cores = n_qps;

        stop_flag = false;
        n_threads = n_cores;

        for (int i = 0; i < n_qps; i++) {
            cq_infos[i].cq               = (volatile nvm_cpl_t*)h_qps[i]->cq.vaddr_host;
            cq_infos[i].qs_minus_1       = h_qps[i]->cq.qs_minus_1;
            cq_infos[i].qs_log2          = h_qps[i]->cq.qs_log2;
            cq_infos[i].cpu_ptr          = notifies[i].cpu_ptr;
            cq_infos[i].cq_table_cpu_ptr = cq_tables ? cq_tables[i].cpu_ptr : nullptr;
            cq_infos[i].head             = 0;
            cq_infos[i].completions      = 0;
        }

        for (int t = 0; t < n_cores; t++) {
            int qps_start = (int64_t)t * n_qps / n_cores;
            int qps_end   = (int64_t)(t + 1) * n_qps / n_cores;

            thread_args[t].cqs    = &cq_infos[qps_start];
            thread_args[t].n_cqs  = qps_end - qps_start;
            thread_args[t].core_id = base_core + t;
            thread_args[t].stop   = &stop_flag;

            pthread_create(&threads[t], nullptr, copilot_poll_thread, &thread_args[t]);
        }
        printf("CoPilotAgent: %d QPs on %d cores (%d-%d), %d QPs/core, SCHED_FIFO 99\n",
               n_qps, n_cores, base_core, base_core + n_cores - 1,
               n_qps / n_cores);
    }

    void stop() {
        stop_flag = true;
        for (int i = 0; i < n_threads; i++)
            pthread_join(threads[i], nullptr);
        printf("CoPilotAgent: stopped\n");
    }
};

#endif
