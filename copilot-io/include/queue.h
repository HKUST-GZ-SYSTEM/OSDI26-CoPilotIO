#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <algorithm>
#include <cstdint>
#include "buffer.h"
#include "ctrl.h"
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include "nvm_admin.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include "util.h"

using error = std::runtime_error;
using std::string;

enum QueuePlacement { QUEUE_GPU = 0, QUEUE_CPU = 1 };

struct QueuePair
{
    uint32_t            pageSize;
    uint32_t            block_size;
    uint32_t            block_size_log;
    uint32_t            block_size_minus_1;
    uint32_t            nvmNamespace;
    //void*               prpList;
    //uint64_t*           prpListIoAddrs;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
    uint16_t            qp_id;
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    //BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    //BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    //BufferPtr           cq_tail_mark;
    BufferPtr           cq_pos_locks;
    //BufferPtr           cq_clean_cid;




#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)
#define MAX_SQ_ENTRIES_2M   (2*1024*1024/64)   // 32768
#define MAX_CQ_ENTRIES_2M   (2*1024*1024/16)   // 131072

    inline void init_gpu_specific_struct( const uint32_t cudaDevice) {
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        //this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_cid = createBuffer(65536 * sizeof(padded_struct), cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();
        //this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
    //    std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
        this->sq.qs_minus_1 = this->sq.qs - 1;
        this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);


        //this->cq_tickets = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq_tail_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq.tickets = (padded_struct*) this->cq_tickets.get();
        this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
        //this->cq.tail_mark = (padded_struct*) this->cq_tail_mark.get();
        this->cq.qs_minus_1 = this->cq.qs - 1;
        this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);
        this->cq_pos_locks = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq.pos_locks = (padded_struct*) this->cq_pos_locks.get();

        //this->cq_clean_cid = createBuffer(this->cq.qs * sizeof(uint16_t), cudaDevice);
       // this->cq.clean_cid = (uint16_t*) this->cq_clean_cid.get();
    }



    inline QueuePair( const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, nvm_aq_ref& aq_ref, const uint16_t qp_id, const uint64_t queueDepth, QueuePlacement sq_placement = QUEUE_GPU, QueuePlacement cq_placement = QUEUE_GPU)
    {
        //this->this = (QueuePairThis*) malloc(sizeof(QueuePairThis));


    //    std::cout << "HERE\n";
        uint64_t cap = ((volatile uint64_t*) ctrl->mm_ptr)[0];
        bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;
        //uint64_t sq_size = 16;
        //uint64_t cq_size = 16;

        uint64_t mqes = (((volatile uint16_t*) ctrl->mm_ptr)[0] + 1);
        bool cpu_queues = (sq_placement == QUEUE_CPU || cq_placement == QUEUE_CPU);
        uint64_t sq_cap = (cqr && cpu_queues) ? MAX_SQ_ENTRIES_2M : (cqr ? MAX_SQ_ENTRIES_64K : mqes);
        uint64_t cq_cap = (cqr && cpu_queues) ? MAX_CQ_ENTRIES_2M : (cqr ? MAX_CQ_ENTRIES_64K : mqes);
        uint64_t sq_size = std::min(sq_cap, mqes);
        uint64_t cq_size = std::min(cq_cap, mqes);
        sq_size = std::min(queueDepth, sq_size);
        cq_size = std::min(queueDepth, cq_size);

        size_t page_sz = ctrl->page_size;
        bool use_hugepage_sq = false;
        bool use_hugepage_cq = false;

        if (cqr) {
            uint64_t max_cq_cpu = page_sz / sizeof(nvm_cpl_t);
            uint64_t max_sq_cpu = page_sz / sizeof(nvm_cmd_t);
            if (cq_placement == QUEUE_CPU && cq_size > max_cq_cpu) {
                use_hugepage_cq = true;
                if (qp_id == 1)
                    printf("CQR=1: using hugepage for CPU CQ (depth %lu, needs %lu KB)\n",
                           cq_size, cq_size * sizeof(nvm_cpl_t) / 1024);
            }
            if (sq_placement == QUEUE_CPU && sq_size > max_sq_cpu) {
                use_hugepage_sq = true;
                if (qp_id == 1)
                    printf("CQR=1: using hugepage for CPU SQ (depth %lu, needs %lu KB)\n",
                           sq_size, sq_size * sizeof(nvm_cmd_t) / 1024);
            }
            if (sq_size > cq_size) sq_size = cq_size;
        }

        size_t sq_data_size = sq_size * sizeof(nvm_cmd_t);
        size_t cq_data_size = cq_size * sizeof(nvm_cpl_t);

        if (sq_placement == QUEUE_GPU) {
            this->sq_mem = createDma(ctrl, NVM_PAGE_ALIGN(sq_data_size, 1UL << 16), cudaDevice);
        } else if (use_hugepage_sq) {
            this->sq_mem = createDmaHugepage(ctrl, sq_data_size);
        } else {
            this->sq_mem = createDmaPinned(ctrl, NVM_PAGE_ALIGN(sq_data_size, page_sz));
        }
        if (cq_placement == QUEUE_GPU) {
            this->cq_mem = createDma(ctrl, NVM_PAGE_ALIGN(cq_data_size, 1UL << 16), cudaDevice);
        } else if (use_hugepage_cq) {
            this->cq_mem = createDmaHugepage(ctrl, cq_data_size);
        } else {
            this->cq_mem = createDmaPinned(ctrl, NVM_PAGE_ALIGN(cq_data_size, page_sz));
        }

        this->pageSize = info.page_size;
        this->block_size = ns.lba_data_size;
        this->block_size_minus_1 = ns.lba_data_size-1;
        this->block_size_log = std::log2(ns.lba_data_size);
        this->nvmNamespace = ns.ns_id;
        this->qp_id = qp_id;

        int status = nvm_admin_cq_create(aq_ref, &this->cq, qp_id, this->cq_mem.get(), 0, cq_size);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
        }
        // std::cout << "after nvm_admin_cq_create\n";

        // Save host-side pointers before GPU override (for CPU polling in CoPilotIO)
        this->cq.vaddr_host = this->cq.vaddr;

        // Get a valid device pointer for CQ doorbell
        void* devicePtr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) this->cq.db, 0);
        if (err != cudaSuccess)
        {
            throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->cq.db = (volatile uint32_t*) devicePtr;

        // Create submission queue
        //  nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)
        status = nvm_admin_sq_create(aq_ref, &this->sq, &this->cq, qp_id, this->sq_mem.get(), 0, sq_size);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
        }


        // Get a valid device pointer for SQ doorbell
        err = cudaHostGetDevicePointer(&devicePtr, (void*) this->sq.db, 0);
        if (err != cudaSuccess)
        {
            throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->sq.db = (volatile uint32_t*) devicePtr;
//        std::cout << "Finish Making Queue\n";

        init_gpu_specific_struct(cudaDevice);
       // std::cout << "in preparequeuepair: " << std::hex << this->sq.cid << std::endl;
        return;



    }

};
#endif
