#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

// Victim buffer compile switch (must be before page_cache_d_t definition).
// Defined here because page_cache.h is parsed before io_scheduler.cuh.
#ifndef IO_SCHED_VICTIM_BUFFER
#define IO_SCHED_VICTIM_BUFFER 1
#endif

// Victim buffer pointer-swap mode:
//   1 = swap virtual/IO address pointers instead of memcpy (eliminates 4KB copy)
//   0 = memcpy dirty page to victim buffer (original behavior)
// Only meaningful when IO_SCHED_VICTIM_BUFFER=1.
#ifndef IO_SCHED_VICTIM_SWAP
#define IO_SCHED_VICTIM_SWAP 1
#endif

// Forward declaration (full definition in io_scheduler.cuh).
struct victim_buffer_t;

// Forward declaration (full definition in copilot.h).
struct copilot_notify_entry;

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif


#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

#define FREE 2
// enum locality {HIGH_SPATIAL, LOW_SPATIAL, MEDIUM_SPATIAL};
// template <typname T>
// stryct array_t {
//   range_t<T>* ranges;
//   uint32_t n_ranges;

//   void add_range(start_idx, end_idx, locality l )
//   {
//   ranges.push_back(new range(star))
//   }
// }

// enum page_state {USE = 1U, USE_DIRTY = ((1U << 31) | 1), VALID_DIRTY = (1U << 31),
//     VALID = 0U, INVALID = (UINT_MAX & 0x7fffffff),
//     BUSY = ((UINT_MAX & 0x7fffffff)-1)};

enum data_dist_t {REPLICATE = 0, STRIPE = 1};

// #define USE (1ULL)
// #define VALID_DIRTY (1ULL << 31)
// #define USE_DIRTY (VALID_DIRTY | USE)
// #define VALID (0ULL)
// #define INVALID (0xffffffffULL)
// #define BUSY ((0xffffffffULL)-1)

#define ALL_CTRLS 0xffffffffffffffff

//broken
#define VALID_DIRTY (1ULL << 31)
#define USE_DIRTY (VALID_DIRTY | USE)

#define INVALID 0x00000000U
#define VALID   0x80000000U
#define BUSY    0x40000000U
#define DIRTY   0x20000000U
#define CNT_SHIFT (29ULL)
#define CNT_MASK 0x1fffffffU
#define VALID_MASK 0x7
#define BUSY_MASK 0xb
#define DISABLE_BUSY_ENABLE_VALID 0xc0000000U
#define DISABLE_BUSY_MASK 0xbfffffffU
#define NV_NB 0x00U
#define NV_B 0x01U
#define V_NB 0x02U
#define V_B 0x03U


struct page_cache_t;

struct page_cache_d_t;

//typedef padded_struct_pc* page_states_t;


template <typename T>
struct range_t;

template<typename T>
struct array_d_t;

template <typename T>
struct range_d_t;

/*struct data_page_t {
  simt::atomic<uint64_t, simt::thread_scope_device>  state; //state
  //
  uint32_t offset;

  };
*/
typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  state; //state
                                                              //
    uint32_t offset;
    //uint8_t pad[32-4-4];

} __attribute__((aligned (32))) data_page_t;

typedef data_page_t* pages_t;

template<typename T>
struct returned_cache_page_t {
    T* addr;
    uint32_t size;
    uint32_t offset;

    T operator[](size_t i) const {
        if (i < size)
            return addr[i];
        else
            return 0;
    }

    T& operator[](size_t i) {
        if (i < size)
            return addr[i];
        else
            return addr[0];
    }
};
#define THREAD_ 0
#define SHARED_ 1
#define GLOBAL_ 2

//#ifdef __CUDACC__
#define TID ( (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)))
//#else
//#define TID 0
//#endif

//#ifdef __CUDACC__
#define BLKSIZE ( (blockDim.x * blockDim.y * blockDim.z) )
//#else
//#define BLKSIZE 1
//#endif

#ifdef __CUDACC__
#define SYNC (loc != THREAD_ ? __syncthreads() : (void)0)
#else
#define SYNC (void)0
#endif




#define INVALID_ 0x00000000
#define VALID_ 0x80000000
#define BUSY_ 0x40000000
#define CNT_MASK_ 0x3fffffff
#define INVALID_MASK_ 0x7fffffff
#define DISABLE_BUSY_MASK_ 0xbfffffff

template<simt::thread_scope _scope = simt::thread_scope_device>
struct tlb_entry {
    uint64_t global_id;
    simt::atomic<uint32_t, _scope> state;
    data_page_t* page = nullptr;

    __forceinline__
    __host__ __device__
    tlb_entry() { init(); }

    __forceinline__
    __host__ __device__
    void init() {
        global_id = 0;
        state.store(0, simt::memory_order_relaxed);
        page = nullptr;
    }

    __forceinline__
    __device__
    void release(const uint32_t count) {
//		    if (global_id == 515920192)
//			printf("--(2)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) count);

        state.fetch_sub(count, simt::memory_order_release); }

    __forceinline__
    __device__
    void release() { if (page != nullptr)  {
//		    if (global_id == 515920192)
//			printf("--(1)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) 1);

            page->state.fetch_sub(1, simt::memory_order_release); }}



};




template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct tlb {
    tlb_entry<_scope> entries[n];
    array_d_t<T>* array = nullptr;

    __forceinline__
    __host__ __device__
    tlb() {}
/*
  __forceinline__
  __device__
  tlb(array_d_t<T>* a) { init(a); }
*/
    __forceinline__
    __device__
    void init(array_d_t<T>* a) {
        //SYNC;
//	__syncthreads();
        if (n) {
            size_t tid = TID;
            if (tid == 0)
                array = a;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].init();
        }

//	__syncthreads();
//        SYNC;

    }

    __forceinline__
    __device__
    void fini() {
        //      SYNC;
//	__syncthreads();

        if (n) {
            size_t tid = TID;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].release();
        }
//	__syncthreads();

//        SYNC;
    }

    __forceinline__
    __device__
    ~tlb() {  }

    __forceinline__
    __device__
    T* acquire(const size_t i, const size_t gid, size_t& start, size_t& end, range_d_t<T>* range, const size_t page_) {

        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);
        uint64_t base_master, base;
        if (lane == master) {
            uint64_t c = 0;
            bool cont = false;
            uint32_t st;
            do {

                //lock;
                do {
                    st = entry->state.fetch_or(VALID_, simt::memory_order_acquire);
                    if ((st & VALID_) == 0)
                        break;
                    __nanosleep(100);
                } while (true);

                if ((entry->page != nullptr) && (gid == entry->global_id)) {
//		    if (gid == 515920192)
//			printf("++(1)st: %llx\tst&Val: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) count);

                    st += count;

                    base_master = (uint64_t) range->get_cache_page_addr(entry->page->offset);

                    entry->state.store(st, simt::memory_order_release);
                    break;
                }
                else if(((entry->page == nullptr)) || (((st & 0x3fffffff) == 0))) {
//		    if (gid == 515920192)
//			printf("++(2)st: %llx\tst&Val: %llx\tVal: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) VALID_, (unsigned long long) count);
//
                    if (entry->page != nullptr)
                        entry->page->state.fetch_sub(1, simt::memory_order_release);
                    data_page_t* page = nullptr;// = (data_page_t*)0xffffffffffffffff;
                    base_master = (uint64_t) array->acquire_page_(i, page, start, end, range, page_);
                    if (((uint64_t) page == 0xffffffffffffffff) || (page == nullptr))
                        printf("failure\n");
                    entry->page = page;
                    entry->global_id = gid;
                    st += count;
                    entry->state.store(st, simt::memory_order_release);
                    break;

                }
                else {
                    if (++c % 100000 == 0)
                        printf("c: %llu\ttid: %llu\twanted_gid: %llu\tgot_gid: %llu\tst: %llx\tst&0x7: %llx\n", (unsigned long long) c, (unsigned long long) (TID), (unsigned long long) gid, (unsigned long long) entry->global_id, (unsigned long long) st, (unsigned long long) (st & 0x7fffffff));
                    entry->state.store(st, simt::memory_order_relaxed);
                    __nanosleep(100);

                }

            } while(true);


        }

        base_master = __shfl_sync(eq_mask,  base_master, master);

        return (T*) base_master;

    }

    __forceinline__
    __device__
    void release(const size_t gid) {
        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);

        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        if (lane == master)
            entry->release(count);
        __syncwarp(eq_mask);

    }

};

template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct bam_ptr_tlb {
    tlb<T,n,_scope,loc>* tlb_ = nullptr;
    array_d_t<T>* array = nullptr;
    range_d_t<T>* range;
    size_t page;
    size_t start = 0;
    size_t end = 0;
    size_t gid = 0;
    //int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr_tlb(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { init(a, t); }

    __forceinline__
    __device__
    ~bam_ptr_tlb() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { array = a; tlb_ = t; }

    __forceinline__
    __device__
    void fini(void) {
        if (addr) {

            tlb_->release(gid);
            addr = nullptr;
        }

    }

    __forceinline__
    __device__
    void update_page(const size_t i) {
        ////printf("++++acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        fini(); //destructor
        array->get_page_gid(i, range, page, gid);
        addr = (T*) tlb_->acquire(i, gid, start, end, range, page);
//        //printf("----acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
    }

    __forceinline__
    __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
            update_page(i);
        }
        return addr[i-start];
    }

    __forceinline__
    __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            range->mark_page_dirty(page);
        }
        return addr[i-start];
    }
};


template<typename T>
struct bam_ptr {
    data_page_t* page = nullptr;
    array_d_t<T>* array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr(array_d_t<T>* a) { init(a); }

    __forceinline__
    __host__ __device__
    ~bam_ptr() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a) { array = a; }

    __forceinline__
    __host__ __device__
    void fini(void) {
        if (page) {
            array->release_page(page, range_id, start);
            page = nullptr;
        }

    }

    __forceinline__
    __host__ __device__
    T* update_page(const size_t i) {
        ////printf("++++acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        fini(); //destructor
        addr = (T*) array->acquire_page(i, page, start, end, range_id);
//        //printf("----acquire: i: %llu\tpage: %llu\tstart: %llu\tend: %llu\trange: %llu\n",
//            (unsigned long long) i, (unsigned long long) page, (unsigned long long) start, (unsigned long long) end, (unsigned long long) range_id);
        return addr;
    }

    __forceinline__
    __host__ __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
           T* tmpaddr =  update_page(i);
        }
        return addr[i-start];
    }
    
    __host__ __device__
    T* memref(size_t i) {
        T* ret_; 
        if ((i < start) || (i >= end)) {
           ret_ =  update_page(i);
        }
        return ret_;
    }


    __forceinline__
    __host__ __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        return addr[i-start];
    }
};

typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock; //state
    //
    uint64_t page_translation;
    uint8_t pad[32-8];

} __attribute__((aligned (32))) cache_page_t;
/*
  struct cache_page_t {
  simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock;
  uint32_t  page_translation;
  uint8_t   range_id;
  };
*/
struct page_cache_d_t {
    uint8_t* base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    cache_page_t* cache_pages;
    //uint32_t* page_translation;         //len = num of pages in cache
    //padded_struct_pc* page_translation;         //len = num of pages in cache
    //padded_struct_pc* page_take_lock;      //len = num of pages in cache
    padded_struct_pc* page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t* prp_list;              //len = num of pages in cache if page_size > ctrl.page_size *2
    uint64_t* page_ioaddrs;          // flat: page_ioaddrs[pc_entry * pages_per_entry + j] = j-th 4KB sub-page ioaddr
    uint32_t  pages_per_entry;       // = page_size / ctrl_page_size (NVMe pages per cache page)
    uint64_t    ctrl_page_size;
    uint64_t  range_cap;
    //uint64_t  range_count;
    pages_t*   ranges;
    pages_t*   h_ranges;
    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;
    uint64_t n_cachelines_for_states;

    uint64_t* ranges_page_starts;
    data_dist_t* ranges_dists;
    simt::atomic<uint64_t, simt::thread_scope_device>* ctrl_counter;

    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;
    simt::atomic<uint64_t, simt::thread_scope_device>* write_io_cmd_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device>* write_io_block_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device>* write_cycles_total;
    simt::atomic<uint64_t, simt::thread_scope_device>* write_call_count;
    uint64_t read_ctrl_override;
    uint64_t evict_ctrl_override;

    Controller** d_ctrls;
    uint64_t n_ctrls;
    bool prps;

    uint64_t n_blocks_per_page;

    uint64_t fake_evict_delay_cycles;  // if >0, replace write_data with busy-wait of this many cycles

    uint8_t* staging_buffer;  // if non-null, fake evict does memcpy to here instead of busy-wait

    struct io_queue_t* evict_queue;  // if non-null, eviction writes go through IO scheduler
    struct io_queue_t* read_queue;   // if non-null, cache-miss reads go through IO scheduler

    struct victim_buffer_t* victim;  // if non-null, eviction uses victim buffer (fire-and-forget)

#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
    uint64_t* page_vaddrs;  // [n_pages] current virtual address of each cache page's data buffer
#endif

    __forceinline__
    __device__
    cache_page_t* get_cache_page(const uint32_t page) const;

    __forceinline__
    __device__
    uint32_t find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_);


};


__device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void read_data_batch_warp32(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void read_data_batch_warp32_profile(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, uint64_t* pre_cycles, uint64_t* poll_cycles, uint64_t* total_cycles);
__device__ void read_data_batch(page_cache_d_t* pc, QueuePair* qp, const uint64_t* starting_lbas, const uint64_t* n_blocks_array, const unsigned long long* pc_entries, const uint32_t batch_size);
__device__ void write_data_batch(page_cache_d_t* pc, QueuePair* qp, const uint64_t* starting_lbas, const uint64_t* n_blocks_array, const unsigned long long* pc_entries, const uint32_t batch_size);
__device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void read_data_copilot(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, volatile copilot_notify_entry* notify);
__device__ void read_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void write_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void access_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode);
__device__ void trim_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);

__forceinline__
__device__
uint64_t get_backing_page_(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t page = page_start;
    if (dist == STRIPE) {
        page += page_offset / n_ctrls;
    }
    else if (dist == REPLICATE) {
        page += page_offset;
    }

    return page;
}
__forceinline__
__device__
uint64_t get_backing_ctrl_(const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t ctrl;

    if (dist == STRIPE) {
        ctrl = page_offset % n_ctrls;
    }
    else if (dist == REPLICATE) {
        ctrl = ALL_CTRLS;
    }
    return ctrl;

}

__global__
void __flush(page_cache_d_t* pc) {
    uint64_t page = threadIdx.x + blockIdx.x * blockDim.x;

    if (page < pc->n_pages) {
        uint64_t previous_global_address = pc->cache_pages[page].page_translation;
        //uint8_t previous_range = this->cache_pages[page].range_id;
        uint64_t previous_range = previous_global_address & pc->n_ranges_mask;
        uint64_t previous_address = previous_global_address >> pc->n_ranges_bits;
        //uint32_t new_state = BUSY;

        uint32_t expected_state = pc->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

        uint32_t d = expected_state & DIRTY;
        uint32_t smid = get_smid();
        if (d) {

            uint64_t ctrl = get_backing_ctrl_(previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            if (pc->evict_ctrl_override != ALL_CTRLS)
                ctrl = pc->evict_ctrl_override % pc->n_ctrls;
            //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
            uint64_t index = get_backing_page_(pc->ranges_page_starts[previous_range], previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            // //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
            //        (unsigned long long) previous_range, (unsigned long long)previous_address,
            //        (unsigned long long) ctrl, (unsigned long long) index);
            if (ctrl == ALL_CTRLS) {
                for (ctrl = 0; ctrl < pc->n_ctrls; ctrl++) {
                    Controller* c = pc->d_ctrls[ctrl];
                    uint32_t queue = smid % (c->n_qps);
                    write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                }
            }
            else {

                Controller* c = pc->d_ctrls[ctrl];
                uint32_t queue = smid % (c->n_qps);

                //index = ranges_page_starts[previous_range] + previous_address;


                write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
            }

            pc->ranges[previous_range][previous_address].state.fetch_and(~DIRTY);

        }
    }
}

struct page_cache_t {


    //void* d_pc;

    //BufferPtr prp2_list_buf;
    page_cache_d_t pdt;
    //void* d_pc;
    //BufferPtr prp2_list_buf;
    //bool prps;
    pages_t*   h_ranges;
    uint64_t* h_ranges_page_starts;
    data_dist_t* h_ranges_dists;
    page_cache_d_t* d_pc_ptr;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr page_ioaddrs_buf;
    BufferPtr cache_pages_buf;
    //BufferPtr page_translation_buf;
    //BufferPtr page_take_lock_buf;
    BufferPtr ranges_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;
    BufferPtr ranges_page_starts_buf;
    BufferPtr ranges_dists_buf;

    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;
    BufferPtr q_head_buf;
    BufferPtr q_tail_buf;
    BufferPtr q_lock_buf;
    BufferPtr extra_reads_buf;
    BufferPtr write_io_cmd_cnt_buf;
    BufferPtr write_io_block_cnt_buf;
    BufferPtr write_cycles_total_buf;
    BufferPtr write_call_count_buf;

    void print_reset_stats(void) {
        uint64_t v = 0;
        cuda_err_chk(cudaMemcpy(&v, pdt.extra_reads, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaMemset(pdt.extra_reads, 0, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));

//        printf("Cache Extra Reads: %llu\n", v);
    }

    void get_reset_io_stats(uint64_t* write_cmds, uint64_t* write_blocks) {
        uint64_t wc = 0;
        uint64_t wb = 0;
        cuda_err_chk(cudaMemcpy(&wc, pdt.write_io_cmd_cnt,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>),
                                cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&wb, pdt.write_io_block_cnt,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>),
                                cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaMemset(pdt.write_io_cmd_cnt, 0,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
        cuda_err_chk(cudaMemset(pdt.write_io_block_cnt, 0,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));

        if (write_cmds != nullptr) *write_cmds = wc;
        if (write_blocks != nullptr) *write_blocks = wb;
    }

    void print_reset_io_stats(uint64_t blk_size) {
        uint64_t write_cmds = 0;
        uint64_t write_blocks = 0;
        get_reset_io_stats(&write_cmds, &write_blocks);
        const double mib = ((double)write_blocks * (double)blk_size) / (1024.0 * 1024.0);
        uint64_t wc_total = 0, wc_count = 0;
        cuda_err_chk(cudaMemcpy(&wc_total, pdt.write_cycles_total,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>),
                                cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&wc_count, pdt.write_call_count,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>),
                                cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemset(pdt.write_cycles_total, 0,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
        cuda_err_chk(cudaMemset(pdt.write_call_count, 0,
                                sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
        std::printf("#PCWritebacks: cmds=%llu blocks=%llu (%.2f MiB)",
                    (unsigned long long)write_cmds,
                    (unsigned long long)write_blocks,
                    mib);
        if (wc_count > 0) {
            std::printf(" avg_write_cycles=%llu (n=%llu)",
                        (unsigned long long)(wc_total / wc_count),
                        (unsigned long long)wc_count);
        }
        std::printf("\n");
    }

    void set_ctrl_overrides(uint64_t read_ctrl, uint64_t evict_ctrl) {
        pdt.read_ctrl_override = read_ctrl;
        pdt.evict_ctrl_override = evict_ctrl;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void set_fake_evict_delay(uint64_t delay_cycles) {
        pdt.fake_evict_delay_cycles = delay_cycles;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void set_evict_queue(struct io_queue_t* q) {
        pdt.evict_queue = q;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void set_read_queue(struct io_queue_t* q) {
        pdt.read_queue = q;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void set_victim(struct victim_buffer_t* vb) {
        pdt.victim = vb;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void enable_staging_buffer() {
        uint64_t buf_size = pdt.page_size * pdt.n_pages;
        uint8_t* buf = nullptr;
        cuda_err_chk(cudaMalloc(&buf, buf_size));
        pdt.staging_buffer = buf;
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    void flush_cache() {
        size_t threads = 64;
        size_t n_blocks = (pdt.n_pages + threads - 1) / threads;

        __flush<<<n_blocks, threads>>>(d_pc_ptr);


    }

    template <typename T>
    void add_range(range_t<T>* range) {
        range->rdt.range_id  = pdt.n_ranges++;
        h_ranges[range->rdt.range_id] = range->rdt.pages;
        h_ranges_page_starts[range->rdt.range_id] = range->rdt.page_start;
        h_ranges_dists[range->rdt.range_id] = range->rdt.dist;
        cuda_err_chk(cudaMemcpy(pdt.ranges_page_starts, h_ranges_page_starts, pdt.n_ranges * sizeof(uint64_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges, h_ranges, pdt.n_ranges* sizeof(pages_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges_dists, h_ranges_dists, pdt.n_ranges* sizeof(data_dist_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));

    }

    page_cache_t(const uint64_t ps, const uint64_t np, const uint32_t cudaDevice, const Controller& ctrl, const uint64_t max_range, const std::vector<Controller*>& ctrls) {

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_head_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_tail_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_lock_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        extra_reads_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        write_io_cmd_cnt_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        write_io_block_cnt_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        write_cycles_total_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        write_call_count_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();
        pdt.page_size = ps;
        pdt.q_head = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_head_buf.get();
        pdt.q_tail = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_tail_buf.get();
        pdt.q_lock = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_lock_buf.get();
        pdt.extra_reads = (simt::atomic<uint64_t, simt::thread_scope_device>*)extra_reads_buf.get();
        pdt.write_io_cmd_cnt = (simt::atomic<uint64_t, simt::thread_scope_device>*)write_io_cmd_cnt_buf.get();
        pdt.write_io_block_cnt = (simt::atomic<uint64_t, simt::thread_scope_device>*)write_io_block_cnt_buf.get();
        pdt.write_cycles_total = (simt::atomic<uint64_t, simt::thread_scope_device>*)write_cycles_total_buf.get();
        pdt.write_call_count = (simt::atomic<uint64_t, simt::thread_scope_device>*)write_call_count_buf.get();
        pdt.read_ctrl_override = ALL_CTRLS;
        pdt.evict_ctrl_override = ALL_CTRLS;
        pdt.page_size_minus_1 = ps - 1;
        pdt.n_pages = np;
        pdt.ctrl_page_size = ctrl.ctrl->page_size;
        pdt.n_pages_minus_1 = np - 1;
        pdt.n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(pdt.n_ctrls * sizeof(Controller*), cudaDevice);
        pdt.d_ctrls = (Controller**) d_ctrls_buff.get();
        pdt.n_blocks_per_page = (ps/ctrl.blk_size);
        pdt.fake_evict_delay_cycles = 0;
        pdt.staging_buffer = nullptr;
        pdt.evict_queue = nullptr;
        pdt.read_queue = nullptr;
        pdt.victim = nullptr;
#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
        pdt.page_vaddrs = nullptr;
#endif
        pdt.n_cachelines_for_states = np/STATES_PER_CACHELINE;
        for (size_t k = 0; k < pdt.n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(pdt.d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        //n_ctrls = ctrls.size();
        //d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), cudaDevice);
        //d_ctrls = (Controller**) d_ctrls_buff.get();
        //for (size_t k = 0; k < n_ctrls; k++)
        //    cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));

        pdt.range_cap = max_range;
        pdt.n_ranges = 0;
        pdt.n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        pdt.n_ranges_mask = max_range-1;
        std::cout << "n_ranges_bits: " << std::dec << pdt.n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << pdt.n_ranges_mask << std::endl;

        pdt.page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(pages_t), cudaDevice);
        pdt.ranges = (pages_t*)ranges_buf.get();
        h_ranges = new pages_t[max_range];

        h_ranges_page_starts = new uint64_t[max_range];
        std::memset(h_ranges_page_starts, 0, max_range * sizeof(uint64_t));

        //pages_translation_buf = createBuffer(np * sizeof(uint32_t), cudaDevice);
        //pdt.page_translation = (uint32_t*)page_translation_buf.get();
        //page_translation_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //page_translation = (padded_struct_pc*)page_translation_buf.get();

        //page_take_lock_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //pdt.page_take_lock =  (padded_struct_pc*)page_take_lock_buf.get();

        cache_pages_buf = createBuffer(np * sizeof(cache_page_t), cudaDevice);
        pdt.cache_pages = (cache_page_t*)cache_pages_buf.get();

        ranges_page_starts_buf = createBuffer(max_range * sizeof(uint64_t), cudaDevice);
        pdt.ranges_page_starts = (uint64_t*) ranges_page_starts_buf.get();

        page_ticket_buf = createBuffer(1 * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_ticket =  (padded_struct_pc*)page_ticket_buf.get();
        //std::vector<padded_struct_pc> tps(np, FREE);
        cache_page_t* tps = new cache_page_t[np];
        for (size_t i = 0; i < np; i++)
            tps[i].page_take_lock = FREE;
        cuda_err_chk(cudaMemcpy(pdt.cache_pages, tps, np*sizeof(cache_page_t), cudaMemcpyHostToDevice));
        delete tps;

        ranges_dists_buf = createBuffer(max_range * sizeof(data_dist_t), cudaDevice);
        pdt.ranges_dists = (data_dist_t*)ranges_dists_buf.get();
        h_ranges_dists = new data_dist_t[max_range];

        uint64_t cache_size = ps*np;
        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        pdt.base_addr = (uint8_t*) this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((pdt.page_size > (ctrl.ctrl->page_size * uints_per_page)) || (np == 0) || (pdt.page_size < ctrl.ns.lba_data_size))
            throw error(string("page_cache_t: Can't have such page size or number of pages"));
        if (ps <= this->pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/ps;
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();


            std::cout << np << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one *  this->pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  this->pages_dma.get()->n_ioaddrs);
            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps;
                    //std::cout << std::dec << "\ti: " << i << "\tj: " << j << "\tindex: "<< (i*how_many_in_one + j) << "\t" << std::hex << (((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps) << std::dec << std::endl;
                }
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete temp;
            //std::cout << "HERE1\n";
            //free(temp);
            //std::cout << "HERE2\n";
            pdt.prps = false;
        }

        else if ((ps > this->pages_dma.get()->page_size) && (ps <= (this->pages_dma.get()->page_size * 2))) {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            //uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            //uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            pdt.prps = true;
        }
        else {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp3 = new uint64_t[prp_list_size];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            std::memset(temp3, 0, prp_list_size);
            uint32_t how_many_in_one = ps /  ctrl.ctrl->page_size ;
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one]);
                temp2[i] = ((uint64_t) this->prp_list_dma.get()->ioaddrs[i]);
                for(size_t j = 0; j < (how_many_in_one-1); j++) {
                    temp3[i*uints_per_page + j] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one + j + 1]);
                }
            }
            /*
              for (size_t i = 0; i < this->pages_dma.get()->n_ioaddrs; i+=how_many_in_one) {
              temp1[i/how_many_in_one] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]);
              temp2[i/how_many_in_one] = ((uint64_t)this->prp_list_dma.get()->ioaddrs[i]);
              for (size_t j = 0; j < (how_many_in_one-1); j++) {

              temp3[(i/how_many_in_one)*uints_per_page + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i+1+j]);
              }
              }
            */

            std::cout << "Done creating PRP\n";
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            delete temp3;
            pdt.prps = true;
        }

        // Build flat page_ioaddrs[] array for cross-entry PRP merge support.
        // page_ioaddrs[pc_entry * ppe + j] = physical address of j-th 4KB sub-page.
        {
            uint32_t ppe = (ps <= ctrl.ctrl->page_size) ? 1 : (uint32_t)(ps / ctrl.ctrl->page_size);
            pdt.pages_per_entry = ppe;
            uint64_t n_sub = (uint64_t)np * ppe;
            this->page_ioaddrs_buf = createBuffer(n_sub * sizeof(uint64_t), cudaDevice);
            pdt.page_ioaddrs = (uint64_t*)this->page_ioaddrs_buf.get();

            uint64_t* tmp_io = new uint64_t[n_sub];
            if (ppe == 1 && (ps <= this->pages_dma.get()->page_size)) {
                // ps <= ctrl_page_size: might have multiple cache pages per DMA page
                uint64_t how_many = ctrl.ctrl->page_size / ps;
                for (size_t i = 0; i < (size_t)np; i++) {
                    size_t dma_page = i / how_many;
                    size_t dma_off  = i % how_many;
                    tmp_io[i] = (uint64_t)this->pages_dma.get()->ioaddrs[dma_page] + dma_off * ps;
                }
            } else {
                // ps >= ctrl_page_size: direct mapping
                for (size_t k = 0; k < (size_t)n_sub; k++)
                    tmp_io[k] = (uint64_t)this->pages_dma.get()->ioaddrs[k];
            }
            cuda_err_chk(cudaMemcpy(pdt.page_ioaddrs, tmp_io, n_sub * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete[] tmp_io;
        }

#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
        // Build page_vaddrs[] indirection array for pointer-swap eviction.
        // Initially page_vaddrs[i] = base_addr + i * page_size (identity mapping).
        {
            uint64_t* h_va = new uint64_t[np];
            for (size_t i = 0; i < (size_t)np; i++)
                h_va[i] = (uint64_t)(pdt.base_addr + (uint64_t)i * ps);
            cuda_err_chk(cudaMalloc(&pdt.page_vaddrs, (size_t)np * sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(pdt.page_vaddrs, h_va, (size_t)np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete[] h_va;
        }
#endif


        pc_buff = createBuffer(sizeof(page_cache_d_t), cudaDevice);
        d_pc_ptr = (page_cache_d_t*)pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";

    }

    ~page_cache_t() {
        delete h_ranges;
        delete h_ranges_page_starts;
        delete h_ranges_dists;
    }





};



template <typename T>
struct range_d_t {
    uint64_t index_start;
    uint64_t count;
    uint64_t range_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_count;
    size_t n_elems_per_page;
    data_dist_t dist;
    uint8_t* src;

    simt::atomic<uint64_t, simt::thread_scope_device> access_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> read_io_cnt;


    pages_t pages;
    //padded_struct_pc* page_addresses;
    //uint32_t* page_addresses;
    //padded_struct_pc* page_vals;  //len = num of pages for data
    //void* self_ptr;
    page_cache_d_t cache;
    //range_d_t(range_t<T>* rt);
    __forceinline__
    __device__
    uint64_t get_backing_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_backing_ctrl(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_sector_size() const;
    __forceinline__
    __device__
    uint64_t get_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_subindex(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_global_address(const size_t page) const;
    __forceinline__
    __device__
    void release_page(const size_t pg) const;
    __forceinline__
    __device__
    void release_page(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    uint64_t acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue) ;
    __forceinline__
    __device__
    void write_done(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    T operator[](const size_t i) ;
    __forceinline__
    __device__
    void operator()(const size_t i, const T val);
    __forceinline__
    __device__
    cache_page_t* get_cache_page(const size_t pg) const;
    __forceinline__
    __device__
    uint64_t get_cache_page_addr(const uint32_t page_trans) const;
    __forceinline__
    __device__
    void mark_page_dirty(const size_t index);
};

template <typename T>
struct range_t {
    range_d_t<T> rdt;

    range_d_t<T>* d_range_ptr;
    page_cache_d_t* cache;

    BufferPtr pages_buff;
    //BufferPtr page_addresses_buff;

    BufferPtr range_buff;



    range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE);



};

template <typename T>
range_t<T>::range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist) {
    rdt.access_cnt = 0;
    rdt.miss_cnt = 0;
    rdt.hit_cnt = 0;
    rdt.read_io_cnt = 0;
    rdt.index_start = is;
    rdt.count = count;
    //range_id = (c_h->range_count)++;
    rdt.page_start = ps;
    rdt.page_count = pc;
    rdt.page_size = c_h->pdt.page_size;
    rdt.page_start_offset = pso;
    rdt.dist = dist;
    size_t s = pc;//(rdt.page_end-rdt.page_start);//*page_size / c_h->page_size;
    rdt.n_elems_per_page = rdt.page_size / sizeof(T);
    cache = (page_cache_d_t*) c_h->d_pc_ptr;
    pages_buff = createBuffer(s * sizeof(data_page_t), cudaDevice);
    rdt.pages = (pages_t) pages_buff.get();
    //std::vector<padded_struct_pc> ts(s, INVALID);
    data_page_t* ts = new data_page_t[s];
    for (size_t i = 0; i < s; i++) {
        ts[i].state = INVALID;
    }
    ////printf("S value: %llu\n", (unsigned long long)s);
    cuda_err_chk(cudaMemcpy(rdt.pages//_states
                            , ts, s * sizeof(data_page_t), cudaMemcpyHostToDevice));
    delete ts;

    //page_addresses_buff = createBuffer(s * sizeof(uint32_t), cudaDevice);
    //rdt.page_addresses = (uint32_t*) page_addresses_buff.get();
    //page_addresses_buff = createBuffer(s * sizeof(padded_struct_pc), cudaDevice);
    //page_addresses = (padded_struct_pc*) page_addresses_buff.get();

    range_buff = createBuffer(sizeof(range_d_t<T>), cudaDevice);
    d_range_ptr = (range_d_t<T>*)range_buff.get();
    //rdt.range_id  = c_h->pdt.n_ranges++;


    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

    c_h->add_range(this);

    rdt.cache = c_h->pdt;
    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_page(const size_t page_offset) const {
    return get_backing_page_(page_start, page_offset, cache.n_ctrls, dist);
}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_ctrl(const size_t page_offset) const {
    return get_backing_ctrl_(page_offset, cache.n_ctrls, dist);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_sector_size() const {
    return page_size;
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_page(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) >> (cache.page_size_log);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_subindex(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) & (cache.page_size_minus_1);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_global_address(const size_t page) const {
    return ((page << cache.n_ranges_bits) | range_id);
}
template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg) const {
    uint64_t index = pg;
    pages[index].state.fetch_sub(1, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg, const uint32_t count) const {
    uint64_t index = pg;
    pages[index].state.fetch_sub(count, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
cache_page_t* range_d_t<T>::get_cache_page(const size_t pg) const {
    uint32_t page_trans = pages[pg].offset;
    return cache.get_cache_page(page_trans);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_cache_page_addr(const uint32_t page_trans) const {
#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
    return cache.page_vaddrs[page_trans];
#else
    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));
#endif
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::mark_page_dirty(const size_t index) {
    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
}

// Include io_scheduler here so acquire_page can use io_sched_read.
#ifdef IO_SCHED_EVICT
#include <io_scheduler.cuh>
#endif

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue) {
    uint64_t index = pg;
    //uint32_t global_address = (index << cache.n_ranges_bits) | range_id;
    //access_cnt.fetch_add(count, simt::memory_order_relaxed);
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    unsigned int ns = 8;
    //bool miss = false;
    //T ret;
    uint64_t read_state,st,st_new;
    read_state = pages[index].state.fetch_add(count, simt::memory_order_acquire);
    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;

        switch (st) {
            //invalid
        case NV_NB:
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);
            if ((st_new & BUSY) == 0) {
                uint64_t st_new_st = (st_new >> (CNT_SHIFT+1)) & 0x03;
                if (st_new_st == NV_NB) { 
                    uint32_t page_trans = cache.find_slot(index, range_id, queue);
                    //fill in
                    //uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                    //uint32_t sm_id = get_smid();
                    //uint32_t ctrl = (tid/32) % (cache.n_ctrls);
                    //uint32_t ctrl = sm_id % (cache.n_ctrls);
                    //uint32_t ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                    uint64_t ctrl = get_backing_ctrl(index);
                    if (cache.read_ctrl_override != ALL_CTRLS) {
                        ctrl = cache.read_ctrl_override % cache.n_ctrls;
                    }
                    else if (ctrl == ALL_CTRLS) {
                        ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                    }
                    //ctrl = ctrl_;
                    uint64_t b_page = get_backing_page(index);
                    Controller* c = cache.d_ctrls[ctrl];
                    c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                    //uint32_t queue = (tid/32) % (c->n_qps);
                    //uint32_t queue = c->queue_counter.fetch_add(1, simt::memory_order_relaxed) % (c->n_qps);
                    //uint32_t queue = ((sm_id * 64) + warp_id()) % (c->n_qps);
                    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
#if 0 // Victim buffer read-hit scan disabled (O(n) too costly; safe when regions are disjoint)
#if defined(IO_SCHED_EVICT) && IO_SCHED_VICTIM_BUFFER
                    // ── Victim buffer read-hit check ──
                    if (cache.victim != nullptr) {
                        victim_buffer_t* vb = cache.victim;
                        uint64_t target_bo = b_page * cache.page_size;
                        bool vb_hit = false;
                        for (uint32_t vi = 0; vi < vb->n_entries; vi++) {
                            if (vb->byte_offsets[vi] == target_bo) {
                                vb->reader_counts[vi].fetch_add(1, simt::memory_order_relaxed);
                                __threadfence();
                                uint32_t vst = vb->states[vi].load(simt::memory_order_acquire);
                                if (vst == VICTIM_OCCUPIED && vb->byte_offsets[vi] == target_bo) {
#if IO_SCHED_VICTIM_SWAP
                                    memcpy((uint8_t*)cache.page_vaddrs[page_trans],
                                           (uint8_t*)vb->entry_vaddrs[vi],
                                           vb->page_size);
#else
                                    memcpy(cache.base_addr + (uint64_t)page_trans * cache.page_size,
                                           vb->data + (uint64_t)vi * vb->page_size,
                                           vb->page_size);
#endif
                                    vb_hit = true;
                                }
                                vb->reader_counts[vi].fetch_sub(1, simt::memory_order_relaxed);
                                // Cooperative free
                                if (vb->byte_offsets[vi] == UINT64_MAX &&
                                    vb->reader_counts[vi].load(simt::memory_order_relaxed) == 0) {
                                    uint32_t exp_v = VICTIM_OCCUPIED;
                                    vb->states[vi].compare_exchange_strong(exp_v, VICTIM_FREE,
                                        simt::memory_order_release, simt::memory_order_relaxed);
                                }
                                if (vb_hit) break;
                            }
                        }
                        if (!vb_hit) {
#ifdef IO_SCHED_EVICT
                            if (cache.read_queue != nullptr) {
                                uint64_t byte_offset_r = b_page * cache.page_size;
                                io_sched_read(cache.read_queue, byte_offset_r, cache.page_size, page_trans);
                            } else
#endif
                            {
                                read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                            }
                        }
                    } else
#endif // IO_SCHED_EVICT && IO_SCHED_VICTIM_BUFFER
#endif // disabled read-hit scan
                    {
#ifdef IO_SCHED_EVICT
                    if (cache.read_queue != nullptr) {
                        uint64_t byte_offset = b_page * cache.page_size;
                        io_sched_read(cache.read_queue, byte_offset, cache.page_size, page_trans);
                    } else
#endif
                    {
                        read_data_no_second(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                    }
                    }
                    //page_addresses[index].store(page_trans, simt::memory_order_release);
                    pages[index].offset = page_trans;
                    // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    //miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                    miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                    //new_state = VALID;
                    if (write)
                        pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                    //new_state |= DIRTY;
                    //pages[index].state.fetch_or(new_state, simt::memory_order_relaxed);
                    pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                    return page_trans;
    
                    fail = false;
                } else {
                    // Another thread beat us to load the page in cache, clear the BUSY bit and retry.
                    pages[index].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                    printf("Race loading page\n");
                }
            }

            break;
            //valid
        case V_NB:
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
            //uint32_t page_trans = pages[index].offset.load(simt::memory_order_acquire);
            uint32_t page_trans = pages[index].offset;
            // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
            //     __nanosleep(100);
            //hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            return page_trans;

            //pages[index].fetch_sub(1, simt::memory_order_release);
            fail = false;

            break;
        case NV_B:
        case V_B:
        default:
            break;



         
        }
        if (fail) {
            //if ((++j % 1000000) == 0)
            //    printf("failed to acquire_page: j: %llu\tcnt_shift+1: %llu\tpage: %llu\tread_state: %llx\tst: %llx\tst_new: %llx\n", (unsigned long long)j, (unsigned long long) (CNT_SHIFT+1), (unsigned long long) index, (unsigned long long)read_state, (unsigned long long)st, (unsigned long long)st_new);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}




template<typename T>
struct array_d_t {
    uint64_t n_elems;
    uint64_t start_offset;
    uint64_t n_ranges;
    uint8_t *src;

    range_d_t<T>* d_ranges;

    __forceinline__
    __device__
    void get_page_gid(const uint64_t i, range_d_t<T>*& r_, size_t& pg, size_t& gid) const {
        int64_t r = find_range(i);
        r_ = d_ranges+r;

        if (r != -1) {
            r_ = d_ranges+r;
            pg = r_->get_page(i);
            gid = r_->get_global_address(pg);
        }
        else {
            r_ = nullptr;
            printf("here\n");
        }
    }
    __forceinline__
    __device__
    void memcpy(const uint64_t i, const uint64_t count, T* dest) {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = 0xffffffff;
#endif
            uint32_t leader = 0;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            //uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = 32;
            uint32_t eq_mask = mask;
            int master = 0;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = 1;
            if (master == lane) {
                //std::pair<uint64_t, bool> base_memcpyflag;
                base = r_->acquire_page(page, count, false, ctrl, queue);
                base_master = base;
//                //printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            //
            ulonglong4* src_ = (ulonglong4*) r_->get_cache_page_addr(base_master);
            ulonglong4* dst_ = (ulonglong4*) dest;
            warp_memcpy<ulonglong4>(dst_, src_, 512/32);

            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }

    }
    __forceinline__
    __device__
    int64_t find_range(const size_t i) const {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++) {
            if ((d_ranges[k].index_start <= i) && (d_ranges[k].count > i)) {
                range = k;
                break;
            }

        }
        return range;
    }
    __forceinline__
    __device__
    void coalesce_page(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const uint64_t gaddr, const bool write,
                       uint32_t& eq_mask, int& master, uint32_t& count, uint64_t& base_master) const {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        auto r_ = d_ranges+r;
        if (lane == leader) {
            page_cache_d_t* pc = &(r_->cache);
            ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[0]->n_qps);
        }

        ctrl = 0; //__shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);


        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        //bool memcpyflag_master;
        //bool memcpyflag;
        count = __popc(eq_mask);
        if (master == lane) {
            //std::pair<uint64_t, bool> base_memcpyflag;
            base = r_->acquire_page(page, count, dirty, ctrl, queue);
            base_master = base;
//                //printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
        }
        base_master = __shfl_sync(eq_mask,  base_master, master);
    }

    __forceinline__
    __device__
    returned_cache_page_t<T> get_raw(const size_t i) const {
        returned_cache_page_t<T> ret;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);



            ret.addr = (T*) r_->get_cache_page_addr(base_master);
            ret.size = r_->get_sector_size()/sizeof(T);
            ret.offset = subindex/sizeof(T);
            //ret.page = page;
            __syncwarp(mask);


        }
        return ret;
    }
    __forceinline__
    __device__
    void release_raw(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    void* acquire_page_(const size_t i, data_page_t*& page_, size_t& start, size_t& end, range_d_t<T>* r_, const size_t page) const {
        //uint32_t lane = lane_id();



        void* ret = nullptr;
        page_ = nullptr;
        if (r_) {
            //uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            page_cache_d_t* pc = &(r_->cache);
            uint32_t ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            uint32_t queue = get_smid() % (pc->d_ctrls[0]->n_qps);
            uint64_t base_master = r_->acquire_page(page, 1, false, ctrl, queue);
            //coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;

        }
        return ret;
    }
    __forceinline__
    __device__
    void* acquire_page(const size_t i, data_page_t*& page_, size_t& start, size_t& end, int64_t& r) const {
        uint32_t lane = lane_id();
        r = find_range(i);
        auto r_ = d_ranges+r;

        void* ret = nullptr;
        page_ = nullptr;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);
            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;
            __syncwarp(mask);
        }
        return ret;
    }

    __forceinline__
    __device__
    void release_page(data_page_t* page_, const int64_t r, const size_t i) const {
        uint32_t lane = lane_id();
        auto r_ = d_ranges+r;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    T seq_read(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T ret;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0];
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
        return ret;
    }
    __forceinline__
    __device__
    void seq_write(const size_t i, const T val) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, true, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0] = val;
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
    }
    __forceinline__
    __device__
    T operator[](size_t i) const {
        return seq_read(i);
        // size_t k = 0;
        // bool found = false;
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }

        // }
        // if (found)
        //     return (((d_ranges[k]))[i-d_ranges[k].index_start]);
    }
    __forceinline__
    __device__
    void operator()(size_t i, T val) const {
        seq_write(i, val);
        // size_t k = 0;
        // bool found = false;
        // uint32_t mask = __activemask();
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }
        // }
        // __syncwarp(mask);
        // if (found)
        //     ((d_ranges[k]))(i-d_ranges[k].index_start, val);
    }


    __forceinline__
    __device__
    T AtomicAdd(const size_t i, const T val) const {
        //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        T old_val = 0;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t leader = __ffs(mask) - 1;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);


            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane) {
                base = r_->acquire_page(page, count, true, ctrl, queue);
                base_master = base;
                //    //printf("++tid: %llu\tbase: %llu  memcpyflag_master:%llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master, (unsigned long long) memcpyflag_master);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            // ((T*)(base_master+subindex))[0] = val;
            old_val = atomicAdd((T*)(r_->get_cache_page_addr(base_master)+subindex), val);
            // //printf("AtomicAdd: tid: %llu\tpage: %llu\tsubindex: %llu\tval: %llu\told_val: %llu\tbase_master: %llx\n",
            //        (unsigned long long) tid, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) val,
            //     (unsigned long long) old_val, (unsigned long long) base_master);
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);
        }

        return old_val;
    }




};

template<typename T>
struct array_t {
    array_d_t<T> adt;

    //range_t<T>** d_ranges;
    array_d_t<T>* d_array_ptr;



    BufferPtr d_array_buff;
    BufferPtr d_ranges_buff;
    BufferPtr d_d_ranges_buff;

    void print_reset_stats(void) {
        std::vector<range_d_t<T>> rdt(adt.n_ranges);
        //range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
        cuda_err_chk(cudaMemcpy(rdt.data(), adt.d_ranges, adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < adt.n_ranges; i++) {

            std::cout << std::dec << "#READ IOs: "  << rdt[i].read_io_cnt 
                                  << "\t#Accesses:" << rdt[i].access_cnt
                                  << "\t#Misses:"   << rdt[i].miss_cnt 
                                  << "\tMiss Rate:" << ((float)rdt[i].miss_cnt/rdt[i].access_cnt)
                                  << "\t#Hits: "    << rdt[i].hit_cnt 
                                  << "\tHit Rate:"  << ((float)rdt[i].hit_cnt/rdt[i].access_cnt) 
                                  << "\tCLSize:"    << rdt[i].page_size 
                                  << std::endl;
            std::cout << "*********************************" << std::endl;
            rdt[i].read_io_cnt = 0;
            rdt[i].access_cnt = 0;
            rdt[i].miss_cnt = 0;
            rdt[i].hit_cnt = 0;
        }
        cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt.data(), adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    }

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, uint32_t cudaDevice) {
        adt.n_elems = num_elems;
        adt.start_offset = disk_start_offset;

        adt.n_ranges = ranges.size();
        d_array_buff = createBuffer(sizeof(array_d_t<T>), cudaDevice);
        d_array_ptr = (array_d_t<T>*) d_array_buff.get();

        //d_ranges_buff = createBuffer(n_ranges * sizeof(range_t<T>*), cudaDevice);
        d_d_ranges_buff = createBuffer(adt.n_ranges * sizeof(range_d_t<T>), cudaDevice);
        adt.d_ranges = (range_d_t<T>*)d_d_ranges_buff.get();
        //d_ranges = (range_t<T>**) d_ranges_buff.get();
        for (size_t k = 0; k < adt.n_ranges; k++) {
            //cuda_err_chk(cudaMemcpy(d_ranges+k, &(ranges[k]->d_range_ptr), sizeof(range_t<T>*), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(adt.d_ranges+k, (ranges[k]->d_range_ptr), sizeof(range_d_t<T>), cudaMemcpyDeviceToDevice));
        }

        cuda_err_chk(cudaMemcpy(d_array_ptr, &adt, sizeof(array_d_t<T>), cudaMemcpyHostToDevice));
    }

};

__forceinline__
__device__
cache_page_t* page_cache_d_t::get_cache_page(const uint32_t page) const {
    return &this->cache_pages[page];
}



// (io_scheduler.cuh now included earlier, before acquire_page)

__forceinline__
__device__
uint32_t page_cache_d_t::find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_) {
    bool fail = true;
    uint64_t count = 0;
    uint64_t global_address =(uint64_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;

    do {

//	if (++count %100000 == 0)
//		printf("here\tc: %llu\n", (unsigned long long) count);

        //if (count < this->n_pages)
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
        //page = page_ticket->fetch_add(1, simt::memory_order_relaxed);
        //if (page < (n_cachelines_for_states*STATES_PER_CACHELINE)) {
        //    page = (page/n_cachelines_for_states) + ((page%n_cachelines_for_states)*STATES_PER_CACHELINE);
        //}
        //uint64_t unlocked = UNLOCKED;

        // uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        ////printf("tid: %llu page: %llu\n", tid, page);

        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);
        //this->page_take_lock[page].compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
        //not assigned to anyone yet
        if ( v == FREE ) {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                this->cache_pages[page].page_translation = global_address;
                //this->cache_pages[page].range_id = range_id;
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {

            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                //uint32_t previous_address = this->cache_pages[page].page_translation;
                uint64_t previous_global_address = this->cache_pages[page].page_translation;
                //uint8_t previous_range = this->cache_pages[page].range_id;
                uint64_t previous_range = previous_global_address & n_ranges_mask;
                uint64_t previous_address = previous_global_address >> n_ranges_bits;
                //uint32_t new_state = BUSY;
                //if ((previous_range >= range_cap) || (previous_address >= n_pages))
                //    //printf("prev_ga: %llu\tprev_range: %llu\tprev_add: %llu\trange_cap: %llu\tn_pages: %llu\n", (unsigned long long) previous_global_address, (unsigned long long) previous_range, (unsigned long long) previous_address,
                //           (unsigned long long) range_cap, (unsigned long long) n_pages);
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
                if ((cnt == 0) && (b == 0) ) {
                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                        //while ((new_expected_state & CNT_MASK ) != 0) new_expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_acquire);
                        if (((new_expected_state & CNT_MASK ) == 0) ) {
                            if ((new_expected_state & DIRTY)) {
                                uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);
                                if (this->evict_ctrl_override != ALL_CTRLS)
                                    ctrl = this->evict_ctrl_override % n_ctrls;
                                //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
                                uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
#ifdef IO_SCHED_EVICT
#if IO_SCHED_VICTIM_BUFFER
                                // ── Victim-only path (no scheduler): swap to victim, no write-back ──
                                if (this->victim != nullptr && this->evict_queue == nullptr) {
                                    victim_buffer_t* vb = this->victim;
                                    bool got_victim = false;
                                    uint32_t v_idx = 0;
                                    // Try claim a victim entry (1 attempt, round-robin)
                                    v_idx = vb->alloc_ticket.fetch_add(1, simt::memory_order_relaxed)
                                            & vb->n_entries_mask;
                                    uint32_t expected_v = VICTIM_FREE;
                                    if (vb->states[v_idx].compare_exchange_strong(
                                            expected_v, VICTIM_COPYING,
                                            simt::memory_order_acquire, simt::memory_order_relaxed)) {
                                        got_victim = true;
                                    }
                                    if (got_victim) {
#if IO_SCHED_VICTIM_SWAP
                                        // Pointer swap
                                        {
                                            uint64_t tmp_va = this->page_vaddrs[page];
                                            this->page_vaddrs[page] = vb->entry_vaddrs[v_idx];
                                            vb->entry_vaddrs[v_idx] = tmp_va;
                                            uint32_t ppe = this->pages_per_entry;
                                            for (uint32_t j = 0; j < ppe; j++) {
                                                uint64_t tmp_io = this->page_ioaddrs[(uint64_t)page * ppe + j];
                                                this->page_ioaddrs[(uint64_t)page * ppe + j] = vb->data_ioaddrs[(uint64_t)v_idx * ppe + j];
                                                vb->data_ioaddrs[(uint64_t)v_idx * ppe + j] = tmp_io;
                                            }
                                            this->prp1[page] = this->page_ioaddrs[(uint64_t)page * ppe];
                                            if (ppe >= 2 && this->prp2 != nullptr)
                                                this->prp2[page] = this->page_ioaddrs[(uint64_t)page * ppe + 1];
                                        }
#else
                                        memcpy(vb->data + (uint64_t)v_idx * vb->page_size,
                                               this->base_addr + (uint64_t)page * this->page_size,
                                               vb->page_size);
#endif
                                        uint64_t byte_offset = index * this->page_size;
                                        vb->byte_offsets[v_idx] = byte_offset;
                                        __threadfence();
                                        vb->states[v_idx].store(VICTIM_OCCUPIED, simt::memory_order_release);
                                        vb->prof_fast_count.fetch_add(1, simt::memory_order_relaxed);
                                    } else {
                                        // Victim full, no scheduler → sync write_data fallback
                                        vb->prof_fallback_count.fetch_add(1, simt::memory_order_relaxed);
                                        Controller* c = this->d_ctrls[ctrl == ALL_CTRLS ? 0 : ctrl];
                                        uint32_t queue = queue_ % (c->n_qps);
                                        uint64_t t0 = clock64();
                                        write_data(this, (c->d_qps)+queue, index, this->n_blocks_per_page, page);
                                        uint64_t t1 = clock64();
                                        this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                        this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                    }
                                } else
#endif
                                if (this->evict_queue != nullptr) {
                                    uint64_t byte_offset = index * this->page_size;
#if IO_SCHED_VICTIM_BUFFER
                                    // ── Victim buffer fast path ──
                                    if (this->victim != nullptr) {
                                        victim_buffer_t* vb = this->victim;
                                        bool got_victim = false;
                                        uint32_t v_idx = 0;
                                        uint64_t claim_t0 = clock64();
                                        for (uint32_t attempt = 0; attempt < 1; attempt++) {
                                            v_idx = vb->alloc_ticket.fetch_add(1, simt::memory_order_relaxed)
                                                    & vb->n_entries_mask;
                                            uint32_t expected_v = VICTIM_FREE;
                                            if (vb->states[v_idx].compare_exchange_strong(
                                                    expected_v, VICTIM_COPYING,
                                                    simt::memory_order_acquire, simt::memory_order_relaxed)) {
                                                got_victim = true;
                                                break;
                                            }
                                        }
                                        uint64_t claim_t1 = clock64();
                                        vb->prof_claim_cycles.fetch_add(claim_t1 - claim_t0, simt::memory_order_relaxed);
                                        if (got_victim) {
                                            uint64_t mc_t0 = clock64();
#if IO_SCHED_VICTIM_SWAP
                                            // ── Pointer swap: exchange buffer ownership ──
                                            {
                                                // Swap virtual addresses
                                                uint64_t tmp_va = this->page_vaddrs[page];
                                                this->page_vaddrs[page] = vb->entry_vaddrs[v_idx];
                                                vb->entry_vaddrs[v_idx] = tmp_va;

                                                // Swap IO addresses (ppe sub-pages each)
                                                uint32_t ppe = this->pages_per_entry;
                                                for (uint32_t j = 0; j < ppe; j++) {
                                                    uint64_t tmp_io = this->page_ioaddrs[(uint64_t)page * ppe + j];
                                                    this->page_ioaddrs[(uint64_t)page * ppe + j] = vb->data_ioaddrs[(uint64_t)v_idx * ppe + j];
                                                    vb->data_ioaddrs[(uint64_t)v_idx * ppe + j] = tmp_io;
                                                }

                                                // Update prp1/prp2 to match swapped page_ioaddrs
                                                this->prp1[page] = this->page_ioaddrs[(uint64_t)page * ppe];
                                                if (ppe >= 2 && this->prp2 != nullptr)
                                                    this->prp2[page] = this->page_ioaddrs[(uint64_t)page * ppe + 1];
                                            }
#else
                                            memcpy(vb->data + (uint64_t)v_idx * vb->page_size,
                                                   this->base_addr + (uint64_t)page * this->page_size,
                                                   vb->page_size);
#endif
                                            vb->byte_offsets[v_idx] = byte_offset;
                                            __threadfence();
                                            vb->states[v_idx].store(VICTIM_OCCUPIED, simt::memory_order_release);
                                            uint64_t mc_t1 = clock64();
                                            vb->prof_memcpy_cycles.fetch_add(mc_t1 - mc_t0, simt::memory_order_relaxed);
                                            // Double-eviction invalidation disabled for now (O(n) scan too costly).
                                            // Safe when dirty and burst regions are disjoint.
                                            // for (uint32_t vi = 0; vi < vb->n_entries; vi++) {
                                            //     if (vi != v_idx && vb->byte_offsets[vi] == byte_offset)
                                            //         vb->byte_offsets[vi] = UINT64_MAX;
                                            // }
                                            uint64_t t0 = clock64();
                                            io_sched_enqueue_forget(this->evict_queue, byte_offset,
                                                                     this->page_size,
                                                                     VICTIM_PC_ENTRY_BIT | (uint64_t)v_idx);
                                            uint64_t t1 = clock64();
                                            vb->prof_enqueue_cycles.fetch_add(t1 - t0, simt::memory_order_relaxed);
                                            vb->prof_fast_count.fetch_add(1, simt::memory_order_relaxed);
                                            this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                            this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                        } else {
                                            // Victim buffer full — fallback to blocking write
                                            vb->prof_fallback_count.fetch_add(1, simt::memory_order_relaxed);
                                            uint64_t t0 = clock64();
                                            io_sched_write(this->evict_queue, byte_offset, this->page_size, page);
                                            uint64_t t1 = clock64();
                                            this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                            this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                        }
                                    } else
#endif // IO_SCHED_VICTIM_BUFFER
                                    {
                                        uint64_t t0 = clock64();
                                        io_sched_write(this->evict_queue, byte_offset, this->page_size, page);
                                        uint64_t t1 = clock64();
                                        this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                        this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                    }
                                } else
#endif
                                if (ctrl == ALL_CTRLS) {
                                    for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                        Controller* c = this->d_ctrls[ctrl];
                                        uint32_t queue = queue_ % (c->n_qps);
                                        if (this->staging_buffer != nullptr) {
                                            memcpy(this->staging_buffer + (uint64_t)page * this->page_size,
#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
                                                   (uint8_t*)this->page_vaddrs[page],
#else
                                                   this->base_addr + (uint64_t)page * this->page_size,
#endif
                                                   this->page_size);
                                        } else if (this->fake_evict_delay_cycles > 0) {
                                            uint64_t start = clock64();
                                            while ((uint64_t)(clock64() - start) < this->fake_evict_delay_cycles) { }
                                        } else {
                                            uint64_t t0 = clock64();
                                            write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                            uint64_t t1 = clock64();
                                            this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                            this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                        }
                                    }
                                }
                                else {

                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);

                                    //index = ranges_page_starts[previous_range] + previous_address;

                                    if (this->staging_buffer != nullptr) {
                                        memcpy(this->staging_buffer + (uint64_t)page * this->page_size,
#if IO_SCHED_VICTIM_BUFFER && IO_SCHED_VICTIM_SWAP
                                               (uint8_t*)this->page_vaddrs[page],
#else
                                               this->base_addr + (uint64_t)page * this->page_size,
#endif
                                               this->page_size);
                                    } else if (this->fake_evict_delay_cycles > 0) {
                                        uint64_t start = clock64();
                                        while ((uint64_t)(clock64() - start) < this->fake_evict_delay_cycles) { }
                                    } else {
                                        uint64_t t0 = clock64();
                                        write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                        uint64_t t1 = clock64();
                                        this->write_cycles_total->fetch_add(t1 - t0, simt::memory_order_relaxed);
                                        this->write_call_count->fetch_add(1, simt::memory_order_relaxed);
                                    }
                                }
                            }

                            fail = false;
                            this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                        }
                        else { 
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
//if ((j % 1000000) == 0) {
//                printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//}
                        }
                    }
                }

                //this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                if (!fail) {
                    //this->cache_pages[page].page_translation = address;
                    //this->cache_pages[page].range_id = range_id;
//                    this->page_translation[page] = global_address;
                    this->cache_pages[page].page_translation = global_address;
                }
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }


        }

        count++;
/*if (fail) {
  if ((++j % 1000000) == 0) {
  printf("failed to find slot j: %llu\n", (unsigned long long) j);
  }
  }*/
        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
//             __nanosleep(ns);
//             if (ns < 256) {
//                 ns *= 2;
//             }
#endif
            //   if ((j % 10000000) == 0) {
            //     printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//            }
//	   expected_state = 0;
//	   new_expected_state = 0;


        }

    } while(fail);
    return page;

}


inline __device__ void poll_async(QueuePair* qp, uint16_t cid, uint16_t sq_pos) {
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    //sq_dequeue(&qp->sq, sq_pos);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq);



    put_cid(&qp->sq, cid);
}

inline __device__ void access_data_async(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode, uint16_t * cid, uint16_t* sq_pos) {
    nvm_cmd_t cmd;
    *cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    *sq_pos = sq_enqueue(&qp->sq, &cmd);



}

inline __device__ void enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
    nvm_cmd_rw_blks(cmd, starting_lba, 1);
    unsigned int ns = 8;
    do {
        //check if new head past pc_pos
        //cur_pc_head == new head
        //prev_pc_head == old head
        //pc_pos == position i wanna move the head past
        uint64_t cur_pc_head = pc->q_head->load(simt::memory_order_relaxed);
        //sec == true when cur_pc_head past pc_pos
        bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
            ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
            ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

        if (sec) break;

        //if not
        uint64_t qlv = pc->q_lock->load(simt::memory_order_relaxed);
        //got lock
        if (qlv == 0) {
            qlv = pc->q_lock->fetch_or(1, simt::memory_order_acquire);
            if (qlv == 0) {
                uint64_t cur_pc_tail;// = pc->q_tail.load(simt::memory_order_acquire);

                uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, pc->q_tail, &cur_pc_tail);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

                pc->q_head->store(cur_pc_tail, simt::memory_order_release);
                pc->q_lock->store(0, simt::memory_order_release);
                pc->extra_reads->fetch_add(1, simt::memory_order_relaxed);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);



                break;
            }
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
    } while(true);

}

inline __device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);


    //enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head)
    enqueue_second(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);



    put_cid(&qp->sq, cid);


}

inline __device__ void enqueue_second_copilot(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head, volatile copilot_notify_entry* notify) {
    nvm_cmd_rw_blks(cmd, starting_lba, 1);
    unsigned int ns = 8;
    do {
        uint64_t cur_pc_head = pc->q_head->load(simt::memory_order_relaxed);
        bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
            ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
            ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

        if (sec) break;

        uint64_t qlv = pc->q_lock->load(simt::memory_order_relaxed);
        if (qlv == 0) {
            qlv = pc->q_lock->fetch_or(1, simt::memory_order_acquire);
            if (qlv == 0) {
                uint64_t cur_pc_tail;
                uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, pc->q_tail, &cur_pc_tail);

                // Poll notify instead of cq_poll
                unsigned int ns2 = 8;
                while (notify[cid].ready == 0) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                    __nanosleep(ns2);
                    if (ns2 < 256) ns2 *= 2;
#endif
                }
                uint32_t cq_pos = notify[cid].pos;
                uint32_t head = notify[cid].loc;
                uint32_t head_ = notify[cid].cq_head;
                notify[cid].ready = 0;

                pc->q_head->store(cur_pc_tail, simt::memory_order_release);
                pc->q_lock->store(0, simt::memory_order_release);
                pc->extra_reads->fetch_add(1, simt::memory_order_relaxed);

                qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

                break;
            }
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) ns *= 2;
#endif
    } while(true);
}

inline __device__ void read_data_copilot(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, volatile copilot_notify_entry* notify) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    // Poll notify instead of cq_poll
    unsigned int ns = 8;
    while (notify[cid].ready == 0) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }
    uint32_t cq_pos = notify[cid].pos;
    uint32_t head = notify[cid].loc;
    uint32_t head_ = notify[cid].cq_head;
    notify[cid].ready = 0;

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    uint64_t pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    uint64_t pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

    enqueue_second_copilot(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head, notify);

    put_cid(&qp->sq, cid);
}

inline __device__ void enqueue_second_profiled(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head, uint64_t* spin_out) {
    nvm_cmd_rw_blks(cmd, starting_lba, 1);
    unsigned int ns = 8;
    do {
        uint64_t cur_pc_head = pc->q_head->load(simt::memory_order_relaxed);
        bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
            ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
            ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

        if (sec) break;

        uint64_t qlv = pc->q_lock->load(simt::memory_order_relaxed);
        if (qlv == 0) {
            qlv = pc->q_lock->fetch_or(1, simt::memory_order_acquire);
            if (qlv == 0) {
                uint64_t cur_pc_tail;

                uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, pc->q_tail, &cur_pc_tail);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_, spin_out);

                pc->q_head->store(cur_pc_tail, simt::memory_order_release);
                pc->q_lock->store(0, simt::memory_order_release);
                pc->extra_reads->fetch_add(1, simt::memory_order_relaxed);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

                break;
            }
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
    } while(true);
}

struct io_profile_t {
    uint64_t sq_cycles;
    uint64_t cq_cycles;
    uint64_t sec_cycles;
    uint64_t cq_spins;
};

inline __device__ void read_data_profiled(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, io_profile_t* prof) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);

    uint64_t t0 = clock64();
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint64_t t1 = clock64();

    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;
    uint64_t spins = 0;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_, &spins);
    uint64_t t2 = clock64();

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

    uint64_t t3 = clock64();
    enqueue_second_profiled(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head, &spins);
    uint64_t t4 = clock64();

    put_cid(&qp->sq, cid);

    prof->sq_cycles += (t1 - t0);
    prof->cq_cycles += (t2 - t1);
    prof->sec_cycles += (t4 - t3);
    prof->cq_spins += spins;
}

inline __device__ void read_data_batch_warp32(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    // Warp-only fast path: assumes 32 lanes, single warp, exclusive queue ownership.
    const uint32_t lane = lane_id();
    const uint32_t mask = __activemask();

    uint32_t base_tail = 0;
    if (lane == 0) {
        base_tail = qp->sq.tail.load(simt::memory_order_relaxed);
    }
    base_tail = __shfl_sync(mask, base_tail, 0);

    uint16_t base_cid = 0;
    if (lane == 0) {
        base_cid = qp->sq.cid_ticket.fetch_add(32, simt::memory_order_relaxed) & 0xffff;
    }
    base_cid = __shfl_sync(mask, base_cid, 0);
    uint16_t cid = (base_cid + lane) & 0xffff;

    nvm_cmd_t cmd;
    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);

    const uint32_t pos = (base_tail + lane) & qp->sq.qs_minus_1;
    copy_type* queue_loc = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
    copy_type* cmd_ = ((copy_type*)(cmd.dword));
#pragma unroll
    for (uint32_t i = 0; i < 64/sizeof(copy_type); i++) {
        queue_loc[i] = cmd_[i];
    }

    __syncwarp(mask);
    if (lane == 0) {
        const uint32_t new_tail = base_tail + 32;
        const uint32_t new_db = new_tail & qp->sq.qs_minus_1;
        asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(qp->sq.db),"r"(new_db) : "memory");
        qp->sq.tail.store(new_tail, simt::memory_order_release);
    }
    __syncwarp(mask);

    if (lane == 0) {
        const uint32_t cur_head = qp->cq.head.load(simt::memory_order_relaxed);
        const uint32_t target = cur_head + 31;
        const uint32_t loc = target & qp->cq.qs_minus_1;
        const bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);
        unsigned int ns = 8;
        while (true) {
            const uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[loc].dword[3];
            const bool phase = (cpl_entry & 0x00010000) >> 16;
            if (phase == expected_phase) {
                break;
            }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
        }
        const uint32_t new_head = cur_head + 32;
        qp->cq.head.store(new_head, simt::memory_order_release);
        const uint32_t new_db = new_head & qp->cq.qs_minus_1;
        asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(qp->cq.db),"r"(new_db) : "memory");
        qp->sq.head.store(base_tail + 32, simt::memory_order_release);
    }
    __syncwarp(mask);
}

inline __device__ void read_data_batch_warp32_profile(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, uint64_t* pre_cycles, uint64_t* poll_cycles, uint64_t* total_cycles) {
    const uint32_t lane = lane_id();
    const uint32_t mask = __activemask();

    uint64_t t_start = 0;
    if (lane == 0) {
        t_start = clock64();
    }
    t_start = __shfl_sync(mask, t_start, 0);

    uint32_t base_tail = 0;
    if (lane == 0) {
        base_tail = qp->sq.tail.load(simt::memory_order_relaxed);
    }
    base_tail = __shfl_sync(mask, base_tail, 0);

    uint16_t base_cid = 0;
    if (lane == 0) {
        base_cid = qp->sq.cid_ticket.fetch_add(32, simt::memory_order_relaxed) & 0xffff;
    }
    base_cid = __shfl_sync(mask, base_cid, 0);
    uint16_t cid = (base_cid + lane) & 0xffff;

    nvm_cmd_t cmd;
    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);

    const uint32_t pos = (base_tail + lane) & qp->sq.qs_minus_1;
    copy_type* queue_loc = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
    copy_type* cmd_ = ((copy_type*)(cmd.dword));
#pragma unroll
    for (uint32_t i = 0; i < 64/sizeof(copy_type); i++) {
        queue_loc[i] = cmd_[i];
    }

    __syncwarp(mask);
    uint64_t t_before_db = 0;
    if (lane == 0) {
        t_before_db = clock64();
        const uint32_t new_tail = base_tail + 32;
        const uint32_t new_db = new_tail & qp->sq.qs_minus_1;
        asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(qp->sq.db),"r"(new_db) : "memory");
        qp->sq.tail.store(new_tail, simt::memory_order_release);
    }
    t_before_db = __shfl_sync(mask, t_before_db, 0);
    __syncwarp(mask);

    uint64_t t_poll_start = 0;
    if (lane == 0) {
        t_poll_start = clock64();
        const uint32_t cur_head = qp->cq.head.load(simt::memory_order_relaxed);
        const uint32_t target = cur_head + 31;
        const uint32_t loc = target & qp->cq.qs_minus_1;
        const bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);
        unsigned int ns = 8;
        while (true) {
            const uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[loc].dword[3];
            const bool phase = (cpl_entry & 0x00010000) >> 16;
            if (phase == expected_phase) {
                break;
            }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
        }
        const uint32_t new_head = cur_head + 32;
        qp->cq.head.store(new_head, simt::memory_order_release);
        const uint32_t new_db = new_head & qp->cq.qs_minus_1;
        asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(qp->cq.db),"r"(new_db) : "memory");
        qp->sq.head.store(base_tail + 32, simt::memory_order_release);
    }
    __syncwarp(mask);

    if (lane == 0) {
        const uint64_t t_end = clock64();
        if (pre_cycles) {
            *pre_cycles += (t_before_db - t_start);
        }
        if (poll_cycles) {
            *poll_cycles += (t_end - t_poll_start);
        }
        if (total_cycles) {
            *total_cycles += (t_end - t_start);
        }
    }
}

#define READ_DATA_BATCH_MAX 32

inline __device__ void read_data_batch(page_cache_d_t* pc, QueuePair* qp, const uint64_t* starting_lbas, const uint64_t* n_blocks_array, const unsigned long long* pc_entries, const uint32_t batch_size) {
    // Single-thread, exclusive-queue batch read API.
    // Submits batch_size NVMe read commands in one doorbell ring, polls CQ for
    // all completions, then releases resources.  No enqueue_second.

    // Step 1: Batch allocate CIDs
    uint16_t cids[READ_DATA_BATCH_MAX];
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(batch_size, simt::memory_order_relaxed) & 0xffff;
    for (uint32_t i = 0; i < batch_size; i++)
        cids[i] = (base_cid + i) & 0xffff;

    // Step 2: Build and write N commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    for (uint32_t i = 0; i < batch_size; i++) {
        nvm_cmd_t cmd;
        nvm_cmd_header(&cmd, cids[i], NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1 = pc->prp1[pc_entries[i]];
        uint64_t prp2 = 0;
        if (pc->prps)
            prp2 = pc->prp2[pc_entries[i]];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], n_blocks_array[i]);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];
    }

    // Step 3: Single doorbell to submit all commands
    uint32_t new_tail = base_tail + batch_size;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: CQ polling — wait for the last CQ entry (Scheme A)
    // NVMe spec guarantees CQ entries are written in position order, so if the
    // entry at position (cq_head + batch_size - 1) has the correct phase bit,
    // all preceding entries are also valid.
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + batch_size - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase)
            break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256)
            ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + batch_size;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");

    qp->sq.head.store(base_tail + batch_size, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < batch_size; i++)
        put_cid(&qp->sq, cids[i]);
}

inline __device__ void write_data_batch(page_cache_d_t* pc, QueuePair* qp, const uint64_t* starting_lbas, const uint64_t* n_blocks_array, const unsigned long long* pc_entries, const uint32_t batch_size) {
    // Single-thread, exclusive-queue batch write API.
    // Submits batch_size NVMe write commands in one doorbell ring, polls CQ for
    // all completions, then releases resources.  No enqueue_second.

    // Step 1: Batch allocate CIDs
    uint16_t cids[READ_DATA_BATCH_MAX];
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(batch_size, simt::memory_order_relaxed) & 0xffff;
    for (uint32_t i = 0; i < batch_size; i++)
        cids[i] = (base_cid + i) & 0xffff;

    // Step 2: Build and write N commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    for (uint32_t i = 0; i < batch_size; i++) {
        nvm_cmd_t cmd;
        nvm_cmd_header(&cmd, cids[i], NVM_IO_WRITE, qp->nvmNamespace);
        uint64_t prp1 = pc->prp1[pc_entries[i]];
        uint64_t prp2 = 0;
        if (pc->prps)
            prp2 = pc->prp2[pc_entries[i]];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], n_blocks_array[i]);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];
    }

    uint64_t total_blocks = 0;
    for (uint32_t i = 0; i < batch_size; i++)
        total_blocks += n_blocks_array[i];
    pc->write_io_cmd_cnt->fetch_add(batch_size, simt::memory_order_relaxed);
    pc->write_io_block_cnt->fetch_add(total_blocks, simt::memory_order_relaxed);

    // Step 3: Single doorbell to submit all commands
    uint32_t new_tail = base_tail + batch_size;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: CQ polling — wait for the last CQ entry
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + batch_size - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase)
            break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256)
            ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + batch_size;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");

    qp->sq.head.store(base_tail + batch_size, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < batch_size; i++)
        put_cid(&qp->sq, cids[i]);
}

// ═══════════════════════════════════════════════════════════════════
// Large batch IO API — no READ_DATA_BATCH_MAX (32) limitation.
//
// Parameters are passed via device global memory pointers, so
// batch_size can be any value < queue_depth (typically < 1024).
// n_blocks is uniform for all commands in the batch.
// Each queue must be used by exactly one thread (no sharing).
// ═══════════════════════════════════════════════════════════════════

inline __device__ void write_data_batch_large(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint64_t n_blocks,
    const unsigned long long* pc_entries,
    const uint32_t batch_size)
{
    // Step 1: Batch allocate CIDs
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(batch_size, simt::memory_order_relaxed) & 0xffff;

    // Step 2: Build and write N commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    for (uint32_t i = 0; i < batch_size; i++) {
        nvm_cmd_t cmd;
        uint16_t cid = (base_cid + i) & 0xffff;
        nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
        uint64_t prp1 = pc->prp1[pc_entries[i]];
        uint64_t prp2 = 0;
        if (pc->prps)
            prp2 = pc->prp2[pc_entries[i]];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], n_blocks);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];
    }

    uint64_t total_blocks = (uint64_t)n_blocks * batch_size;
    pc->write_io_cmd_cnt->fetch_add(batch_size, simt::memory_order_relaxed);
    pc->write_io_block_cnt->fetch_add(total_blocks, simt::memory_order_relaxed);

    // Step 3: Single doorbell to submit all commands
    uint32_t new_tail = base_tail + batch_size;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: CQ polling — wait for the last CQ entry
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + batch_size - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase)
            break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + batch_size;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");
    qp->sq.head.store(base_tail + batch_size, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < batch_size; i++)
        put_cid(&qp->sq, (base_cid + i) & 0xffff);
}

inline __device__ void read_data_batch_large(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint64_t n_blocks,
    const unsigned long long* pc_entries,
    const uint32_t batch_size)
{
    // Step 1: Batch allocate CIDs
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(batch_size, simt::memory_order_relaxed) & 0xffff;

    // Step 2: Build and write N commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    for (uint32_t i = 0; i < batch_size; i++) {
        nvm_cmd_t cmd;
        uint16_t cid = (base_cid + i) & 0xffff;
        nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1 = pc->prp1[pc_entries[i]];
        uint64_t prp2 = 0;
        if (pc->prps)
            prp2 = pc->prp2[pc_entries[i]];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], n_blocks);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];
    }

    // Step 3: Single doorbell to submit all commands
    uint32_t new_tail = base_tail + batch_size;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: CQ polling — wait for the last CQ entry
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + batch_size - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase)
            break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + batch_size;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");
    qp->sq.head.store(base_tail + batch_size, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < batch_size; i++)
        put_cid(&qp->sq, (base_cid + i) & 0xffff);
}


// ═══════════════════════════════════════════════════════════════════
// Scheduler batch IO API — per-IO LBA, block count, and PRP.
//
// No 32-entry limit (parameters via device/shared memory pointers).
// Batch size up to queue_depth - 1. Single doorbell, poll last CQ entry.
// Each queue must be used by exactly one thread (no sharing).
// ═══════════════════════════════════════════════════════════════════

inline __device__ void sched_batch_rw(
    page_cache_d_t* pc, QueuePair* qp,
    const uint8_t opcode,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const unsigned long long* pc_entries,
    const uint32_t count)
{
    if (count == 0) return;

    // Step 1: Batch allocate CIDs
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(count, simt::memory_order_relaxed) & 0xffff;

    // Step 2: Build all commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    uint64_t total_blocks = 0;
    for (uint32_t i = 0; i < count; i++) {
        nvm_cmd_t cmd;
        uint16_t cid = (base_cid + i) & 0xffff;
        nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
        uint64_t prp1 = pc->prp1[pc_entries[i]];
        uint64_t prp2 = 0;
        if (pc->prps) prp2 = pc->prp2[pc_entries[i]];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], (uint64_t)n_blocks_array[i]);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];

        total_blocks += n_blocks_array[i];
    }

    if (opcode == NVM_IO_WRITE) {
        pc->write_io_cmd_cnt->fetch_add(count, simt::memory_order_relaxed);
        pc->write_io_block_cnt->fetch_add(total_blocks, simt::memory_order_relaxed);
    }

    // Step 3: Single doorbell
    uint32_t new_tail = base_tail + count;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: Poll last CQ entry
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + count - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase) break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + count;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");
    qp->sq.head.store(base_tail + count, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < count; i++)
        put_cid(&qp->sq, (base_cid + i) & 0xffff);
}

inline __device__ void sched_write_batch(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const unsigned long long* pc_entries,
    const uint32_t count)
{
    sched_batch_rw(pc, qp, NVM_IO_WRITE, starting_lbas, n_blocks_array, pc_entries, count);
}

inline __device__ void sched_read_batch(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const unsigned long long* pc_entries,
    const uint32_t count)
{
    sched_batch_rw(pc, qp, NVM_IO_READ, starting_lbas, n_blocks_array, pc_entries, count);
}

// Batch submit with pre-built PRP arrays (for cross-entry merged IOs).
// prp1_array[i] and prp2_array[i] are pre-built PRP1/PRP2 for command i.
inline __device__ void sched_batch_merged_rw(
    page_cache_d_t* pc, QueuePair* qp,
    const uint8_t opcode,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const uint64_t* prp1_array,
    const uint64_t* prp2_array,
    const uint32_t count)
{
    if (count == 0) return;

    // Step 1: Batch allocate CIDs
    uint16_t base_cid = qp->sq.cid_ticket.fetch_add(count, simt::memory_order_relaxed) & 0xffff;

    // Step 2: Build all commands into SQ
    uint32_t base_tail = qp->sq.tail.load(simt::memory_order_relaxed);

    uint64_t total_blocks = 0;
    for (uint32_t i = 0; i < count; i++) {
        nvm_cmd_t cmd;
        uint16_t cid = (base_cid + i) & 0xffff;
        nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd, prp1_array[i], prp2_array[i]);
        nvm_cmd_rw_blks(&cmd, starting_lbas[i], (uint64_t)n_blocks_array[i]);

        uint32_t pos = (base_tail + i) & qp->sq.qs_minus_1;
        copy_type* dst = ((copy_type*)(((nvm_cmd_t*)(qp->sq.vaddr)) + pos));
        copy_type* src = ((copy_type*)(cmd.dword));
        for (uint32_t j = 0; j < 64 / sizeof(copy_type); j++)
            dst[j] = src[j];

        total_blocks += n_blocks_array[i];
    }

    if (opcode == NVM_IO_WRITE) {
        pc->write_io_cmd_cnt->fetch_add(count, simt::memory_order_relaxed);
        pc->write_io_block_cnt->fetch_add(total_blocks, simt::memory_order_relaxed);
    }

    // Step 3: Single doorbell
    uint32_t new_tail = base_tail + count;
    uint32_t new_db = new_tail & qp->sq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->sq.db), "r"(new_db) : "memory");
    qp->sq.tail.store(new_tail, simt::memory_order_release);

    // Step 4: Poll last CQ entry
    uint32_t cq_base_head = qp->cq.head.load(simt::memory_order_relaxed);
    uint32_t target = cq_base_head + count - 1;
    uint32_t target_loc = target & qp->cq.qs_minus_1;
    bool expected_phase = ((~(target >> qp->cq.qs_log2)) & 0x01);

    unsigned int ns = 8;
    while (true) {
        uint32_t cpl_entry = ((volatile nvm_cpl_t*)qp->cq.vaddr)[target_loc].dword[3];
        bool phase = (cpl_entry & 0x00010000) >> 16;
        if (phase == expected_phase) break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }

    // Step 5: Batch release CQ + SQ
    uint32_t new_cq_head = cq_base_head + count;
    qp->cq.head.store(new_cq_head, simt::memory_order_release);
    uint32_t cq_db = new_cq_head & qp->cq.qs_minus_1;
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(qp->cq.db), "r"(cq_db) : "memory");
    qp->sq.head.store(base_tail + count, simt::memory_order_release);

    // Step 6: Release CIDs
    for (uint32_t i = 0; i < count; i++)
        put_cid(&qp->sq, (base_cid + i) & 0xffff);
}

inline __device__ void sched_write_batch_merged(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const uint64_t* prp1_array,
    const uint64_t* prp2_array,
    const uint32_t count)
{
    sched_batch_merged_rw(pc, qp, NVM_IO_WRITE, starting_lbas, n_blocks_array, prp1_array, prp2_array, count);
}

inline __device__ void sched_read_batch_merged(
    page_cache_d_t* pc, QueuePair* qp,
    const uint64_t* starting_lbas,
    const uint32_t* n_blocks_array,
    const uint64_t* prp1_array,
    const uint64_t* prp2_array,
    const uint32_t count)
{
    sched_batch_merged_rw(pc, qp, NVM_IO_READ, starting_lbas, n_blocks_array, prp1_array, prp2_array, count);
}


inline __device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);

    pc->write_io_cmd_cnt->fetch_add(1, simt::memory_order_relaxed);
    pc->write_io_block_cnt->fetch_add(n_blocks, simt::memory_order_relaxed);

}

// Minimal synchronous read: no enqueue_second, no page cache bookkeeping.
inline __device__ void read_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps) prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    (void)sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

// Minimal synchronous write: no page cache bookkeeping, no IO counters.
inline __device__ void write_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps) prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    (void)sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

// Dispatch read or write based on opcode, no page cache bookkeeping.
inline __device__ void access_data_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    if (opcode == NVM_IO_WRITE)
        write_data_no_second(pc, qp, starting_lba, n_blocks, pc_entry);
    else
        read_data_no_second(pc, qp, starting_lba, n_blocks, pc_entry);
}

// ---- Merged PRP API: single NVMe command with pre-built PRP ----
// These take explicit prp1/prp2 instead of looking up from pc_entry.
// Used by the scheduler for cross-pc_entry merged IOs.

inline __device__ void write_data_merged_no_second(
    QueuePair* qp, uint64_t starting_lba, uint64_t n_blocks,
    uint64_t prp1, uint64_t prp2)
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    (void)sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

inline __device__ void read_data_merged_no_second(
    QueuePair* qp, uint64_t starting_lba, uint64_t n_blocks,
    uint64_t prp1, uint64_t prp2)
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    (void)sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

inline __device__ void access_data_merged_no_second(
    QueuePair* qp, uint64_t starting_lba, uint64_t n_blocks,
    uint64_t prp1, uint64_t prp2, uint8_t opcode)
{
    if (opcode == NVM_IO_WRITE)
        write_data_merged_no_second(qp, starting_lba, n_blocks, prp1, prp2);
    else
        read_data_merged_no_second(qp, starting_lba, n_blocks, prp1, prp2);
}

// ---- Async two-phase API (no_second variant) ----
// Phase 1: submit IO command, return CID handle for later polling.
inline __device__ uint16_t submit_io_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps) prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    (void)sq_enqueue(&qp->sq, &cmd);
    return cid;
}

inline __device__ uint16_t submit_read(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    return submit_io_no_second(pc, qp, starting_lba, n_blocks, pc_entry, NVM_IO_READ);
}

inline __device__ uint16_t submit_write_no_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    return submit_io_no_second(pc, qp, starting_lba, n_blocks, pc_entry, NVM_IO_WRITE);
}

// Phase 2: poll CQ for a previously submitted IO and release resources.
inline __device__ void poll_no_second(QueuePair* qp, uint16_t cid) {
    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

inline __device__ void poll_copilot(QueuePair* qp, uint16_t cid, volatile copilot_notify_entry* notify) {
    unsigned int ns = 8;
    while (notify[cid].ready == 0) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }
    notify[cid].ready = 0;

    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
}

inline __device__ void drain_copilot_cq(QueuePair* qp, copilot_cq_table* cq_table) {
    if (atomicCAS((uint32_t*)&cq_table->drain_lock, 0u, 1u) != 0)
        return;

    uint32_t head = cq_table->gpu_head;
    uint32_t qs_minus_1 = qp->cq.qs_minus_1;
    uint32_t count = 0;
    uint16_t last_sq_head = 0;

    while (true) {
        uint32_t pos = head & qs_minus_1;
        if (cq_table->ready[pos] == 0) break;
        __threadfence();
        last_sq_head = cq_table->entries[pos].sq_head;
        cq_table->ready[pos] = 0;
        head++;
        count++;
    }

    if (count > 0) {
        uint32_t cur_sq_head = qp->sq.head.load(simt::memory_order_relaxed);
        uint32_t sq_mask = qp->sq.qs_minus_1;
        uint32_t loc = cur_sq_head & sq_mask;
        uint32_t sq_move = 0;
        for (; loc != last_sq_head; sq_move++, loc = (loc + 1) & sq_mask)
            qp->sq.tickets[loc].val.fetch_add(1, simt::memory_order_relaxed);
        if (sq_move > 0)
            qp->sq.head.fetch_add(sq_move, simt::memory_order_acq_rel);

        cq_table->gpu_head = head;
        __threadfence();
        uint32_t new_db = head & qs_minus_1;
        asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                     :: "l"(qp->cq.db), "r"(new_db) : "memory");
    }

    __threadfence();
    atomicExch((uint32_t*)&cq_table->drain_lock, 0u);
}

inline __device__ void poll_copilot(QueuePair* qp, uint16_t cid,
    volatile copilot_notify_entry* notify, copilot_cq_table* cq_table) {
    unsigned int ns = 8;
    while (notify[cid].ready == 0) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }
    notify[cid].ready = 0;

    drain_copilot_cq(qp, cq_table);
    put_cid(&qp->sq, cid);
}

inline __device__ void access_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);


}

inline __device__ void trim_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba,
                                 const uint64_t n_blocks, const unsigned long long pc_entry) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    // Build DSM range list in the page cache entry.
    uint8_t* range_buf = pc->base_addr + (pc_entry * pc->page_size);
    nvm_dsm_range_t* range = reinterpret_cast<nvm_dsm_range_t*>(range_buf);
    range->slba = starting_lba;
    range->nlb = (n_blocks - 1);
    range->attr = 0;

    nvm_cmd_header(&cmd, cid, NVM_IO_DATASET_MGMT, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps) {
        prp2 = pc->prp2[pc_entry];
    }
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_dsm(&cmd, 0, NVM_DSM_ATTR_DEALLOCATE);

    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    (void)sq_pos;
    (void)pc_prev_head;
    (void)pc_pos;

    put_cid(&qp->sq, cid);
}



//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif


#endif // __PAGE_CACHE_H__
