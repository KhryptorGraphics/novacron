#ifndef RDMA_NATIVE_H
#define RDMA_NATIVE_H

#include <stdint.h>
#include <stddef.h>

// Forward declarations for InfiniBand structures
struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_qp;
struct ibv_mr;
struct ibv_comp_channel;

// RDMA context structure
typedef struct rdma_context {
    // Device context
    struct ibv_context *device_ctx;
    struct ibv_pd *pd;

    // Queue pairs and completion queues
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_qp *qp;
    struct ibv_comp_channel *comp_channel;

    // Memory region
    struct ibv_mr *mr;
    void *buffer;
    size_t buffer_size;

    // Connection information
    uint16_t lid;
    uint32_t qp_num;
    uint32_t psn;
    uint8_t gid[16];

    // Configuration
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_inline_data;
    uint32_t max_sge;
    int port;
    int gid_index;

    // State
    int connected;
    int use_event_channel;
} rdma_context_t;

// RDMA connection info for exchange
typedef struct rdma_conn_info {
    uint16_t lid;
    uint32_t qp_num;
    uint32_t psn;
    uint8_t gid[16];
} rdma_conn_info_t;

// RDMA device info
typedef struct rdma_device_info {
    char name[64];
    char guid[64];
    int num_ports;
    uint64_t max_mr_size;
    uint32_t max_qp;
    uint32_t max_cq;
    uint32_t max_cqe;
    int supports_rc;
    int supports_ud;
    int supports_rdma_write;
    int supports_rdma_read;
    int supports_atomic;
} rdma_device_info_t;

// RDMA statistics
typedef struct rdma_stats {
    uint64_t send_completions;
    uint64_t recv_completions;
    uint64_t send_errors;
    uint64_t recv_errors;
    uint64_t bytes_sent;
    uint64_t bytes_received;
} rdma_stats_t;

// Initialization and cleanup
rdma_context_t* rdma_init(const char* device_name, int port, int use_event_channel);
void rdma_cleanup(rdma_context_t* ctx);

// Device enumeration
int rdma_get_device_list(rdma_device_info_t** devices);
void rdma_free_device_list(rdma_device_info_t* devices, int count);
int rdma_check_availability(void);

// Memory management
int rdma_register_memory(rdma_context_t* ctx, void* addr, size_t length);
int rdma_unregister_memory(rdma_context_t* ctx);

// Connection management
int rdma_get_conn_info(rdma_context_t* ctx, rdma_conn_info_t* info);
int rdma_connect(rdma_context_t* ctx, rdma_conn_info_t* remote_info);
int rdma_modify_qp_to_rts(rdma_context_t* ctx, rdma_conn_info_t* remote_info);

// Data operations
int rdma_post_send(rdma_context_t* ctx, void* buf, size_t len, uint64_t wr_id);
int rdma_post_recv(rdma_context_t* ctx, void* buf, size_t len, uint64_t wr_id);
int rdma_post_write(rdma_context_t* ctx, void* local_buf, size_t len,
                    uint64_t remote_addr, uint32_t rkey, uint64_t wr_id);
int rdma_post_read(rdma_context_t* ctx, void* local_buf, size_t len,
                   uint64_t remote_addr, uint32_t rkey, uint64_t wr_id);

// Completion handling
int rdma_poll_completion(rdma_context_t* ctx, int is_send, uint64_t* wr_id, size_t* len);
int rdma_wait_completion(rdma_context_t* ctx, int is_send, uint64_t* wr_id, size_t* len);

// Statistics and monitoring
int rdma_get_stats(rdma_context_t* ctx, rdma_stats_t* stats);
const char* rdma_get_error_string(void);

// Utility functions
uint64_t rdma_get_buffer_addr(rdma_context_t* ctx);
uint32_t rdma_get_rkey(rdma_context_t* ctx);
int rdma_is_connected(rdma_context_t* ctx);

#endif // RDMA_NATIVE_H
