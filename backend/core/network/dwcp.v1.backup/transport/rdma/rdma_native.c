#include "rdma_native.h"
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdarg.h>

// Thread-local error storage
static __thread char error_buffer[256] = {0};

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(error_buffer, sizeof(error_buffer), fmt, args);
    va_end(args);
}

const char* rdma_get_error_string(void) {
    return error_buffer;
}

// Check if RDMA is available
int rdma_check_availability(void) {
    struct ibv_device **dev_list;
    int num_devices;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        set_error("No RDMA devices found");
        return 0;
    }

    ibv_free_device_list(dev_list);
    return num_devices;
}

// Get list of RDMA devices
int rdma_get_device_list(rdma_device_info_t** devices) {
    struct ibv_device **dev_list;
    struct ibv_context *ctx;
    struct ibv_device_attr dev_attr;
    int num_devices, i;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        set_error("No RDMA devices found");
        return 0;
    }

    *devices = calloc(num_devices, sizeof(rdma_device_info_t));
    if (!*devices) {
        ibv_free_device_list(dev_list);
        set_error("Memory allocation failed");
        return -1;
    }

    for (i = 0; i < num_devices; i++) {
        rdma_device_info_t *info = &(*devices)[i];

        strncpy(info->name, ibv_get_device_name(dev_list[i]), sizeof(info->name) - 1);
        snprintf(info->guid, sizeof(info->guid), "%016llx",
                 (unsigned long long)ibv_get_device_guid(dev_list[i]));

        // Open device to get attributes
        ctx = ibv_open_device(dev_list[i]);
        if (ctx) {
            if (ibv_query_device(ctx, &dev_attr) == 0) {
                info->max_mr_size = dev_attr.max_mr_size;
                info->max_qp = dev_attr.max_qp;
                info->max_cq = dev_attr.max_cq;
                info->max_cqe = dev_attr.max_cqe;

                // Check capabilities
                info->supports_rc = 1; // RC is always supported
                info->supports_ud = 1; // UD is always supported
                info->supports_rdma_write = (dev_attr.device_cap_flags & IBV_DEVICE_MEM_MGT_EXTENSIONS) != 0;
                info->supports_rdma_read = (dev_attr.device_cap_flags & IBV_DEVICE_MEM_MGT_EXTENSIONS) != 0;
                info->supports_atomic = (dev_attr.atomic_cap != IBV_ATOMIC_NONE);
            }
            ibv_close_device(ctx);
        }
    }

    ibv_free_device_list(dev_list);
    return num_devices;
}

void rdma_free_device_list(rdma_device_info_t* devices, int count) {
    if (devices) {
        free(devices);
    }
}

// Initialize RDMA context
rdma_context_t* rdma_init(const char* device_name, int port, int use_event_channel) {
    rdma_context_t* ctx = NULL;
    struct ibv_device **dev_list = NULL;
    struct ibv_device *device = NULL;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr;
    struct ibv_qp_init_attr qp_init_attr;
    int num_devices, i;

    ctx = calloc(1, sizeof(rdma_context_t));
    if (!ctx) {
        set_error("Failed to allocate context");
        return NULL;
    }

    // Set defaults
    ctx->port = port > 0 ? port : 1;
    ctx->gid_index = 0;
    ctx->max_send_wr = 1024;
    ctx->max_recv_wr = 1024;
    ctx->max_inline_data = 256;
    ctx->max_sge = 16;
    ctx->use_event_channel = use_event_channel;
    ctx->buffer_size = 4 * 1024 * 1024; // 4MB default buffer

    // Get device list
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        set_error("No RDMA devices found");
        goto cleanup;
    }

    // Find requested device or use first one
    if (device_name && strlen(device_name) > 0) {
        for (i = 0; i < num_devices; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), device_name) == 0) {
                device = dev_list[i];
                break;
            }
        }
        if (!device) {
            set_error("Device %s not found", device_name);
            goto cleanup;
        }
    } else {
        device = dev_list[0];
    }

    // Open device
    ctx->device_ctx = ibv_open_device(device);
    if (!ctx->device_ctx) {
        set_error("Failed to open device: %s", strerror(errno));
        goto cleanup;
    }

    // Query device attributes
    if (ibv_query_device(ctx->device_ctx, &device_attr)) {
        set_error("Failed to query device: %s", strerror(errno));
        goto cleanup;
    }

    // Query port attributes
    if (ibv_query_port(ctx->device_ctx, ctx->port, &port_attr)) {
        set_error("Failed to query port: %s", strerror(errno));
        goto cleanup;
    }

    if (port_attr.state != IBV_PORT_ACTIVE) {
        set_error("Port %d is not active", ctx->port);
        goto cleanup;
    }

    ctx->lid = port_attr.lid;

    // Get GID
    union ibv_gid gid;
    if (ibv_query_gid(ctx->device_ctx, ctx->port, ctx->gid_index, &gid)) {
        set_error("Failed to query GID: %s", strerror(errno));
        goto cleanup;
    }
    memcpy(ctx->gid, &gid, 16);

    // Allocate protection domain
    ctx->pd = ibv_alloc_pd(ctx->device_ctx);
    if (!ctx->pd) {
        set_error("Failed to allocate PD: %s", strerror(errno));
        goto cleanup;
    }

    // Create completion channel if requested
    if (use_event_channel) {
        ctx->comp_channel = ibv_create_comp_channel(ctx->device_ctx);
        if (!ctx->comp_channel) {
            set_error("Failed to create completion channel: %s", strerror(errno));
            goto cleanup;
        }
    }

    // Create completion queues
    ctx->send_cq = ibv_create_cq(ctx->device_ctx, ctx->max_send_wr, NULL,
                                  ctx->comp_channel, 0);
    if (!ctx->send_cq) {
        set_error("Failed to create send CQ: %s", strerror(errno));
        goto cleanup;
    }

    ctx->recv_cq = ibv_create_cq(ctx->device_ctx, ctx->max_recv_wr, NULL,
                                  ctx->comp_channel, 0);
    if (!ctx->recv_cq) {
        set_error("Failed to create recv CQ: %s", strerror(errno));
        goto cleanup;
    }

    // Request notifications if using event channel
    if (use_event_channel) {
        if (ibv_req_notify_cq(ctx->send_cq, 0) ||
            ibv_req_notify_cq(ctx->recv_cq, 0)) {
            set_error("Failed to request CQ notifications: %s", strerror(errno));
            goto cleanup;
        }
    }

    // Create queue pair
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = ctx->send_cq;
    qp_init_attr.recv_cq = ctx->recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC; // Reliable Connection
    qp_init_attr.cap.max_send_wr = ctx->max_send_wr;
    qp_init_attr.cap.max_recv_wr = ctx->max_recv_wr;
    qp_init_attr.cap.max_send_sge = ctx->max_sge;
    qp_init_attr.cap.max_recv_sge = ctx->max_sge;
    qp_init_attr.cap.max_inline_data = ctx->max_inline_data;

    ctx->qp = ibv_create_qp(ctx->pd, &qp_init_attr);
    if (!ctx->qp) {
        set_error("Failed to create QP: %s", strerror(errno));
        goto cleanup;
    }

    ctx->qp_num = ctx->qp->qp_num;
    ctx->psn = lrand48() & 0xffffff; // Random packet sequence number

    // Transition QP to INIT state
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = ctx->port;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_ATOMIC;

    if (ibv_modify_qp(ctx->qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        set_error("Failed to modify QP to INIT: %s", strerror(errno));
        goto cleanup;
    }

    if (dev_list) {
        ibv_free_device_list(dev_list);
    }

    return ctx;

cleanup:
    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
    if (ctx) {
        rdma_cleanup(ctx);
    }
    return NULL;
}

void rdma_cleanup(rdma_context_t* ctx) {
    if (!ctx) return;

    if (ctx->mr) {
        ibv_dereg_mr(ctx->mr);
    }
    if (ctx->buffer) {
        free(ctx->buffer);
    }
    if (ctx->qp) {
        ibv_destroy_qp(ctx->qp);
    }
    if (ctx->send_cq) {
        ibv_destroy_cq(ctx->send_cq);
    }
    if (ctx->recv_cq) {
        ibv_destroy_cq(ctx->recv_cq);
    }
    if (ctx->comp_channel) {
        ibv_destroy_comp_channel(ctx->comp_channel);
    }
    if (ctx->pd) {
        ibv_dealloc_pd(ctx->pd);
    }
    if (ctx->device_ctx) {
        ibv_close_device(ctx->device_ctx);
    }

    free(ctx);
}

// Memory registration
int rdma_register_memory(rdma_context_t* ctx, void* addr, size_t length) {
    if (!ctx || !ctx->pd) {
        set_error("Invalid context");
        return -1;
    }

    // Unregister existing memory if any
    if (ctx->mr) {
        ibv_dereg_mr(ctx->mr);
        ctx->mr = NULL;
    }

    // Register new memory region
    int access = IBV_ACCESS_LOCAL_WRITE |
                 IBV_ACCESS_REMOTE_READ |
                 IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_ATOMIC;

    ctx->mr = ibv_reg_mr(ctx->pd, addr, length, access);
    if (!ctx->mr) {
        set_error("Failed to register memory: %s", strerror(errno));
        return -1;
    }

    ctx->buffer = addr;
    ctx->buffer_size = length;

    return 0;
}

int rdma_unregister_memory(rdma_context_t* ctx) {
    if (!ctx || !ctx->mr) {
        return 0;
    }

    if (ibv_dereg_mr(ctx->mr)) {
        set_error("Failed to unregister memory: %s", strerror(errno));
        return -1;
    }

    ctx->mr = NULL;
    ctx->buffer = NULL;
    ctx->buffer_size = 0;

    return 0;
}

// Get connection info
int rdma_get_conn_info(rdma_context_t* ctx, rdma_conn_info_t* info) {
    if (!ctx || !info) {
        set_error("Invalid parameters");
        return -1;
    }

    info->lid = ctx->lid;
    info->qp_num = ctx->qp_num;
    info->psn = ctx->psn;
    memcpy(info->gid, ctx->gid, 16);

    return 0;
}

// Modify QP to Ready-To-Send state
int rdma_modify_qp_to_rts(rdma_context_t* ctx, rdma_conn_info_t* remote_info) {
    struct ibv_qp_attr qp_attr;
    int flags;

    // Transition to RTR (Ready to Receive)
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = remote_info->qp_num;
    qp_attr.rq_psn = remote_info->psn;
    qp_attr.max_dest_rd_atomic = 16;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid = *(union ibv_gid*)remote_info->gid;
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.hop_limit = 255;
    qp_attr.ah_attr.grh.sgid_index = ctx->gid_index;
    qp_attr.ah_attr.grh.traffic_class = 0;
    qp_attr.ah_attr.dlid = remote_info->lid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = ctx->port;

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if (ibv_modify_qp(ctx->qp, &qp_attr, flags)) {
        set_error("Failed to modify QP to RTR: %s", strerror(errno));
        return -1;
    }

    // Transition to RTS (Ready to Send)
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = ctx->psn;
    qp_attr.max_rd_atomic = 16;

    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(ctx->qp, &qp_attr, flags)) {
        set_error("Failed to modify QP to RTS: %s", strerror(errno));
        return -1;
    }

    ctx->connected = 1;
    return 0;
}

int rdma_connect(rdma_context_t* ctx, rdma_conn_info_t* remote_info) {
    return rdma_modify_qp_to_rts(ctx, remote_info);
}

// Post send request
int rdma_post_send(rdma_context_t* ctx, void* buf, size_t len, uint64_t wr_id) {
    if (!ctx || !ctx->qp || !ctx->mr) {
        set_error("Invalid context or memory not registered");
        return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)buf;
    sge.length = len;
    sge.lkey = ctx->mr->lkey;

    struct ibv_send_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    // Use inline data for small messages
    if (len <= ctx->max_inline_data) {
        wr.send_flags |= IBV_SEND_INLINE;
    }

    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        set_error("Failed to post send: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// Post receive request
int rdma_post_recv(rdma_context_t* ctx, void* buf, size_t len, uint64_t wr_id) {
    if (!ctx || !ctx->qp || !ctx->mr) {
        set_error("Invalid context or memory not registered");
        return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)buf;
    sge.length = len;
    sge.lkey = ctx->mr->lkey;

    struct ibv_recv_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(ctx->qp, &wr, &bad_wr)) {
        set_error("Failed to post recv: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// Post RDMA write
int rdma_post_write(rdma_context_t* ctx, void* local_buf, size_t len,
                    uint64_t remote_addr, uint32_t rkey, uint64_t wr_id) {
    if (!ctx || !ctx->qp || !ctx->mr) {
        set_error("Invalid context or memory not registered");
        return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_buf;
    sge.length = len;
    sge.lkey = ctx->mr->lkey;

    struct ibv_send_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        set_error("Failed to post RDMA write: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// Post RDMA read
int rdma_post_read(rdma_context_t* ctx, void* local_buf, size_t len,
                   uint64_t remote_addr, uint32_t rkey, uint64_t wr_id) {
    if (!ctx || !ctx->qp || !ctx->mr) {
        set_error("Invalid context or memory not registered");
        return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_buf;
    sge.length = len;
    sge.lkey = ctx->mr->lkey;

    struct ibv_send_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        set_error("Failed to post RDMA read: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// Poll for completion
int rdma_poll_completion(rdma_context_t* ctx, int is_send, uint64_t* wr_id, size_t* len) {
    struct ibv_wc wc;
    struct ibv_cq* cq = is_send ? ctx->send_cq : ctx->recv_cq;

    int ne = ibv_poll_cq(cq, 1, &wc);
    if (ne < 0) {
        set_error("Failed to poll CQ: %s", strerror(errno));
        return -1;
    }

    if (ne == 0) {
        return 0; // No completion
    }

    if (wc.status != IBV_WC_SUCCESS) {
        set_error("Work completion failed: %s", ibv_wc_status_str(wc.status));
        return -1;
    }

    if (wr_id) {
        *wr_id = wc.wr_id;
    }
    if (len) {
        *len = wc.byte_len;
    }

    return 1; // Completion found
}

// Wait for completion using event channel
int rdma_wait_completion(rdma_context_t* ctx, int is_send, uint64_t* wr_id, size_t* len) {
    if (!ctx->use_event_channel) {
        // Fall back to polling
        while (1) {
            int ret = rdma_poll_completion(ctx, is_send, wr_id, len);
            if (ret != 0) {
                return ret;
            }
            usleep(1); // Small delay to avoid spinning
        }
    }

    struct ibv_cq* cq = is_send ? ctx->send_cq : ctx->recv_cq;
    struct ibv_cq* ev_cq;
    void* ev_ctx;

    // Wait for notification
    if (ibv_get_cq_event(ctx->comp_channel, &ev_cq, &ev_ctx)) {
        set_error("Failed to get CQ event: %s", strerror(errno));
        return -1;
    }

    ibv_ack_cq_events(ev_cq, 1);

    // Request next notification
    if (ibv_req_notify_cq(ev_cq, 0)) {
        set_error("Failed to request CQ notification: %s", strerror(errno));
        return -1;
    }

    // Poll for the actual completion
    return rdma_poll_completion(ctx, is_send, wr_id, len);
}

// Get statistics
int rdma_get_stats(rdma_context_t* ctx, rdma_stats_t* stats) {
    if (!ctx || !stats) {
        set_error("Invalid parameters");
        return -1;
    }

    // For now, we don't track detailed stats in the C layer
    // This would be better tracked in the Go layer
    memset(stats, 0, sizeof(rdma_stats_t));
    return 0;
}

// Utility functions
uint64_t rdma_get_buffer_addr(rdma_context_t* ctx) {
    if (!ctx || !ctx->buffer) {
        return 0;
    }
    return (uint64_t)(uintptr_t)ctx->buffer;
}

uint32_t rdma_get_rkey(rdma_context_t* ctx) {
    if (!ctx || !ctx->mr) {
        return 0;
    }
    return ctx->mr->rkey;
}

int rdma_is_connected(rdma_context_t* ctx) {
    return ctx ? ctx->connected : 0;
}
