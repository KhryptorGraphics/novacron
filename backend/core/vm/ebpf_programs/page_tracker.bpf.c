// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/*
 * eBPF Page Tracker for VM Migration Optimization
 *
 * This program tracks page access patterns in guest VM memory to identify
 * unused pages that can be skipped during migration, reducing transfer overhead.
 */

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define PAGE_SIZE 4096
#define MAX_PAGES 1048576  // Track up to 4GB worth of pages (1M * 4KB)

/* Page tracking entry */
struct page_access_info {
	__u64 last_access_time;  // Nanoseconds since boot
	__u32 access_count;      // Number of accesses
	__u8 is_dirty;           // Whether page has been written to
	__u8 is_unused;          // Whether page is marked as unused
	__u8 reserved[2];        // Padding for alignment
};

/* Configuration for page tracking */
struct page_tracker_config {
	__u64 aging_threshold_ns;  // Time threshold for considering a page unused
	__u32 min_access_count;    // Minimum access count to consider a page active
	__u32 pid;                 // Target process ID (VM process)
};

/* BPF maps */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_PAGES);
	__type(key, __u64);  // Page frame number
	__type(value, struct page_access_info);
} page_access_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct page_tracker_config);
} config_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);
	__type(key, __u64);   // Virtual address
	__type(value, __u64); // Page frame number
} vaddr_to_pfn_map SEC(".maps");

/* Helper to get current timestamp in nanoseconds */
static __always_inline __u64 get_current_time_ns(void) {
	return bpf_ktime_get_ns();
}

/* Trace page fault events to track page accesses */
SEC("tp/exceptions/page_fault_user")
int trace_page_fault(struct trace_event_raw_page_fault_user *ctx) {
	__u32 zero = 0;
	struct page_tracker_config *config;
	__u64 current_time;
	__u32 pid;

	// Get configuration
	config = bpf_map_lookup_elem(&config_map, &zero);
	if (!config)
		return 0;

	// Filter by target PID
	pid = bpf_get_current_pid_tgid() >> 32;
	if (config->pid != 0 && pid != config->pid)
		return 0;

	// Get page frame number from faulting address
	__u64 address = ctx->address;
	__u64 pfn = address / PAGE_SIZE;

	current_time = get_current_time_ns();

	// Look up existing page access info
	struct page_access_info *info = bpf_map_lookup_elem(&page_access_map, &pfn);
	if (info) {
		// Update existing entry
		info->last_access_time = current_time;
		info->access_count++;
		info->is_unused = 0;

		// Mark as dirty if it's a write fault
		if (ctx->error_code & 0x2) {
			info->is_dirty = 1;
		}
	} else {
		// Create new entry
		struct page_access_info new_info = {
			.last_access_time = current_time,
			.access_count = 1,
			.is_dirty = (ctx->error_code & 0x2) ? 1 : 0,
			.is_unused = 0,
		};
		bpf_map_update_elem(&page_access_map, &pfn, &new_info, BPF_ANY);
	}

	// Update virtual address to PFN mapping
	bpf_map_update_elem(&vaddr_to_pfn_map, &address, &pfn, BPF_ANY);

	return 0;
}

/* Trace memory write events */
SEC("kprobe/handle_mm_fault")
int trace_memory_write(struct pt_regs *ctx) {
	__u32 zero = 0;
	struct page_tracker_config *config;
	__u64 current_time;
	__u32 pid;

	// Get configuration
	config = bpf_map_lookup_elem(&config_map, &zero);
	if (!config)
		return 0;

	// Filter by target PID
	pid = bpf_get_current_pid_tgid() >> 32;
	if (config->pid != 0 && pid != config->pid)
		return 0;

	// Get faulting address from first argument
	__u64 address = PT_REGS_PARM1(ctx);
	__u64 pfn = address / PAGE_SIZE;

	current_time = get_current_time_ns();

	// Update page access info
	struct page_access_info *info = bpf_map_lookup_elem(&page_access_map, &pfn);
	if (info) {
		info->last_access_time = current_time;
		info->access_count++;
		info->is_dirty = 1;
		info->is_unused = 0;
	}

	return 0;
}

/* Periodic cleanup function to mark aged-out pages as unused */
SEC("perf_event")
int periodic_aging(struct bpf_perf_event_data *ctx) {
	__u32 zero = 0;
	struct page_tracker_config *config;
	__u64 current_time;

	config = bpf_map_lookup_elem(&config_map, &zero);
	if (!config)
		return 0;

	current_time = get_current_time_ns();

	// This would ideally iterate over all pages, but BPF doesn't allow
	// iteration over all map entries in a single invocation.
	// In practice, the userspace program will periodically read and
	// update the map to mark aged-out pages.

	return 0;
}

char _license[] SEC("license") = "Dual BSD/GPL";
