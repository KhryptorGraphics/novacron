#!/bin/bash
# NovaCron Kernel Optimization Validation Script
# Ubuntu 24.04 LTS - Validate and benchmark kernel optimizations

set -euo pipefail

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/tmp/novacron-validation-$(date +%Y%m%d_%H%M%S)"
VERBOSE=false
RUN_BENCHMARKS=false
GENERATE_REPORT=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING] $*${NC}" >&2
}

error() {
    echo -e "${RED}[ERROR] $*${NC}" >&2
}

info() {
    echo -e "${BLUE}[INFO] $*${NC}" >&2
}

debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG] $*${NC}" >&2
    fi
}

create_results_dir() {
    mkdir -p "$RESULTS_DIR"
    log "Results directory created: $RESULTS_DIR"
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_grub_config() {
    log "Validating GRUB configuration..."
    local score=0
    local max_score=10
    
    local cmdline=$(cat /proc/cmdline)
    
    # Check performance governor
    if echo "$cmdline" | grep -q "cpufreq.default_governor=performance"; then
        info "✓ CPU governor set to performance"
        ((score++))
    else
        warn "✗ CPU governor not set to performance"
    fi
    
    # Check CPU isolation
    if echo "$cmdline" | grep -q "isolcpus="; then
        local isolated=$(echo "$cmdline" | grep -o "isolcpus=[0-9-,]*" | cut -d= -f2)
        info "✓ CPU isolation configured: $isolated"
        ((score++))
    else
        warn "✗ CPU isolation not configured"
    fi
    
    # Check huge pages
    if echo "$cmdline" | grep -q "hugepages=[0-9]*"; then
        local hugepages=$(echo "$cmdline" | grep -o "hugepages=[0-9]*" | cut -d= -f2)
        info "✓ Huge pages configured: $hugepages x 2MB"
        ((score++))
    else
        warn "✗ Huge pages not configured"
    fi
    
    # Check KVM nested virtualization
    if echo "$cmdline" | grep -q "kvm.nested=1"; then
        info "✓ KVM nested virtualization enabled"
        ((score++))
    else
        warn "✗ KVM nested virtualization not enabled"
    fi
    
    # Check I/O scheduler
    if echo "$cmdline" | grep -q "elevator=mq-deadline"; then
        info "✓ Multi-queue deadline I/O scheduler configured"
        ((score++))
    else
        warn "✗ I/O scheduler not optimized"
    fi
    
    # Check NUMA balancing
    if echo "$cmdline" | grep -q "numa_balancing=enable"; then
        info "✓ NUMA balancing enabled"
        ((score++))
    else
        warn "✗ NUMA balancing not enabled"
    fi
    
    # Check CPU idle optimization
    if echo "$cmdline" | grep -q "processor.max_cstate=1"; then
        info "✓ CPU idle states optimized"
        ((score++))
    else
        warn "✗ CPU idle states not optimized"
    fi
    
    # Check IOMMU configuration
    if echo "$cmdline" | grep -q "iommu=pt"; then
        info "✓ IOMMU passthrough enabled"
        ((score++))
    else
        warn "✗ IOMMU not configured for passthrough"
    fi
    
    # Check security mitigations
    if echo "$cmdline" | grep -q "mitigations=off"; then
        warn "⚠ CPU mitigations disabled (performance vs security trade-off)"
        ((score++))
    elif echo "$cmdline" | grep -q "mitigations=auto"; then
        info "✓ Balanced security configuration"
        ((score++))
    else
        warn "? Unknown mitigation configuration"
    fi
    
    # Check memory test disabled
    if echo "$cmdline" | grep -q "memtest=0"; then
        info "✓ Memory test disabled for faster boot"
        ((score++))
    else
        info "- Memory test not disabled"
    fi
    
    local percentage=$((score * 100 / max_score))
    log "GRUB configuration score: $score/$max_score ($percentage%)"
    
    echo "$score,$max_score,GRUB Configuration" >> "$RESULTS_DIR/validation_scores.csv"
    echo "GRUB Command Line: $cmdline" >> "$RESULTS_DIR/kernel_config.txt"
    
    return 0
}

validate_sysctl_config() {
    log "Validating sysctl configuration..."
    local score=0
    local max_score=12
    
    # Check memory settings
    local swappiness=$(sysctl -n vm.swappiness 2>/dev/null || echo "60")
    if [[ "$swappiness" -le 10 ]]; then
        info "✓ Swappiness optimized: $swappiness"
        ((score++))
    else
        warn "✗ Swappiness not optimized: $swappiness (should be ≤10)"
    fi
    
    local dirty_ratio=$(sysctl -n vm.dirty_ratio 2>/dev/null || echo "20")
    if [[ "$dirty_ratio" -le 10 ]]; then
        info "✓ Dirty ratio optimized: $dirty_ratio%"
        ((score++))
    else
        warn "✗ Dirty ratio not optimized: $dirty_ratio% (should be ≤10%)"
    fi
    
    # Check network settings
    local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null || echo "0")
    if [[ "$rmem_max" -ge 16777216 ]]; then
        info "✓ Network receive buffer optimized: $rmem_max bytes"
        ((score++))
    else
        warn "✗ Network receive buffer not optimized: $rmem_max (should be ≥16MB)"
    fi
    
    local netdev_backlog=$(sysctl -n net.core.netdev_max_backlog 2>/dev/null || echo "1000")
    if [[ "$netdev_backlog" -ge 5000 ]]; then
        info "✓ Network device backlog optimized: $netdev_backlog"
        ((score++))
    else
        warn "✗ Network device backlog not optimized: $netdev_backlog (should be ≥5000)"
    fi
    
    # Check TCP settings
    local tcp_congestion=$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null || echo "cubic")
    if [[ "$tcp_congestion" == "bbr" ]]; then
        info "✓ TCP congestion control optimized: $tcp_congestion"
        ((score++))
    else
        warn "✗ TCP congestion control not optimized: $tcp_congestion (should be bbr)"
    fi
    
    # Check scheduler settings
    local sched_latency=$(sysctl -n kernel.sched_latency_ns 2>/dev/null || echo "0")
    if [[ "$sched_latency" -le 6000000 && "$sched_latency" -gt 0 ]]; then
        info "✓ Scheduler latency optimized: $sched_latency ns"
        ((score++))
    else
        warn "✗ Scheduler latency not optimized: $sched_latency ns"
    fi
    
    # Check file system settings
    local aio_max=$(sysctl -n fs.aio-max-nr 2>/dev/null || echo "65536")
    if [[ "$aio_max" -ge 1048576 ]]; then
        info "✓ AIO maximum optimized: $aio_max"
        ((score++))
    else
        warn "✗ AIO maximum not optimized: $aio_max (should be ≥1048576)"
    fi
    
    local file_max=$(sysctl -n fs.file-max 2>/dev/null || echo "0")
    if [[ "$file_max" -ge 2097152 ]]; then
        info "✓ File descriptor limit optimized: $file_max"
        ((score++))
    else
        warn "✗ File descriptor limit not optimized: $file_max"
    fi
    
    # Check memory management
    local min_free_kbytes=$(sysctl -n vm.min_free_kbytes 2>/dev/null || echo "0")
    if [[ "$min_free_kbytes" -ge 131072 ]]; then
        info "✓ Minimum free memory optimized: $min_free_kbytes KB"
        ((score++))
    else
        warn "✗ Minimum free memory not optimized: $min_free_kbytes KB"
    fi
    
    # Check NUMA settings
    local numa_balancing=$(sysctl -n vm.numa_balancing 2>/dev/null || echo "0")
    if [[ "$numa_balancing" == "1" ]]; then
        info "✓ NUMA balancing enabled"
        ((score++))
    else
        warn "✗ NUMA balancing not enabled"
    fi
    
    # Check performance monitoring
    local perf_paranoid=$(sysctl -n kernel.perf_event_paranoid 2>/dev/null || echo "3")
    if [[ "$perf_paranoid" == "-1" ]]; then
        info "✓ Performance monitoring enabled"
        ((score++))
    else
        warn "✗ Performance monitoring restricted: $perf_paranoid"
    fi
    
    # Check compaction settings
    local compaction=$(sysctl -n vm.compact_memory 2>/dev/null || echo "0")
    if [[ "$compaction" == "1" ]]; then
        info "✓ Memory compaction enabled"
        ((score++))
    else
        info "- Memory compaction not explicitly enabled"
    fi
    
    local percentage=$((score * 100 / max_score))
    log "Sysctl configuration score: $score/$max_score ($percentage%)"
    
    echo "$score,$max_score,Sysctl Configuration" >> "$RESULTS_DIR/validation_scores.csv"
    sysctl -a 2>/dev/null | grep -E "(vm\.|net\.|kernel\.)" | head -50 >> "$RESULTS_DIR/sysctl_settings.txt"
    
    return 0
}

validate_modules() {
    log "Validating kernel modules..."
    local score=0
    local max_score=8
    
    # Check KVM modules
    if lsmod | grep -q "^kvm "; then
        info "✓ KVM module loaded"
        ((score++))
    else
        warn "✗ KVM module not loaded"
    fi
    
    # Check CPU-specific KVM modules
    if lsmod | grep -q "kvm_intel\|kvm_amd"; then
        local kvm_cpu_module=$(lsmod | grep -o "kvm_intel\|kvm_amd" | head -1)
        info "✓ CPU-specific KVM module loaded: $kvm_cpu_module"
        ((score++))
    else
        warn "✗ CPU-specific KVM module not loaded"
    fi
    
    # Check VFIO modules
    if lsmod | grep -q "vfio"; then
        info "✓ VFIO modules loaded"
        ((score++))
    else
        warn "✗ VFIO modules not loaded"
    fi
    
    # Check network modules
    if lsmod | grep -q "bridge\|tun"; then
        info "✓ Network virtualization modules loaded"
        ((score++))
    else
        warn "✗ Network virtualization modules not loaded"
    fi
    
    # Check storage modules
    if lsmod | grep -q "loop\|dm_mod"; then
        info "✓ Storage virtualization modules loaded"
        ((score++))
    else
        warn "✗ Storage virtualization modules not loaded"
    fi
    
    # Check module parameters
    local kvm_halt_poll=$(cat /sys/module/kvm/parameters/halt_poll_ns 2>/dev/null || echo "0")
    if [[ "$kvm_halt_poll" -gt 0 ]]; then
        info "✓ KVM halt polling optimized: $kvm_halt_poll ns"
        ((score++))
    else
        warn "✗ KVM halt polling not optimized"
    fi
    
    # Check nested virtualization
    local nested_enabled=false
    if [[ -f "/sys/module/kvm_intel/parameters/nested" ]]; then
        local nested=$(cat /sys/module/kvm_intel/parameters/nested 2>/dev/null || echo "N")
        if [[ "$nested" == "1" || "$nested" == "Y" ]]; then
            nested_enabled=true
        fi
    elif [[ -f "/sys/module/kvm_amd/parameters/nested" ]]; then
        local nested=$(cat /sys/module/kvm_amd/parameters/nested 2>/dev/null || echo "0")
        if [[ "$nested" == "1" ]]; then
            nested_enabled=true
        fi
    fi
    
    if [[ "$nested_enabled" == "true" ]]; then
        info "✓ Nested virtualization enabled"
        ((score++))
    else
        warn "✗ Nested virtualization not enabled"
    fi
    
    # Check device availability
    if [[ -c "/dev/kvm" ]]; then
        info "✓ KVM device available"
        ((score++))
    else
        error "✗ KVM device not available"
    fi
    
    local percentage=$((score * 100 / max_score))
    log "Module configuration score: $score/$max_score ($percentage%)"
    
    echo "$score,$max_score,Module Configuration" >> "$RESULTS_DIR/validation_scores.csv"
    lsmod | head -30 >> "$RESULTS_DIR/loaded_modules.txt"
    
    return 0
}

validate_huge_pages() {
    log "Validating huge pages configuration..."
    local score=0
    local max_score=6
    
    # Check huge pages configured
    local hugepages_2mb=$(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}')
    if [[ "$hugepages_2mb" -gt 0 ]]; then
        info "✓ Huge pages configured: $hugepages_2mb x 2MB"
        ((score++))
    else
        warn "✗ No huge pages configured"
    fi
    
    # Check huge pages free
    local hugepages_free=$(grep "HugePages_Free:" /proc/meminfo | awk '{print $2}')
    local utilization=0
    if [[ "$hugepages_2mb" -gt 0 ]]; then
        utilization=$(( (hugepages_2mb - hugepages_free) * 100 / hugepages_2mb ))
        info "✓ Huge pages utilization: $utilization%"
        if [[ "$utilization" -gt 0 && "$utilization" -lt 100 ]]; then
            ((score++))
        fi
    fi
    
    # Check huge page size
    local hugepage_size=$(grep "Hugepagesize:" /proc/meminfo | awk '{print $2}')
    if [[ "$hugepage_size" == "2048" ]]; then
        info "✓ Huge page size optimized: ${hugepage_size}kB"
        ((score++))
    else
        warn "✗ Huge page size not optimal: ${hugepage_size}kB"
    fi
    
    # Check transparent huge pages
    local thp_setting=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "unknown")
    if echo "$thp_setting" | grep -q "\[madvise\]"; then
        info "✓ Transparent huge pages set to madvise"
        ((score++))
    else
        warn "✗ Transparent huge pages not set to madvise: $thp_setting"
    fi
    
    # Check huge page mount
    if mountpoint -q /dev/hugepages; then
        info "✓ Huge pages filesystem mounted"
        ((score++))
    else
        warn "✗ Huge pages filesystem not mounted"
    fi
    
    # Check huge page pool size
    local total_hugepage_mem=$((hugepages_2mb * 2048))  # KB
    local total_system_mem=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}')
    local hugepage_percentage=$((total_hugepage_mem * 100 / total_system_mem))
    
    if [[ "$hugepage_percentage" -ge 10 && "$hugepage_percentage" -le 50 ]]; then
        info "✓ Huge page allocation reasonable: $hugepage_percentage% of system memory"
        ((score++))
    else
        warn "✗ Huge page allocation may be suboptimal: $hugepage_percentage% of system memory"
    fi
    
    local percentage=$((score * 100 / max_score))
    log "Huge pages configuration score: $score/$max_score ($percentage%)"
    
    echo "$score,$max_score,Huge Pages Configuration" >> "$RESULTS_DIR/validation_scores.csv"
    grep -i huge /proc/meminfo >> "$RESULTS_DIR/hugepages_status.txt"
    
    return 0
}

validate_cpu_performance() {
    log "Validating CPU performance configuration..."
    local score=0
    local max_score=6
    
    # Check CPU governor
    local governors=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | head -5)
    if echo "$governors" | grep -q "performance"; then
        info "✓ Performance CPU governor active"
        ((score++))
    else
        warn "✗ CPU governor not set to performance"
    fi
    
    # Check CPU frequency scaling
    local cpu0_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
    local cpu0_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "1")
    if [[ "$cpu0_freq" -gt 0 && "$cpu0_max" -gt 0 ]]; then
        local freq_percentage=$((cpu0_freq * 100 / cpu0_max))
        if [[ "$freq_percentage" -ge 95 ]]; then
            info "✓ CPU frequency at maximum: $cpu0_freq MHz ($freq_percentage%)"
            ((score++))
        else
            warn "✗ CPU frequency not at maximum: $cpu0_freq MHz ($freq_percentage%)"
        fi
    else
        warn "✗ Cannot determine CPU frequency"
    fi
    
    # Check CPU idle states
    local idle_states=$(find /sys/devices/system/cpu/cpu0/cpuidle -name "disable" 2>/dev/null | wc -l)
    if [[ "$idle_states" -gt 0 ]]; then
        local disabled_states=$(find /sys/devices/system/cpu/cpu0/cpuidle -name "disable" -exec cat {} \; 2>/dev/null | grep -c "1" || echo "0")
        if [[ "$disabled_states" -gt 0 ]]; then
            info "✓ CPU idle states optimized: $disabled_states states disabled"
            ((score++))
        else
            warn "✗ CPU idle states not optimized"
        fi
    fi
    
    # Check CPU isolation
    if cat /proc/cmdline | grep -q "isolcpus="; then
        local isolated_cores=$(cat /proc/cmdline | grep -o "isolcpus=[0-9-,]*" | cut -d= -f2)
        info "✓ CPU isolation configured: $isolated_cores"
        ((score++))
    else
        warn "✗ CPU isolation not configured"
    fi
    
    # Check NUMA topology
    local numa_nodes=$(find /sys/devices/system/node -name "node*" -type d 2>/dev/null | wc -l)
    if [[ "$numa_nodes" -gt 1 ]]; then
        info "✓ NUMA system detected: $numa_nodes nodes"
        ((score++))
    else
        info "- Non-NUMA system (single node)"
        ((score++))  # Not a problem for single-socket systems
    fi
    
    # Check context switch rate
    local context_switches=$(grep "^ctxt" /proc/stat | awk '{print $2}')
    sleep 1
    local context_switches_after=$(grep "^ctxt" /proc/stat | awk '{print $2}')
    local switch_rate=$((context_switches_after - context_switches))
    
    if [[ "$switch_rate" -lt 100000 ]]; then
        info "✓ Context switch rate reasonable: $switch_rate/sec"
        ((score++))
    else
        warn "✗ High context switch rate: $switch_rate/sec (may indicate overload)"
    fi
    
    local percentage=$((score * 100 / max_score))
    log "CPU performance score: $score/$max_score ($percentage%)"
    
    echo "$score,$max_score,CPU Performance" >> "$RESULTS_DIR/validation_scores.csv"
    lscpu >> "$RESULTS_DIR/cpu_info.txt"
    cat /proc/cpuinfo | head -30 >> "$RESULTS_DIR/cpu_info.txt"
    
    return 0
}

run_performance_benchmarks() {
    if [[ "$RUN_BENCHMARKS" != "true" ]]; then
        return 0
    fi
    
    log "Running performance benchmarks..."
    
    # CPU benchmark
    if command -v sysbench >/dev/null; then
        info "Running CPU benchmark..."
        sysbench cpu --cpu-max-prime=20000 --threads=4 --time=30 run > "$RESULTS_DIR/cpu_benchmark.txt" 2>&1
    fi
    
    # Memory benchmark  
    if command -v sysbench >/dev/null; then
        info "Running memory benchmark..."
        sysbench memory --memory-total-size=2G --threads=4 run > "$RESULTS_DIR/memory_benchmark.txt" 2>&1
    fi
    
    # I/O benchmark
    if command -v fio >/dev/null; then
        info "Running I/O benchmark..."
        fio --name=test --ioengine=libaio --rw=randrw --bs=4k --size=1G --numjobs=4 --time_based --runtime=30 > "$RESULTS_DIR/io_benchmark.txt" 2>&1
    else
        warn "fio not available, skipping I/O benchmark"
    fi
    
    log "Benchmarks completed"
}

generate_summary_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    log "Generating summary report..."
    
    local report_file="$RESULTS_DIR/optimization_validation_report.txt"
    
    cat > "$report_file" << EOF
NovaCron Kernel Optimization Validation Report
==============================================

Date: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
Distribution: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)

VALIDATION SCORES
================
EOF
    
    if [[ -f "$RESULTS_DIR/validation_scores.csv" ]]; then
        echo "" >> "$report_file"
        while IFS=',' read -r score max_score category; do
            local percentage=$((score * 100 / max_score))
            printf "%-25s %2d/%2d (%3d%%)\n" "$category:" "$score" "$max_score" "$percentage" >> "$report_file"
        done < "$RESULTS_DIR/validation_scores.csv"
        
        # Calculate overall score
        local total_score=$(awk -F',' '{sum+=$1} END {print sum}' "$RESULTS_DIR/validation_scores.csv")
        local total_max_score=$(awk -F',' '{sum+=$2} END {print sum}' "$RESULTS_DIR/validation_scores.csv")
        local overall_percentage=$((total_score * 100 / total_max_score))
        
        echo "" >> "$report_file"
        printf "%-25s %2d/%2d (%3d%%)\n" "OVERALL SCORE:" "$total_score" "$total_max_score" "$overall_percentage" >> "$report_file"
        
        # Performance assessment
        echo "" >> "$report_file"
        echo "PERFORMANCE ASSESSMENT" >> "$report_file"
        echo "=====================" >> "$report_file"
        
        if [[ "$overall_percentage" -ge 90 ]]; then
            echo "✓ EXCELLENT: Kernel optimizations properly configured" >> "$report_file"
        elif [[ "$overall_percentage" -ge 75 ]]; then
            echo "✓ GOOD: Most optimizations configured, minor adjustments recommended" >> "$report_file"
        elif [[ "$overall_percentage" -ge 60 ]]; then
            echo "⚠ FAIR: Some optimizations missing, review configuration" >> "$report_file"
        else
            echo "✗ POOR: Major optimization issues, reinstallation recommended" >> "$report_file"
        fi
    fi
    
    # System summary
    cat >> "$report_file" << EOF

SYSTEM SUMMARY
==============
CPU Cores: $(nproc)
Total Memory: $(free -h | grep "^Mem:" | awk '{print $2}')
Available Memory: $(free -h | grep "^Mem:" | awk '{print $7}')
Huge Pages: $(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}') x 2MB
KVM Available: $([[ -c "/dev/kvm" ]] && echo "Yes" || echo "No")
IOMMU Status: $(dmesg | grep -i "iommu.*enabled" >/dev/null && echo "Enabled" || echo "Disabled/Not Available")

RECOMMENDATIONS
===============
EOF

    # Add specific recommendations based on validation results
    if ! cat /proc/cmdline | grep -q "isolcpus="; then
        echo "- Configure CPU isolation for better VM performance" >> "$report_file"
    fi
    
    if [[ "$(grep "HugePages_Total:" /proc/meminfo | awk '{print $2}')" -eq 0 ]]; then
        echo "- Configure huge pages for improved memory performance" >> "$report_file"
    fi
    
    if ! lsmod | grep -q "kvm"; then
        echo "- Load KVM modules for virtualization support" >> "$report_file"
    fi
    
    local swappiness=$(sysctl -n vm.swappiness 2>/dev/null || echo "60")
    if [[ "$swappiness" -gt 10 ]]; then
        echo "- Reduce swappiness for better VM memory performance" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "For detailed optimization guide, see:" >> "$report_file"
    echo "docs/KERNEL_OPTIMIZATION_GUIDE.md" >> "$report_file"
    echo "" >> "$report_file"
    echo "Results directory: $RESULTS_DIR" >> "$report_file"
    
    log "Summary report generated: $report_file"
    
    # Display summary on console
    echo
    echo -e "${BLUE}=== VALIDATION SUMMARY ===${NC}"
    cat "$report_file" | grep -A 20 "VALIDATION SCORES"
    echo
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

show_help() {
    cat << EOF
NovaCron Kernel Optimization Validation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -b, --benchmarks    Run performance benchmarks (requires sysbench, fio)
    -r, --results-dir   Custom results directory
    --no-report         Skip generating summary report

DESCRIPTION:
    Validates the NovaCron kernel optimizations and measures their effectiveness.
    Checks GRUB configuration, sysctl parameters, kernel modules, huge pages,
    and CPU performance settings.

EXAMPLES:
    $0                          # Basic validation
    $0 --verbose                # Detailed output
    $0 --benchmarks             # Include performance benchmarks
    $0 --results-dir /tmp/test  # Custom results location

VALIDATION CATEGORIES:
    - GRUB Configuration        (Boot parameters)
    - Sysctl Configuration     (Runtime parameters)  
    - Module Configuration     (Kernel modules)
    - Huge Pages Configuration (Memory optimization)
    - CPU Performance          (CPU optimization)

SCORING:
    90-100%: Excellent optimization
    75-89%:  Good optimization
    60-74%:  Fair optimization  
    <60%:    Poor optimization

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -b|--benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

main() {
    log "Starting NovaCron kernel optimization validation..."
    
    create_results_dir
    
    # Initialize results file
    echo "Score,MaxScore,Category" > "$RESULTS_DIR/validation_scores.csv"
    
    # Run validation checks
    validate_grub_config
    validate_sysctl_config  
    validate_modules
    validate_huge_pages
    validate_cpu_performance
    
    # Run benchmarks if requested
    run_performance_benchmarks
    
    # Generate summary report
    generate_summary_report
    
    log "Validation completed successfully!"
    log "Results available in: $RESULTS_DIR"
}

# Execute main function
main "$@"