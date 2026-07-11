package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const netSampleWindow = 120 * time.Millisecond

// hostMetrics returns REAL host resource metrics read from the Linux /proc
// filesystem and a statfs of the storage path. The four current* keys are
// ALWAYS present (value null when a metric cannot be measured on this platform)
// so API clients can render without guarding for missing fields — but values are
// only ever real measurements, never fabricated placeholders. This replaces a
// handler that returned hardcoded values (currentCpuUsage: 45.2, etc.) plus
// invented trend/analysis narrative.
func hostMetrics(storagePath string) map[string]interface{} {
	m := map[string]interface{}{
		"timestamp":           time.Now().UTC().Format(time.RFC3339),
		"source":              "host:/proc",
		"currentCpuUsage":     nil, // percent, busy
		"currentMemoryUsage":  nil, // percent, used
		"currentDiskUsage":    nil, // percent, used, of storage path
		"currentNetworkUsage": nil, // MB/s, rx+tx across non-loopback ifaces
	}

	// Single sampling window shared by CPU and network deltas.
	idle0, cpuTot0, net0, ok0 := sampleCPUNet()
	if ok0 {
		time.Sleep(netSampleWindow)
		idle1, cpuTot1, net1, ok1 := sampleCPUNet()
		if ok1 {
			if dt := cpuTot1 - cpuTot0; dt > 0 {
				busy := (1 - (idle1-idle0)/dt) * 100
				if busy < 0 {
					busy = 0
				}
				m["currentCpuUsage"] = round2(busy)
			}
			if net1 >= net0 {
				bytesPerSec := float64(net1-net0) / netSampleWindow.Seconds()
				m["currentNetworkUsage"] = round2(bytesPerSec / (1024 * 1024))
			}
		}
	}

	if mem, ok := memUsagePercent(); ok {
		m["currentMemoryUsage"] = round2(mem)
	}
	if disk, ok := diskUsagePercent(storagePath); ok {
		m["currentDiskUsage"] = round2(disk)
	}
	if l1, l5, l15, ok := loadAverages(); ok {
		m["loadAverage1m"] = l1
		m["loadAverage5m"] = l5
		m["loadAverage15m"] = l15
	}
	return m
}

func round2(f float64) float64 { return float64(int64(f*100+0.5)) / 100 }

// memUsagePercent parses /proc/meminfo -> (1 - MemAvailable/MemTotal) * 100.
func memUsagePercent() (float64, bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	defer f.Close()
	var total, avail float64
	var haveTotal, haveAvail bool
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		v, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			total, haveTotal = v, true
		case "MemAvailable:":
			avail, haveAvail = v, true
		}
	}
	if !haveTotal || !haveAvail || total == 0 {
		return 0, false
	}
	return (1 - avail/total) * 100, true
}

// sampleCPUNet reads one instantaneous sample of aggregate CPU jiffies (from
// /proc/stat) and total non-loopback network bytes (rx+tx, from /proc/net/dev).
func sampleCPUNet() (idle, total float64, netBytes uint64, ok bool) {
	idle, total, cok := cpuSample()
	netBytes, nok := netSample()
	// CPU is required for a meaningful sample; network may legitimately be 0.
	return idle, total, netBytes, cok && nok
}

func cpuSample() (idle, total float64, ok bool) {
	f, err := os.Open("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "cpu ") {
			continue
		}
		fields := strings.Fields(line)[1:]
		for i, fld := range fields {
			v, err := strconv.ParseFloat(fld, 64)
			if err != nil {
				continue
			}
			total += v
			if i == 3 || i == 4 { // idle + iowait
				idle += v
			}
		}
		return idle, total, true
	}
	return 0, 0, false
}

// netSample sums received+transmitted bytes across all non-loopback interfaces
// from /proc/net/dev.
func netSample() (uint64, bool) {
	f, err := os.Open("/proc/net/dev")
	if err != nil {
		return 0, false
	}
	defer f.Close()
	var sum uint64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		colon := strings.IndexByte(line, ':')
		if colon < 0 {
			continue // header lines have no colon
		}
		iface := strings.TrimSpace(line[:colon])
		if iface == "lo" {
			continue
		}
		fields := strings.Fields(line[colon+1:])
		if len(fields) < 9 {
			continue
		}
		// field 0 = rx bytes, field 8 = tx bytes
		if rx, err := strconv.ParseUint(fields[0], 10, 64); err == nil {
			sum += rx
		}
		if tx, err := strconv.ParseUint(fields[8], 10, 64); err == nil {
			sum += tx
		}
	}
	return sum, true
}

// loadAverages parses /proc/loadavg (1m, 5m, 15m).
func loadAverages() (l1, l5, l15 float64, ok bool) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, 0, 0, false
	}
	fields := strings.Fields(string(b))
	if len(fields) < 3 {
		return 0, 0, 0, false
	}
	l1, _ = strconv.ParseFloat(fields[0], 64)
	l5, _ = strconv.ParseFloat(fields[1], 64)
	l15, _ = strconv.ParseFloat(fields[2], 64)
	return l1, l5, l15, true
}

// diskUsagePercent statfs's the storage path -> used%.
func diskUsagePercent(path string) (float64, bool) {
	if path == "" {
		return 0, false
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, false
	}
	total := float64(st.Blocks) * float64(st.Bsize)
	avail := float64(st.Bavail) * float64(st.Bsize)
	if total == 0 {
		return 0, false
	}
	return (1 - avail/total) * 100, true
}
