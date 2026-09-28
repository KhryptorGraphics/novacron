package network

import (
	"bufio"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"
)

// newTestMonitor starts a monitor whose collection loop never ticks during the
// test, so measurements are driven explicitly.
func newTestMonitor(t *testing.T, cfg *BandwidthMonitorConfig) *BandwidthMonitor {
	t.Helper()
	if cfg.MonitoringInterval == 0 {
		cfg.MonitoringInterval = time.Hour
	}
	bm := NewBandwidthMonitor(cfg, zap.NewNop())
	if err := bm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { bm.Stop() })
	return bm
}

type recordingHandler struct {
	mu     sync.Mutex
	alerts []*BandwidthAlert
}

func (h *recordingHandler) HandleAlert(alert *BandwidthAlert) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.alerts = append(h.alerts, alert)
	return nil
}

// take returns the alerts raised since the last call as "severity: message".
func (h *recordingHandler) take() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []string
	for _, a := range h.alerts {
		out = append(out, a.Severity+": "+a.Message)
	}
	h.alerts = nil
	return out
}

func TestThresholdInterfaceMatching(t *testing.T) {
	cases := []struct {
		name      string
		threshold string
		enabled   bool
		wantAlert bool
	}{
		{name: "exact name", threshold: "eth0", enabled: true, wantAlert: true},
		{name: "wildcard", threshold: "*", enabled: true, wantAlert: true},
		{name: "prefix glob", threshold: "eth*", enabled: true, wantAlert: true},
		{name: "other prefix glob", threshold: "wlan*", enabled: true},
		{name: "other interface", threshold: "eth1", enabled: true},
		{name: "disabled threshold", threshold: "*", enabled: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			handler := &recordingHandler{}
			bm := newTestMonitor(t, &BandwidthMonitorConfig{
				Interfaces:        []string{"eth0"},
				AlertHandlers:     []BandwidthAlertHandler{handler},
				DefaultThresholds: []BandwidthThreshold{{InterfaceName: tc.threshold, WarningThreshold: 70, CriticalThreshold: 90, Enabled: tc.enabled}},
			})

			bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 95})

			if got := handler.take(); (len(got) == 1) != tc.wantAlert || len(got) > 1 {
				t.Errorf("alerts = %q, want alert=%v", got, tc.wantAlert)
			}
		})
	}
}

func TestAlertKindsAreRateLimitedIndependently(t *testing.T) {
	handler := &recordingHandler{}
	var hookCalls []float64
	bm := newTestMonitor(t, &BandwidthMonitorConfig{
		Interfaces:        []string{"eth0"},
		AlertHandlers:     []BandwidthAlertHandler{handler},
		EnableQoSHooks:    true,
		DefaultThresholds: []BandwidthThreshold{{InterfaceName: "*", WarningThreshold: 70, CriticalThreshold: 90, AbsoluteLimit: 1_000_000, Enabled: true}},
	})
	bm.AddQoSHook(func(iface string, utilization float64) {
		if iface != "eth0" {
			t.Errorf("hook for %q", iface)
		}
		hookCalls = append(hookCalls, utilization)
	})

	steps := []struct {
		name      string
		m         BandwidthMeasurement
		backdate  bool // pretend the previous alerts are older than the one-minute limit
		want      []string
		wantHooks int
	}{
		{
			name: "critical utilization and absolute limit raise separate alerts",
			m:    BandwidthMeasurement{Utilization: 95, RXRate: 1_500_000, TXRate: 500_000},
			want: []string{
				"critical: Critical bandwidth utilization on eth0: 95.00% (threshold: 90.00%)",
				"critical: Absolute bandwidth limit exceeded on eth0: 2000000 bps (limit: 1000000 bps)",
			},
			wantHooks: 2,
		},
		{
			name: "repeats within a minute are suppressed",
			m:    BandwidthMeasurement{Utilization: 96, RXRate: 1_500_000, TXRate: 500_000},
		},
		{
			name: "warning uses its own rate-limit key and does not trigger QoS",
			m:    BandwidthMeasurement{Utilization: 75},
			want: []string{"warning: Warning bandwidth utilization on eth0: 75.00% (threshold: 70.00%)"},
		},
		{
			name:      "alerts resume after the rate-limit window",
			m:         BandwidthMeasurement{Utilization: 95},
			backdate:  true,
			want:      []string{"critical: Critical bandwidth utilization on eth0: 95.00% (threshold: 90.00%)"},
			wantHooks: 1,
		},
	}

	for _, step := range steps {
		if step.backdate {
			bm.alertsMutex.Lock()
			for key := range bm.lastAlerts {
				bm.lastAlerts[key] = time.Now().Add(-2 * time.Minute)
			}
			bm.alertsMutex.Unlock()
		}
		hookCalls = nil
		step.m.InterfaceName = "eth0"

		bm.checkThresholds("eth0", &step.m)

		if got := handler.take(); strings.Join(got, "\n") != strings.Join(step.want, "\n") {
			t.Errorf("%s: alerts = %q, want %q", step.name, got, step.want)
		}
		if len(hookCalls) != step.wantHooks {
			t.Errorf("%s: QoS hook ran %d times, want %d", step.name, len(hookCalls), step.wantHooks)
		}
	}
}

// A QoS manager may register its hook while the monitor is already raising
// alerts from its collection goroutine; run under -race.
func TestQoSHookRegistrationDuringAlerts(t *testing.T) {
	bm := newTestMonitor(t, &BandwidthMonitorConfig{
		Interfaces:        []string{"eth0"},
		EnableQoSHooks:    true,
		DefaultThresholds: []BandwidthThreshold{{InterfaceName: "*", WarningThreshold: 70, CriticalThreshold: 90, Enabled: true}},
	})
	alert := func() {
		bm.alertsMutex.Lock()
		clear(bm.lastAlerts)
		bm.alertsMutex.Unlock()
		bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 95})
	}

	var fired atomic.Int32
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 2000 {
			alert()
		}
	}()
	const hooks = 2000
	for range hooks {
		bm.AddQoSHook(func(string, float64) { fired.Add(1) })
	}
	<-done

	fired.Store(0)
	alert()
	if got := fired.Load(); got != hooks {
		t.Errorf("%d hooks ran for one critical alert, want %d", got, hooks)
	}
}

func TestQoSHooksRequireOptIn(t *testing.T) {
	bm := newTestMonitor(t, &BandwidthMonitorConfig{
		Interfaces:        []string{"eth0"},
		DefaultThresholds: []BandwidthThreshold{{InterfaceName: "*", WarningThreshold: 70, CriticalThreshold: 90, AbsoluteLimit: 1, Enabled: true}},
	})
	bm.AddQoSHook(func(string, float64) { t.Error("QoS hook ran with EnableQoSHooks unset") })

	bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 99, RXRate: 10})
}

func TestSetThresholdIsPerInterface(t *testing.T) {
	handler := &recordingHandler{}
	bm := newTestMonitor(t, &BandwidthMonitorConfig{
		Interfaces:        []string{"eth0", "eth1"},
		AlertHandlers:     []BandwidthAlertHandler{handler},
		DefaultThresholds: []BandwidthThreshold{{InterfaceName: "*", WarningThreshold: 70, CriticalThreshold: 90, Enabled: true}},
	})

	if err := bm.SetThreshold("eth0", BandwidthThreshold{InterfaceName: "*", WarningThreshold: 30, CriticalThreshold: 50, Enabled: true}); err != nil {
		t.Fatal(err)
	}

	bm.checkThresholds("eth1", &BandwidthMeasurement{InterfaceName: "eth1", Utilization: 60})
	if got := handler.take(); len(got) != 0 {
		t.Errorf("eth0's threshold leaked to eth1: %q", got)
	}
	bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 60})
	if got := handler.take(); len(got) != 1 || !strings.HasPrefix(got[0], "critical:") {
		t.Errorf("eth0 alerts = %q, want one critical alert", got)
	}

	if err := bm.SetThreshold("eth9", BandwidthThreshold{}); err == nil || err.Error() != "interface eth9 not monitored" {
		t.Errorf("unmonitored interface err = %v", err)
	}
}

func TestWindowedRate(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	sample := func(offset time.Duration, rx, tx uint64) BandwidthMeasurement {
		return BandwidthMeasurement{Timestamp: t0.Add(offset), RXBytes: rx, TXBytes: tx}
	}

	cases := []struct {
		name           string
		samples        []BandwidthMeasurement
		window         time.Duration
		wantRX, wantTX float64
		wantOK         bool
	}{
		{
			name:    "bytes per second become bits per second",
			samples: []BandwidthMeasurement{sample(0, 0, 0), sample(time.Second, 1000, 250)},
			window:  time.Minute, wantRX: 8000, wantTX: 2000, wantOK: true,
		},
		{
			name:    "uneven intervals are weighted by elapsed time",
			samples: []BandwidthMeasurement{sample(0, 0, 0), sample(time.Second, 3000, 0), sample(4*time.Second, 4000, 0)},
			window:  time.Minute, wantRX: 8000, wantOK: true,
		},
		{
			name:    "samples older than the window are ignored",
			samples: []BandwidthMeasurement{sample(0, 0, 0), sample(10*time.Second, 100_000, 0), sample(12*time.Second, 102_000, 0)},
			window:  5 * time.Second, wantRX: 8000, wantOK: true,
		},
		{
			name:    "window holding only the newest sample",
			samples: []BandwidthMeasurement{sample(0, 0, 0), sample(10*time.Second, 1000, 0)},
			window:  time.Second,
		},
		{
			name:    "counter reset inside the window",
			samples: []BandwidthMeasurement{sample(0, 5000, 0), sample(time.Second, 10, 0)},
			window:  time.Minute,
		},
		{
			name:    "single sample",
			samples: []BandwidthMeasurement{sample(0, 5000, 0)},
			window:  time.Minute,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rx, tx, ok := windowedRate(tc.samples, tc.window)
			if ok != tc.wantOK || rx != tc.wantRX || tx != tc.wantTX {
				t.Errorf("windowedRate = (%v, %v, %v), want (%v, %v, %v)", rx, tx, ok, tc.wantRX, tc.wantTX, tc.wantOK)
			}
		})
	}
}

func loopbackInProcNetDev(t *testing.T) bool {
	t.Helper()
	f, err := os.Open("/proc/net/dev")
	if err != nil {
		return false
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if strings.HasPrefix(strings.TrimSpace(scanner.Text()), "lo:") {
			return true
		}
	}
	return false
}

func TestCollectMetricsMeasuresLoopbackTraffic(t *testing.T) {
	if !loopbackInProcNetDev(t) {
		t.Skip("no loopback entry in /proc/net/dev")
	}
	bm := newTestMonitor(t, &BandwidthMonitorConfig{Interfaces: []string{"lo"}, SlidingWindowDuration: time.Hour})

	if _, err := bm.GetCurrentMeasurement("lo"); err == nil || err.Error() != "no measurements available for interface lo" {
		t.Fatalf("before collection err = %v", err)
	}
	bm.collectMetrics()
	first, err := bm.GetCurrentMeasurement("lo")
	if err != nil {
		t.Fatal(err)
	}
	if first.RXRate != 0 || first.TXRate != 0 {
		t.Errorf("first sample has no predecessor yet reports %v/%v bps", first.RXRate, first.TXRate)
	}

	receiver, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()
	go func() {
		buf := make([]byte, 65536)
		for {
			if _, _, err := receiver.ReadFromUDP(buf); err != nil {
				return
			}
		}
	}()
	sender, err := net.DialUDP("udp", nil, receiver.LocalAddr().(*net.UDPAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer sender.Close()

	const datagrams, size = 256, 16 << 10
	payload := make([]byte, size)
	for range datagrams {
		if _, err := sender.Write(payload); err != nil {
			t.Fatal(err)
		}
	}

	bm.collectMetrics()
	current, err := bm.GetCurrentMeasurement("lo")
	if err != nil {
		t.Fatal(err)
	}
	history, err := bm.GetHistoricalMeasurements("lo", time.Time{})
	if err != nil || len(history) != 2 {
		t.Fatalf("history = %d samples, err %v", len(history), err)
	}

	// Every datagram crossed lo, so the measured rate must cover at least the
	// payload bits over the interval between the two samples.
	elapsed := history[1].Timestamp.Sub(history[0].Timestamp).Seconds()
	floor := float64(datagrams*size) * 8 / elapsed
	if current.RXRate < floor || current.TXRate < floor {
		t.Errorf("rates rx=%.0f tx=%.0f bps, want >= %.0f bps (payload over %.4fs)", current.RXRate, current.TXRate, floor, elapsed)
	}
	if summary := bm.GetNetworkUtilizationSummary(); summary["lo"] != current.Utilization {
		t.Errorf("utilization summary = %v, want lo=%v", summary, current.Utilization)
	}

	if _, err := bm.GetCurrentMeasurement("eth9"); err == nil || err.Error() != "interface eth9 not monitored" {
		t.Errorf("unmonitored interface err = %v", err)
	}
}
