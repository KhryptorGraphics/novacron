package network

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// tcRecorder stands in for the tc binary: it records each invocation and fails
// those whose argument line contains any of failOn.
type tcRecorder struct {
	mu     sync.Mutex
	calls  []string
	failOn []string
}

func (r *tcRecorder) run(args ...string) error {
	line := strings.Join(args, " ")
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, line)
	for _, f := range r.failOn {
		if strings.Contains(line, f) {
			return fmt.Errorf("tc %s: simulated failure", line)
		}
	}
	return nil
}

// take returns the recorded invocations and resets the log.
func (r *tcRecorder) take() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := r.calls
	r.calls = nil
	return calls
}

func (r *tcRecorder) fail(substrings ...string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.failOn = substrings
}

func newTestQoSManager(cfg *QoSManagerConfig, bm *BandwidthMonitor) (*QoSManager, *tcRecorder) {
	qm := NewQoSManager(cfg, bm, zap.NewNop())
	rec := &tcRecorder{}
	qm.shaper.runTC = rec.run
	return qm, rec
}

// addPolicy fails the test instead of hanging if AddPolicy blocks.
func addPolicy(t *testing.T, qm *QoSManager, policy *QoSPolicy) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- qm.AddPolicy(policy) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("AddPolicy(%s): %v", policy.Name, err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("AddPolicy(%s) blocked", policy.Name)
	}
}

func rateLimitPolicy(name, iface string, rate, burst uint64) *QoSPolicy {
	return &QoSPolicy{
		Name:          name,
		InterfaceName: iface,
		Enabled:       true,
		Actions:       []QoSAction{{Type: "rate_limit", RateLimit: rate, BurstLimit: burst, Priority: 5}},
	}
}

func equalLines(t *testing.T, got, want []string) {
	t.Helper()
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("tc invocations:\n  got  %q\n  want %q", got, want)
	}
}

func TestRateLimitPolicyProgramsTrafficControl(t *testing.T) {
	cases := []struct {
		name         string
		rootRate     uint64
		defaultClass string
	}{
		{name: "default root rate", rootRate: 0, defaultClass: "class add dev eth0 parent 1: classid 1:999 htb rate 1000mbit"},
		{name: "configured root rate", rootRate: 2_000_000_000, defaultClass: "class add dev eth0 parent 1: classid 1:999 htb rate 2.0gbit"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true, DefaultRateBps: tc.rootRate}, nil)
			policy := rateLimitPolicy("ssh-limit", "eth0", 100_000_000, 10_000_000)
			policy.Rules = []ClassificationRule{{Name: "ssh", DestPort: 22, Protocol: "tcp"}}

			addPolicy(t, qm, policy)

			equalLines(t, rec.take(), []string{
				"qdisc add dev eth0 root handle 1: htb default 999",
				tc.defaultClass,
				"class add dev eth0 parent 1: classid 1:10 htb rate 50.0mbit ceil 100.0mbit prio 5",
				"qdisc add dev eth0 parent 1:10 pfifo",
				"class change dev eth0 parent 1: classid 1:10 htb rate 100.0mbit burst 10000000",
			})

			if policy.ID == "" || policy.CreatedAt.IsZero() {
				t.Errorf("policy not initialised: id=%q created=%v", policy.ID, policy.CreatedAt)
			}
			if got, err := qm.GetPolicy(policy.ID); err != nil || got != policy {
				t.Fatalf("GetPolicy = %v, %v", got, err)
			}
			if qm.appliedClasses[policy.ID] == "" {
				t.Error("rate-limit class not recorded as applied")
			}
			if rule, err := qm.classifier.ClassifyPacket("eth0", "10.0.0.1", "10.0.0.2", 40000, 22, "TCP"); err != nil || rule == nil || rule.Name != "ssh" {
				t.Errorf("ClassifyPacket = %v, %v; want the policy's ssh rule", rule, err)
			}

			if err := qm.RemovePolicy(policy.ID); err != nil {
				t.Fatalf("RemovePolicy: %v", err)
			}
			if _, err := qm.GetPolicy(policy.ID); err == nil {
				t.Error("policy still retrievable after removal")
			}
			if err := qm.RemovePolicy(policy.ID); err == nil {
				t.Error("removing an unknown policy should fail")
			}
		})
	}
}

func TestPoliciesShareInterfaceShaping(t *testing.T) {
	qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true}, nil)
	policies := []*QoSPolicy{
		rateLimitPolicy("a", "eth0", 10_000_000, 0),
		rateLimitPolicy("b", "eth0", 20_000_000, 0),
		rateLimitPolicy("c", "eth0", 30_000_000, 0),
	}
	for _, p := range policies {
		addPolicy(t, qm, p)
	}

	roots := 0
	for _, call := range rec.take() {
		if strings.Contains(call, "root handle 1:") {
			roots++
		}
	}
	if roots != 1 {
		t.Errorf("root qdisc installed %d times, want once per interface", roots)
	}

	// Each class keeps the tc id it was created with; rate changes must target
	// that id regardless of map iteration order.
	for round := range 5 {
		for i, p := range policies {
			rate := uint64(1_000_000 * (round + 1))
			if err := qm.shaper.ApplyRateLimit("eth0", p.ID, rate, 0); err != nil {
				t.Fatalf("ApplyRateLimit(%s): %v", p.Name, err)
			}
			want := fmt.Sprintf("class change dev eth0 parent 1: classid 1:%d htb rate %s", 10+i, bpsToTcRate(rate))
			equalLines(t, rec.take(), []string{want})
		}
	}
}

func TestTrafficShaperErrors(t *testing.T) {
	cases := []struct {
		name    string
		failOn  []string
		run     func(*TrafficShaper) error
		wantErr string
	}{
		{
			name: "interface configured twice",
			run: func(ts *TrafficShaper) error {
				if err := ts.SetupInterface("eth0"); err != nil {
					return err
				}
				return ts.SetupInterface("eth0")
			},
			wantErr: "interface eth0 already configured",
		},
		{
			name:   "root qdisc failure leaves interface unconfigured",
			failOn: []string{"root handle"},
			run: func(ts *TrafficShaper) error {
				if err := ts.SetupInterface("eth0"); err == nil || !strings.Contains(err.Error(), "failed to setup root qdisc") {
					return fmt.Errorf("SetupInterface err = %v", err)
				}
				return ts.AddTrafficClass("eth0", &TrafficClass{ID: "c1"})
			},
			wantErr: "interface eth0 not configured for shaping",
		},
		{
			name:   "default class and leaf qdisc failures are not fatal",
			failOn: []string{"classid 1:999", "pfifo"},
			run: func(ts *TrafficShaper) error {
				if err := ts.SetupInterface("eth0"); err != nil {
					return err
				}
				return ts.AddTrafficClass("eth0", &TrafficClass{ID: "c1", MaxBandwidth: 1000})
			},
		},
		{
			name:   "failed class is not registered",
			failOn: []string{"class add dev eth0 parent 1: classid 1:10"},
			run: func(ts *TrafficShaper) error {
				if err := ts.SetupInterface("eth0"); err != nil {
					return err
				}
				if err := ts.AddTrafficClass("eth0", &TrafficClass{ID: "c1"}); err == nil || !strings.Contains(err.Error(), "failed to add traffic class") {
					return fmt.Errorf("AddTrafficClass err = %v", err)
				}
				return ts.ApplyRateLimit("eth0", "c1", 1000, 0)
			},
			wantErr: "traffic class c1 not found",
		},
		{
			name:    "rate limit on unknown interface",
			run:     func(ts *TrafficShaper) error { return ts.ApplyRateLimit("eth9", "c1", 1000, 0) },
			wantErr: "interface eth9 not configured",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts := NewTrafficShaper(zap.NewNop())
			rec := &tcRecorder{failOn: tc.failOn}
			ts.runTC = rec.run

			err := tc.run(ts)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || err.Error() != tc.wantErr {
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestHandleBandwidthAlert(t *testing.T) {
	cases := []struct {
		name        string
		iface       string
		utilization float64
		wantRates   map[string]uint64
		wantTC      []string
	}{
		{
			name: "congested interface throttles its enabled limits by 20%", iface: "eth0", utilization: 85,
			wantRates: map[string]uint64{"eth0-on": 160_000_000, "eth0-off": 100_000_000, "eth1-on": 100_000_000},
			wantTC:    []string{"class change dev eth0 parent 1: classid 1:10 htb rate 160.0mbit burst 20000000"},
		},
		{
			name: "80% is not congestion", iface: "eth0", utilization: 80,
			wantRates: map[string]uint64{"eth0-on": 200_000_000, "eth0-off": 100_000_000, "eth1-on": 100_000_000},
		},
		{
			name: "other interfaces keep their limits", iface: "eth1", utilization: 95,
			wantRates: map[string]uint64{"eth0-on": 200_000_000, "eth0-off": 100_000_000, "eth1-on": 80_000_000},
			wantTC:    []string{"class change dev eth1 parent 1: classid 1:10 htb rate 80.0mbit"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true}, nil)
			disabled := rateLimitPolicy("eth0-off", "eth0", 100_000_000, 0)
			disabled.Enabled = false
			policies := []*QoSPolicy{
				rateLimitPolicy("eth0-on", "eth0", 200_000_000, 20_000_000),
				rateLimitPolicy("eth1-on", "eth1", 100_000_000, 0),
				disabled,
				{Name: "eth0-dscp", InterfaceName: "eth0", Enabled: true, Actions: []QoSAction{{Type: "dscp_mark", DSCPMark: 46}}},
			}
			for _, p := range policies {
				addPolicy(t, qm, p)
			}
			rec.take()

			qm.handleBandwidthAlert(tc.iface, tc.utilization)

			for _, p := range policies[:3] {
				if got := p.Actions[0].RateLimit; got != tc.wantRates[p.Name] {
					t.Errorf("%s rate = %d, want %d", p.Name, got, tc.wantRates[p.Name])
				}
			}
			equalLines(t, rec.take(), tc.wantTC)
		})
	}
}

func TestReconcileStateReappliesMissingClasses(t *testing.T) {
	cases := []struct {
		name   string
		failOn string
		want   []string
	}{
		{
			name:   "interface setup failed",
			failOn: "root handle",
			want: []string{
				"qdisc add dev eth0 root handle 1: htb default 999",
				"class add dev eth0 parent 1: classid 1:999 htb rate 1000mbit",
				"class add dev eth0 parent 1: classid 1:10 htb rate 25.0mbit ceil 50.0mbit prio 5",
				"qdisc add dev eth0 parent 1:10 pfifo",
				"class change dev eth0 parent 1: classid 1:10 htb rate 50.0mbit",
			},
		},
		{
			name:   "class creation failed",
			failOn: "class add dev eth0 parent 1: classid 1:10",
			want: []string{
				"class add dev eth0 parent 1: classid 1:10 htb rate 25.0mbit ceil 50.0mbit prio 5",
				"qdisc add dev eth0 parent 1:10 pfifo",
				"class change dev eth0 parent 1: classid 1:10 htb rate 50.0mbit",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true}, nil)
			limited := rateLimitPolicy("limited", "eth0", 50_000_000, 0)
			disabled := rateLimitPolicy("disabled", "eth0", 10_000_000, 0)
			disabled.Enabled = false

			// Both policies are added while tc fails, so neither has a class yet.
			rec.fail(tc.failOn)
			addPolicy(t, qm, limited)
			addPolicy(t, qm, disabled)
			if len(qm.appliedClasses) != 0 {
				t.Fatalf("classes recorded as applied although tc failed: %v", qm.appliedClasses)
			}
			rec.fail()
			rec.take()

			qm.reconcileState()
			equalLines(t, rec.take(), tc.want)
			if _, applied := qm.appliedClasses[limited.ID]; !applied {
				t.Error("reconciliation did not record the re-applied class")
			}
			if _, applied := qm.appliedClasses[disabled.ID]; applied {
				t.Error("reconciliation applied a disabled policy")
			}

			qm.reconcileState()
			equalLines(t, rec.take(), nil)
		})
	}
}

func TestPolicyQueries(t *testing.T) {
	qm, _ := newTestQoSManager(&QoSManagerConfig{}, nil)
	policies := []*QoSPolicy{
		{Name: "voip", NetworkID: "net-1", InterfaceName: "eth0", Enabled: true, Actions: []QoSAction{{Type: "dscp_mark", DSCPMark: 46}}},
		{Name: "bulk", NetworkID: "net-1", InterfaceName: "eth0", Enabled: false},
		{Name: "web", NetworkID: "net-1", InterfaceName: "eth1", Enabled: true},
		{Name: "other", NetworkID: "net-2", InterfaceName: "eth0", Enabled: true},
	}
	for _, p := range policies {
		addPolicy(t, qm, p)
	}

	status := qm.GetNetworkQoSStatus("net-1")
	if status["network_id"] != "net-1" || status["total_policies"] != 3 || status["active_policies"] != 2 {
		t.Errorf("status = %v, want 3 total / 2 active", status)
	}
	if listed := status["policies"].([]*QoSPolicy); len(listed) != 3 {
		t.Errorf("status lists %d policies, want 3", len(listed))
	}
	if empty := qm.GetNetworkQoSStatus("net-9"); empty["total_policies"] != 0 || empty["active_policies"] != 0 {
		t.Errorf("unknown network status = %v", empty)
	}

	names := map[string]bool{}
	for _, p := range qm.GetInterfacePolicies("eth0") {
		names[p.Name] = true
	}
	if len(names) != 2 || !names["voip"] || !names["other"] {
		t.Errorf("eth0 policies = %v, want enabled voip and other", names)
	}
	if got := len(qm.ListPolicies()); got != len(policies) {
		t.Errorf("ListPolicies returned %d, want %d", got, len(policies))
	}
}

func TestQueueActionCreatesQueue(t *testing.T) {
	qm, _ := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true}, nil)
	addPolicy(t, qm, &QoSPolicy{Name: "q", Enabled: true, Actions: []QoSAction{{Type: "queue", QueueName: "gold", RateLimit: 5000}}})

	stats, err := qm.queueManager.GetQueueStatistics("gold")
	if err != nil || stats.LastUpdated.IsZero() {
		t.Fatalf("queue statistics = %v, %v", stats, err)
	}
	if err := qm.queueManager.CreateQueue(&QueueConfig{Name: "gold"}); err == nil {
		t.Error("duplicate queue name accepted")
	}
	if err := qm.queueManager.CreateQueue(&QueueConfig{}); err == nil {
		t.Error("unnamed queue accepted")
	}
}

func TestTrafficShapingDisabledSkipsTrafficControl(t *testing.T) {
	qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: false}, nil)
	addPolicy(t, qm, rateLimitPolicy("limit", "eth0", 1_000_000, 0))

	if calls := rec.take(); len(calls) != 0 {
		t.Errorf("tc invoked with shaping disabled: %q", calls)
	}
	if len(qm.appliedClasses) != 0 {
		t.Errorf("applied classes = %v", qm.appliedClasses)
	}
}

func TestQoSManagerStartAppliesDefaultPolicies(t *testing.T) {
	policy := rateLimitPolicy("default", "eth0", 1_000_000, 0)
	qm, rec := newTestQoSManager(&QoSManagerConfig{
		EnableTrafficShaping: true,
		UpdateInterval:       time.Hour,
		DefaultPolicies:      []*QoSPolicy{policy},
	}, nil)

	if err := qm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer qm.Stop()

	if _, err := qm.GetPolicy(policy.ID); err != nil {
		t.Fatalf("default policy not registered: %v", err)
	}
	if calls := rec.take(); len(calls) == 0 {
		t.Error("default policy was not programmed into tc")
	}
	if err := qm.Start(); err == nil || err.Error() != "QoS manager is already running" {
		t.Errorf("second Start err = %v", err)
	}
	if err := qm.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := qm.Stop(); err != nil {
		t.Errorf("second Stop err = %v", err)
	}
}

func TestUpdateStatisticsFromBandwidthMonitor(t *testing.T) {
	bm := newTestMonitor(t, &BandwidthMonitorConfig{Interfaces: []string{"eth0"}})
	monitored := bm.interfaces["eth0"]
	monitored.mu.Lock()
	monitored.lastMeasure = &BandwidthMeasurement{InterfaceName: "eth0", RXRate: 3_000_000, TXRate: 1_000_000, Utilization: 40}
	monitored.mu.Unlock()

	qm, _ := newTestQoSManager(&QoSManagerConfig{}, bm)
	onEth0 := &QoSPolicy{Name: "eth0", InterfaceName: "eth0", Enabled: true}
	unmonitored := &QoSPolicy{Name: "eth9", InterfaceName: "eth9", Enabled: true}
	addPolicy(t, qm, onEth0)
	addPolicy(t, qm, unmonitored)

	qm.updateStatistics()

	if s := onEth0.Statistics; s.ThroughputBps != 4_000_000 || s.UtilizationPercent != 40 || s.LastUpdated.IsZero() {
		t.Errorf("eth0 statistics = %+v", s)
	}
	if s := unmonitored.Statistics; s.ThroughputBps != 0 || s.LastUpdated.IsZero() {
		t.Errorf("unmonitored statistics = %+v", s)
	}
}

func TestCongestionAlertThrottlesQoS(t *testing.T) {
	bm := newTestMonitor(t, &BandwidthMonitorConfig{
		Interfaces:        []string{"eth0"},
		EnableQoSHooks:    true,
		DefaultThresholds: []BandwidthThreshold{{InterfaceName: "*", WarningThreshold: 70, CriticalThreshold: 90, Enabled: true}},
	})
	qm, rec := newTestQoSManager(&QoSManagerConfig{EnableTrafficShaping: true}, bm)
	policy := rateLimitPolicy("limit", "eth0", 100_000_000, 0)
	addPolicy(t, qm, policy)
	rec.take()

	bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 75})
	if got := policy.Actions[0].RateLimit; got != 100_000_000 {
		t.Fatalf("warning-level utilization changed the limit to %d", got)
	}

	bm.checkThresholds("eth0", &BandwidthMeasurement{InterfaceName: "eth0", Utilization: 95})
	if got := policy.Actions[0].RateLimit; got != 80_000_000 {
		t.Errorf("rate after critical alert = %d, want 80000000", got)
	}
	equalLines(t, rec.take(), []string{"class change dev eth0 parent 1: classid 1:10 htb rate 80.0mbit"})
}

func TestTrafficClassifier(t *testing.T) {
	tc := NewTrafficClassifier(zap.NewNop())
	for _, rule := range []ClassificationRule{
		{Name: "ssh", DestPort: 22, Protocol: "tcp"},
		{Name: "lan-dns", SourceIP: "10.0.0.0/8", DestPort: 53, Protocol: "udp"},
		{Name: "nas", DestIP: "192.168.1.10"},
	} {
		if err := tc.AddRule("eth0", rule); err != nil {
			t.Fatal(err)
		}
	}

	cases := []struct {
		name             string
		src, dst         string
		srcPort, dstPort int
		proto            string
		want             string
	}{
		{name: "protocol match is case-insensitive", src: "1.1.1.1", dst: "2.2.2.2", srcPort: 5000, dstPort: 22, proto: "TCP", want: "ssh"},
		{name: "source CIDR", src: "10.1.2.3", dst: "8.8.8.8", srcPort: 5000, dstPort: 53, proto: "udp", want: "lan-dns"},
		{name: "source outside CIDR", src: "172.16.0.1", dst: "8.8.8.8", srcPort: 5000, dstPort: 53, proto: "udp"},
		{name: "exact destination IP", src: "172.16.0.1", dst: "192.168.1.10", srcPort: 5000, dstPort: 445, proto: "tcp", want: "nas"},
		{name: "first matching rule wins", src: "172.16.0.1", dst: "192.168.1.10", srcPort: 5000, dstPort: 22, proto: "tcp", want: "ssh"},
		{name: "unparseable address never matches a CIDR", src: "not-an-ip", dst: "8.8.8.8", srcPort: 5000, dstPort: 53, proto: "udp"},
		{name: "wrong protocol", src: "1.1.1.1", dst: "2.2.2.2", srcPort: 5000, dstPort: 22, proto: "udp"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rule, err := tc.ClassifyPacket("eth0", c.src, c.dst, c.srcPort, c.dstPort, c.proto)
			if err != nil {
				t.Fatal(err)
			}
			got := ""
			if rule != nil {
				got = rule.Name
			}
			if got != c.want {
				t.Errorf("matched %q, want %q", got, c.want)
			}
		})
	}

	if _, err := tc.ClassifyPacket("eth1", "1.1.1.1", "2.2.2.2", 1, 2, "tcp"); err == nil || err.Error() != "no rules defined for interface eth1" {
		t.Errorf("unknown interface err = %v", err)
	}
}

func TestBpsToTcRate(t *testing.T) {
	for _, c := range []struct {
		bps  uint64
		want string
	}{
		{2_000_000_000, "2.0gbit"},
		{1_000_000_000, "1.0gbit"},
		{1_500_000, "1.5mbit"},
		{64_000, "64.0kbit"},
		{999, "999bit"},
	} {
		if got := bpsToTcRate(c.bps); got != c.want {
			t.Errorf("bpsToTcRate(%d) = %q, want %q", c.bps, got, c.want)
		}
	}
}
