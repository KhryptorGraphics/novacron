package main

import (
	"os"
	"testing"
)

// TestComputeUsageTotalsZeroRates: with the default all-zero rate card every
// quantity is still measurable (the whole point of metering-before-pricing)
// but no revenue is invented.
func TestComputeUsageTotalsZeroRates(t *testing.T) {
	totals := computeUsageTotals(64<<30, 3, 7200, 14400, usageRates{})
	if totals.EgressGB != 64 {
		t.Fatalf("egress GB conversion wrong: got %v want 64", totals.EgressGB)
	}
	if totals.VCPUHours != 4 {
		t.Fatalf("vcpu hours conversion wrong: got %v want 4", totals.VCPUHours)
	}
	if totals.EstimatedCost != 0 {
		t.Fatalf("zero rate card must estimate $0, got %v", totals.EstimatedCost)
	}
}

// TestComputeUsageTotalsPricing: each term contributes its own unit price.
func TestComputeUsageTotalsPricing(t *testing.T) {
	rates := usageRates{PerGBEgress: 0.04, PerVCPUHour: 0.027, PerJobSecond: 0.001, PerMigration: 0.50}
	totals := computeUsageTotals(100<<30 /*100 GiB*/, 2, 200, 7200 /*2 vCPU-hours*/, rates)
	want := 100*0.04 + 2*0.027 + 200*0.001 + 2*0.50
	if absDiff(totals.EstimatedCost, want) > 1e-9 {
		t.Fatalf("estimated cost: got %v want %v", totals.EstimatedCost, want)
	}
}

// TestLoadUsageRatesRejectsGarbage: negative or unparseable rates are forced
// to zero so a misconfigured operator env never invents revenue.
func TestLoadUsageRatesRejectsGarbage(t *testing.T) {
	t.Setenv("NOVACRON_RATE_PER_GB_EGRESS", "-1")
	t.Setenv("NOVACRON_RATE_PER_VCPU_HOUR", "not-a-number")
	t.Setenv("NOVACRON_RATE_PER_JOB_SECOND", "0.005")
	os.Unsetenv("NOVACRON_RATE_PER_MIGRATION")

	rates := loadUsageRates()
	if rates.PerGBEgress != 0 || rates.PerVCPUHour != 0 {
		t.Fatalf("invalid rates must zero out: %+v", rates)
	}
	if rates.PerJobSecond != 0.005 {
		t.Fatalf("valid rate lost: %+v", rates)
	}
	if rates.PerMigration != 0 {
		t.Fatalf("unset rate must default to 0: %+v", rates)
	}
}

// TestRecordUsageEventGuards: nil DB and non-positive quantities are silently
// dropped — billing telemetry must never fail the operation it observes.
func TestRecordUsageEventGuards(t *testing.T) {
	recordUsageEvent(t.Context(), nil, usageEvent{EventType: "egress_bytes", Quantity: 100})
	recordUsageEvent(t.Context(), nil, usageEvent{EventType: "egress_bytes", Quantity: 0})
	recordUsageEvent(t.Context(), nil, usageEvent{})
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}
