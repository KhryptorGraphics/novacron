package main

import (
	"testing"
	"time"
)

// TestPercentile locks down the index math (a plausible off-by-one spot) with
// known small inputs, matching the "one runnable check for non-trivial logic"
// project convention -- this file's real end-to-end validation is the actual
// load-test runs recorded in the commit/report, not this unit test.
func TestPercentile(t *testing.T) {
	ms := func(vals ...int) []time.Duration {
		out := make([]time.Duration, len(vals))
		for i, v := range vals {
			out[i] = time.Duration(v) * time.Millisecond
		}
		return out
	}

	cases := []struct {
		name string
		in   []time.Duration
		p    float64
		want time.Duration
	}{
		{"empty", nil, 0.95, 0},
		{"single", ms(100), 0.95, 100 * time.Millisecond},
		{"p50 of 4", ms(10, 20, 30, 40), 0.50, 20 * time.Millisecond},
		{"p95 of 100", func() []time.Duration {
			v := make([]time.Duration, 100)
			for i := range v {
				v[i] = time.Duration(i+1) * time.Millisecond
			}
			return v
		}(), 0.95, 95 * time.Millisecond},
		{"p99 of 4 clamps to max", ms(10, 20, 30, 40), 0.99, 40 * time.Millisecond},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := percentile(c.in, c.p)
			if got != c.want {
				t.Errorf("percentile(%v, %v) = %v, want %v", c.in, c.p, got, c.want)
			}
		})
	}
}
