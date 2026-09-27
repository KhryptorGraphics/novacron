package metrics

import (
	"sync"
	"testing"
	"time"
)

func TestMetricsCollector_ConcurrentGetAndRecord(t *testing.T) {
	mc := NewMetricsCollector()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			mc.RecordSynapticOps(int64(i), float64(i+1))
		}()
		go func() {
			defer wg.Done()
			m := mc.GetMetrics()
			if m.LastUpdate.After(time.Now()) {
				t.Errorf("GetMetrics returned a future LastUpdate")
			}
		}()
	}
	wg.Wait()
}
