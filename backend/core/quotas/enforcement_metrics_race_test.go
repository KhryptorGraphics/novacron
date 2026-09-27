package quotas

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestEnforcementEngine_GetMetrics_ConcurrentAccess(t *testing.T) {
	manager := NewManager(DefaultManagerConfig())
	require.NoError(t, manager.Start())
	defer manager.Stop()

	engine := NewEnforcementEngine(manager, nil)
	defer engine.Stop()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			engine.updateMetrics(time.Millisecond)
		}()
		go func() {
			defer wg.Done()
			m := engine.GetMetrics()
			_ = m.TotalRequests
		}()
	}
	wg.Wait()
}
