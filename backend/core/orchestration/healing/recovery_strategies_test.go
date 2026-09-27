package healing

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeVMController struct {
	mu           sync.Mutex
	restartCalls []string
	migrateCalls []struct{ vmID, target string }
	restartErr   error
	migrateErr   error
}

func (f *fakeVMController) RestartVM(ctx context.Context, vmID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.restartCalls = append(f.restartCalls, vmID)
	return f.restartErr
}

func (f *fakeVMController) MigrateVM(ctx context.Context, vmID, targetNode string, options map[string]string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.migrateCalls = append(f.migrateCalls, struct{ vmID, target string }{vmID, targetNode})
	return f.migrateErr
}

type fakeTargetSelector struct {
	target string
	err    error
}

func (f *fakeTargetSelector) SelectTarget(ctx context.Context, vmID string) (string, error) {
	return f.target, f.err
}

func testLoggerQuiet() *logrus.Logger {
	l := logrus.New()
	l.SetLevel(logrus.ErrorLevel)
	return l
}

func TestRestartRecoveryStrategy_VM_CallsInjectedController(t *testing.T) {
	cases := []struct {
		name    string
		fakeErr error
	}{
		{"success", nil},
		{"backend error", errors.New("driver refused restart")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			strategy := NewRestartRecoveryStrategy(testLoggerQuiet())
			fake := &fakeVMController{restartErr: tc.fakeErr}
			strategy.SetVMController(fake)

			result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "vm-1", Type: TargetTypeVM})

			require.Len(t, fake.restartCalls, 1)
			assert.Equal(t, "vm-1", fake.restartCalls[0])
			if tc.fakeErr == nil {
				require.NoError(t, err)
				assert.True(t, result.Success)
			} else {
				require.Error(t, err)
				assert.Equal(t, tc.fakeErr, err)
				assert.False(t, result.Success)
			}
		})
	}
}

func TestRestartRecoveryStrategy_UnsupportedTargetTypes(t *testing.T) {
	for _, tt := range []TargetType{TargetTypeService, TargetTypeNode, TargetTypeCluster, ""} {
		t.Run(string(tt), func(t *testing.T) {
			strategy := NewRestartRecoveryStrategy(testLoggerQuiet())
			fake := &fakeVMController{}
			strategy.SetVMController(fake)

			result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "x", Type: tt})

			require.Error(t, err)
			assert.Contains(t, err.Error(), "not supported")
			assert.False(t, result.Success)
			assert.Empty(t, fake.restartCalls)
		})
	}
}

func TestRestartRecoveryStrategy_NoControllerConfigured(t *testing.T) {
	strategy := NewRestartRecoveryStrategy(testLoggerQuiet())
	result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "vm-1", Type: TargetTypeVM})
	require.Error(t, err)
	assert.False(t, result.Success)
}

func TestMigrateRecoveryStrategy_VM_UsesSelectedTarget(t *testing.T) {
	strategy := NewMigrateRecoveryStrategy(testLoggerQuiet())
	fake := &fakeVMController{}
	strategy.SetVMController(fake)
	strategy.SetMigrationTargetSelector(&fakeTargetSelector{target: "node-b"})

	result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "vm-1", Type: TargetTypeVM})

	require.NoError(t, err)
	assert.True(t, result.Success)
	require.Len(t, fake.migrateCalls, 1)
	assert.Equal(t, "vm-1", fake.migrateCalls[0].vmID)
	assert.Equal(t, "node-b", fake.migrateCalls[0].target)
	assert.Equal(t, "node-b", result.Metadata["target_node"])
}

func TestMigrateRecoveryStrategy_SelectorError(t *testing.T) {
	strategy := NewMigrateRecoveryStrategy(testLoggerQuiet())
	fake := &fakeVMController{}
	strategy.SetVMController(fake)
	selErr := errors.New("no capacity")
	strategy.SetMigrationTargetSelector(&fakeTargetSelector{err: selErr})

	result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "vm-1", Type: TargetTypeVM})

	require.Error(t, err)
	assert.False(t, result.Success)
	assert.Empty(t, fake.migrateCalls)
}

func TestMigrateRecoveryStrategy_UnsupportedTargetTypes(t *testing.T) {
	for _, tt := range []TargetType{TargetTypeService, TargetTypeNode, TargetTypeCluster, ""} {
		t.Run(string(tt), func(t *testing.T) {
			strategy := NewMigrateRecoveryStrategy(testLoggerQuiet())
			fake := &fakeVMController{}
			strategy.SetVMController(fake)
			strategy.SetMigrationTargetSelector(&fakeTargetSelector{target: "node-b"})

			result, err := strategy.Recover(&FailureInfo{}, &HealingTarget{ID: "x", Type: tt})

			require.Error(t, err)
			assert.Contains(t, err.Error(), "not supported")
			assert.False(t, result.Success)
			assert.Empty(t, fake.migrateCalls)
		})
	}
}

func TestRestartRecoveryStrategy_CanRecoverManualTrigger(t *testing.T) {
	// TriggerHealing builds FailureTypeCustom/SeverityMedium; restart is the
	// strategy that must accept it so the manual heal endpoint works.
	failure := &FailureInfo{FailureType: FailureTypeCustom, Severity: SeverityMedium}

	restart := NewRestartRecoveryStrategy(testLoggerQuiet())
	assert.True(t, restart.CanRecover(failure), "restart must accept manual heal triggers")

	migrate := NewMigrateRecoveryStrategy(testLoggerQuiet())
	assert.False(t, migrate.CanRecover(failure), "migrate must not claim manual heal triggers")
}
