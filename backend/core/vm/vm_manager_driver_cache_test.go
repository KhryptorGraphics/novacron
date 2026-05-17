package vm

import (
	"errors"
	"testing"
)

func TestVMManagerGetDriverReusesInitializedDriver(t *testing.T) {
	expected := &CoreStubDriver{}
	manager := &VMManager{
		drivers: map[VMType]VMDriver{
			VMTypeKVM: expected,
		},
		driverFactory: func(VMConfig) (VMDriver, error) {
			return nil, errors.New("driver factory should not be called for initialized drivers")
		},
	}

	driver, err := manager.GetDriverForConfig(VMConfig{
		Type: VMTypeKVM,
		Tags: map[string]string{"vm_type": string(VMTypeKVM)},
	})
	if err != nil {
		t.Fatalf("get initialized driver: %v", err)
	}
	if driver != expected {
		t.Fatalf("expected initialized driver %p, got %p", expected, driver)
	}
}
