package provision

import (
	"context"
	"sync"
)

// Fake is an in-memory Provisioner for testing Provisioner consumers (e.g.
// the api-server's /networks handlers) without touching the host. It
// validates specs like the real one and records every call.
type Fake struct {
	mu        sync.Mutex
	ensureErr error
	removeErr error
	networks  map[string]Spec // by bridge
	calls     []string
}

var _ Provisioner = (*Fake)(nil)

// FailEnsure makes subsequent Ensure calls return err (nil clears it)
// without changing state.
func (f *Fake) FailEnsure(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ensureErr = err
}

// FailRemove makes subsequent Remove calls return err (nil clears it)
// without changing state.
func (f *Fake) FailRemove(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.removeErr = err
}

func (f *Fake) Ensure(_ context.Context, spec Spec) error {
	if err := spec.Validate(); err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "ensure "+spec.Bridge)
	if f.ensureErr != nil {
		return f.ensureErr
	}
	if f.networks == nil {
		f.networks = map[string]Spec{}
	}
	f.networks[spec.Bridge] = spec
	return nil
}

func (f *Fake) Remove(_ context.Context, spec Spec) error {
	bridge, err := BridgeName(spec.NetworkID)
	if err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "remove "+bridge)
	if f.removeErr != nil {
		return f.removeErr
	}
	delete(f.networks, bridge)
	return nil
}

// Provisioned reports whether bridge is currently provisioned, and its spec.
func (f *Fake) Provisioned(bridge string) (Spec, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	spec, ok := f.networks[bridge]
	return spec, ok
}

// Calls returns the recorded "ensure <bridge>" / "remove <bridge>" calls in
// order.
func (f *Fake) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}
