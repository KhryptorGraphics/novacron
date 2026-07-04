package main

import "testing"

// twoNodes is the heterogeneous baseline used by the pooling demo: arm64 thor +
// x86 dl560, both idle. Returns a fresh slice each call so cases can't bleed state.
func twoNodes() []NodeCapacity {
	return []NodeCapacity{
		{NodeID: "thor", MemTotalMB: 125748, StorageFreeGB: 2000, Reachable: true},
		{NodeID: "dl560", MemTotalMB: 322377, StorageFreeGB: 900, Reachable: true},
	}
}

func TestPlaceVM(t *testing.T) {
	// Idle cluster: an 8 GiB VM lands on the lower post-placement memory fraction
	// -- dl560 (8192/322377 < 8192/125748). Discriminates against first-fit, which
	// would pick thor.
	if got, ok := placeVM(twoNodes(), 8192, 2); !ok || got.NodeID != "dl560" {
		t.Fatalf("idle: got %q ok=%v, want dl560", got.NodeID, ok)
	}

	// Load dl560 to ~93%: thor is now the lower fraction, so placement flips. If the
	// fraction math were ignored, this would still pick dl560.
	loaded := twoNodes()
	loaded[1].MemAllocatedMB = 300000
	if got, ok := placeVM(loaded, 8192, 2); !ok || got.NodeID != "thor" {
		t.Fatalf("dl560 loaded: got %q ok=%v, want thor", got.NodeID, ok)
	}

	// Disk filter: a request larger than either node's free storage does not fit,
	// even though both have ample memory.
	if got, ok := placeVM(twoNodes(), 1024, 5000); ok {
		t.Fatalf("oversized disk: got %q, want no placement", got.NodeID)
	}

	// Memory filter -- the "nothing fits" branch: both nodes nearly full, request
	// exceeds each node's remaining headroom. Must return found=false.
	full := twoNodes()
	full[0].MemAllocatedMB = 121000 // thor avail 4748 < 8192
	full[1].MemAllocatedMB = 318000 // dl560 avail 4377 < 8192
	if got, ok := placeVM(full, 8192, 2); ok {
		t.Fatalf("both full: got %q, want no placement", got.NodeID)
	}

	// Unreachable nodes are never chosen, however much capacity they advertise.
	down := []NodeCapacity{{NodeID: "ghost", MemTotalMB: 999999, StorageFreeGB: 9999, Reachable: false}}
	if got, ok := placeVM(down, 1024, 1); ok {
		t.Fatalf("unreachable: got %q, want no placement", got.NodeID)
	}
}
