package optimization

import (
	"reflect"
	"testing"
)

func TestParseCPUList(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []int
		wantErr bool
	}{
		{
			name:  "single range",
			input: "0-3",
			want:  []int{0, 1, 2, 3},
		},
		{
			name:  "mixed ranges",
			input: "0-2, 8, 10-11\n",
			want:  []int{0, 1, 2, 8, 10, 11},
		},
		{
			name:  "empty",
			input: " \n",
			want:  nil,
		},
		{
			name:    "reversed range",
			input:   "4-2",
			wantErr: true,
		},
		{
			name:    "invalid CPU",
			input:   "0,nope",
			wantErr: true,
		},
		{
			name:    "negative CPU",
			input:   "-1",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseCPUList(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("parseCPUList returned error: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parseCPUList = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNUMANodeMask(t *testing.T) {
	mask := numaNodeMask(65)
	if len(mask) < 2 {
		t.Fatalf("mask too short: %v", mask)
	}
	if mask[1] != 2 {
		t.Fatalf("node 65 should set bit 1 in word 1, got %b", mask[1])
	}
}

func TestAllocateNUMAMemoryRejectsInvalidInput(t *testing.T) {
	if _, err := AllocateNUMAMemory(-1, 0); err == nil {
		t.Fatal("expected negative size error")
	}
	if _, err := AllocateNUMAMemory(1, -1); err == nil {
		t.Fatal("expected negative node error")
	}
}

func TestSetSchedulerAffinityRejectsNegativePriority(t *testing.T) {
	if err := SetSchedulerAffinity(0, -1); err == nil {
		t.Fatal("expected negative priority error")
	}
}
