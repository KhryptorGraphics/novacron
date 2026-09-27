package vm

import (
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
)

func TestMigrationPortRangeDefault(t *testing.T) {
	t.Setenv(MigrationPortRangeEnv, "placeholder")
	if err := os.Unsetenv(MigrationPortRangeEnv); err != nil {
		t.Fatal(err)
	}
	lo, hi, err := MigrationPortRange()
	if err != nil {
		t.Fatal(err)
	}
	if lo != DefaultMigrationPortMin || hi != DefaultMigrationPortMax {
		t.Fatalf("unset range = %d-%d, want %d-%d", lo, hi, DefaultMigrationPortMin, DefaultMigrationPortMax)
	}

	t.Setenv(MigrationPortRangeEnv, "   ")
	lo, hi, err = MigrationPortRange()
	if err != nil {
		t.Fatal(err)
	}
	if lo != DefaultMigrationPortMin || hi != DefaultMigrationPortMax {
		t.Fatalf("blank range = %d-%d, want defaults", lo, hi)
	}
}

func TestMigrationPortRangeParsesPadded(t *testing.T) {
	t.Setenv(MigrationPortRangeEnv, " 50000 - 50063 ")
	lo, hi, err := MigrationPortRange()
	if err != nil {
		t.Fatal(err)
	}
	if lo != 50000 || hi != 50063 {
		t.Fatalf("got %d-%d", lo, hi)
	}
}

func TestMigrationPortRangeRejects(t *testing.T) {
	for _, v := range []string{"abc", "50000", "60000-50000", "80-90", "50000-70000"} {
		t.Run(v, func(t *testing.T) {
			t.Setenv(MigrationPortRangeEnv, v)
			_, _, err := MigrationPortRange()
			if err == nil {
				t.Fatal("expected error")
			}
			want := fmt.Sprintf("invalid %s %q: want LO-HI with 1024<=LO<=HI<=65535", MigrationPortRangeEnv, v)
			if err.Error() != want {
				t.Fatalf("error %q, want %q", err.Error(), want)
			}
		})
	}
}

func TestAllocateMigrationPortSkipsHeldAndExhausts(t *testing.T) {
	p := 0
	for cand := 20000; cand < 40000; cand++ {
		a, errA := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", cand))
		if errA != nil {
			continue
		}
		b, errB := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", cand+1))
		if errB != nil {
			a.Close()
			continue
		}
		a.Close()
		b.Close()
		p = cand
		break
	}
	if p == 0 {
		t.Fatal("no consecutive free ports from 20000")
	}

	held, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", p))
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	t.Setenv(MigrationPortRangeEnv, fmt.Sprintf("%d-%d", p, p+1))

	for i := 0; i < 10; i++ {
		got, err := AllocateMigrationPort()
		if err != nil {
			t.Fatal(err)
		}
		if got != p+1 {
			t.Fatalf("call %d returned %d, want %d", i, got, p+1)
		}
	}

	heldNext, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", p+1))
	if err != nil {
		t.Fatal(err)
	}
	defer heldNext.Close()
	_, err = AllocateMigrationPort()
	if err == nil || !strings.Contains(err.Error(), MigrationPortRangeEnv) {
		t.Fatalf("exhausted error %v, want it to mention %s", err, MigrationPortRangeEnv)
	}
}

func TestAllocateMigrationPortRejectsBadRange(t *testing.T) {
	t.Setenv(MigrationPortRangeEnv, "abc")
	_, err := AllocateMigrationPort()
	if err == nil || !strings.Contains(err.Error(), `invalid NOVACRON_MIGRATION_PORT_RANGE "abc"`) {
		t.Fatalf("got %v", err)
	}
}
