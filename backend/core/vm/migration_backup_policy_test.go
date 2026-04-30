package vm

import "testing"

func TestDefaultMigrationBackupPolicyGatesLiveMigration(t *testing.T) {
	policy := DefaultMigrationBackupPolicy(MigrationModeLive)

	if got, want := policy.DefaultMigrationMode, MigrationModeCheckpoint; got != want {
		t.Fatalf("default migration mode = %q, want %q", got, want)
	}
	if containsPolicyValue(policy.AllowedMigrationModes, MigrationModeLive) {
		t.Fatalf("live migration should not be allowed by default: %#v", policy.AllowedMigrationModes)
	}
	if policy.LiveMigrationGate.Enabled {
		t.Fatal("live migration gate should be disabled by default")
	}
	if !policy.Backup.RequireRecentBackup {
		t.Fatal("live policy should require a recent backup")
	}
	if !policy.Recovery.CheckpointRestore {
		t.Fatal("live policy should enable checkpoint restore")
	}
	if err := policy.Validate(); err != nil {
		t.Fatalf("default live policy should validate while live remains gated: %v", err)
	}
}

func TestMigrationBackupPolicyRejectsUngatedLiveMigration(t *testing.T) {
	policy := DefaultMigrationBackupPolicy(MigrationModeCold)
	policy.AllowedMigrationModes = append(policy.AllowedMigrationModes, MigrationModeLive)
	policy.LiveMigrationGate.Enabled = false

	if err := policy.Validate(); err == nil {
		t.Fatal("expected ungated live migration policy to be rejected")
	}
}
