package auth

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/pquerna/otp/totp"
)

func TestTwoFactorServiceRecoversEnrollmentAfterRestart(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create mock database: %v", err)
	}
	defer db.Close()

	service := NewTwoFactorService("NovaCron", []byte("test-encryption-key"), db)
	mock.ExpectExec("INSERT INTO user_two_factor").WillReturnResult(sqlmock.NewResult(0, 1))
	setup, err := service.SetupTwoFactor("user-1", "user@example.com")
	if err != nil {
		t.Fatalf("setup 2FA: %v", err)
	}

	code, err := totp.GenerateCode(setup.Secret, time.Now().UTC())
	if err != nil {
		t.Fatalf("generate TOTP code: %v", err)
	}
	mock.ExpectExec("INSERT INTO user_two_factor").WillReturnResult(sqlmock.NewResult(0, 1))
	if err := service.VerifyAndEnable("user-1", code); err != nil {
		t.Fatalf("enable 2FA: %v", err)
	}

	storedCodes, err := json.Marshal(setup.BackupCodes)
	if err != nil {
		t.Fatalf("encode backup codes: %v", err)
	}
	now := time.Now()
	mock.ExpectQuery("SELECT secret, enabled, setup_at, last_used, backup_codes").
		WithArgs("user-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"secret", "enabled", "setup_at", "last_used", "backup_codes", "algorithm", "digits", "period",
		}).AddRow(setup.Secret, true, now, now, storedCodes, "SHA1", 6, 30))

	// A new service instance has no in-memory enrollment state. It must recover
	// both enabled status and the TOTP secret/backup codes from the database.
	restarted := NewTwoFactorService("NovaCron", []byte("test-encryption-key"), db)
	if !restarted.IsEnabled("user-1") {
		t.Fatal("enrollment was not restored after service restart")
	}
	codes, err := restarted.GetBackupCodes("user-1")
	if err != nil {
		t.Fatalf("read restored backup codes: %v", err)
	}
	if len(codes) != len(setup.BackupCodes) {
		t.Fatalf("restored %d backup codes, want %d", len(codes), len(setup.BackupCodes))
	}
	for i := range codes {
		if codes[i] != setup.BackupCodes[i] {
			t.Fatalf("restored backup code %d differs from persisted code", i)
		}
	}

	code, err = totp.GenerateCode(setup.Secret, time.Now().UTC())
	if err != nil {
		t.Fatalf("generate post-restart TOTP code: %v", err)
	}
	mock.ExpectExec("INSERT INTO user_two_factor").WillReturnResult(sqlmock.NewResult(0, 1))
	verified, err := restarted.VerifyCode(TwoFactorVerifyRequest{UserID: "user-1", Code: code})
	if err != nil {
		t.Fatalf("verify restored TOTP secret: %v", err)
	}
	if !verified.Valid {
		t.Fatalf("restored TOTP secret did not verify: %s", verified.Error)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("database expectations: %v", err)
	}
}
