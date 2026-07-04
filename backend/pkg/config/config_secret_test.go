package config

import "testing"

// validBaseConfig returns a Config that passes every Validate() check except
// whatever the caller overrides on Auth.Secret. Keeps the secret-strength
// assertions isolated from the other validation rules.
func validBaseConfig(secret string) *Config {
	return &Config{
		Auth: AuthConfig{
			Secret:            secret,
			PasswordMinLength: 8,
		},
		Database: DatabaseConfig{
			URL: "postgresql://postgres:postgres@localhost:5432/novacron",
		},
		VM: VMConfig{
			HypervisorAddrs: []string{"localhost:9000"},
		},
		Logging: LoggingConfig{
			Level: "info",
		},
	}
}

func TestValidate_RejectsInsecureAuthSecret(t *testing.T) {
	cases := []struct {
		name    string
		secret  string
		wantErr bool
	}{
		{"empty", "", true},
		{"literal default", "changeme_in_production", true},
		{"too short", "abc1", true},
		{"strong secret", "0123456789abcdef0123456789ABCDEF", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validBaseConfig(tc.secret).Validate()
			if tc.wantErr && err == nil {
				t.Errorf("Validate() with secret %q = nil error, want error", tc.secret)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Validate() with secret %q = %v, want nil", tc.secret, err)
			}
		})
	}
}
