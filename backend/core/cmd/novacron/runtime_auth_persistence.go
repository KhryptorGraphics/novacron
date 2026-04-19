package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"
	coreauth "github.com/khryptorgraphics/novacron/backend/core/auth"
	_ "github.com/lib/pq"
)

const (
	defaultRuntimeFrontendURL        = "http://localhost:3000"
	defaultRuntimeTenantID           = "default"
	defaultRuntimeClusterID          = "cluster-local"
	defaultRuntimeGitHubRedirectURL  = "http://localhost:8090/api/auth/oauth/github/callback"
	defaultRuntimeOAuthRedirectRoute = "/dashboard"
)

type runtimeAuthConfig struct {
	Enabled          bool                            `yaml:"enabled"`
	FrontendURL      string                          `yaml:"frontend_url"`
	DefaultTenantID  string                          `yaml:"default_tenant_id"`
	DefaultClusterID string                          `yaml:"default_cluster_id"`
	OAuth            runtimeOAuthConfig              `yaml:"oauth"`
	Persistence      runtimeAuthPersistenceConfig    `yaml:"persistence"`
	Session          runtimeAuthSessionConfig        `yaml:"session"`
	Membership       runtimeMembershipPolicyConfig   `yaml:"membership"`
	Clustering       runtimeClusterPerformanceConfig `yaml:"clustering"`
}

type runtimeOAuthConfig struct {
	GitHub runtimeGitHubOAuthConfig `yaml:"github"`
}

type runtimeGitHubOAuthConfig struct {
	ClientID     string `yaml:"client_id"`
	ClientSecret string `yaml:"client_secret"`
	RedirectURL  string `yaml:"redirect_url"`
}

type runtimeAuthPersistenceConfig struct {
	PostgresURL string `yaml:"postgres_url"`
	RedisURL    string `yaml:"redis_url"`
}

type runtimeAuthSessionConfig struct {
	Transport string                   `yaml:"transport"`
	Cookie    runtimeAuthCookieConfig  `yaml:"cookie"`
}

type runtimeAuthCookieConfig struct {
	AccessTokenName  string `yaml:"access_token_name"`
	RefreshTokenName string `yaml:"refresh_token_name"`
	Domain           string `yaml:"domain"`
	Path             string `yaml:"path"`
	SameSite         string `yaml:"same_site"`
	Secure           bool   `yaml:"secure"`
}

type runtimeMembershipPolicyConfig struct {
	AutoAdmit    bool   `yaml:"auto_admit"`
	DefaultState string `yaml:"default_state"`
}

type runtimeClusterPerformanceConfig struct {
	ReevaluationIntervalSeconds int                          `yaml:"reevaluation_interval_seconds"`
	GrowthLatencyPenaltyMS      float64                      `yaml:"growth_latency_penalty_ms"`
	GrowthBandwidthPenaltyMBPS  float64                      `yaml:"growth_bandwidth_penalty_mbps"`
	Tiers                       []runtimeClusterTierConfig   `yaml:"tiers"`
}

type runtimeClusterTierConfig struct {
	Name             string  `yaml:"name"`
	MaxLatencyMS     float64 `yaml:"max_latency_ms"`
	MinBandwidthMBPS float64 `yaml:"min_bandwidth_mbps"`
}

type authConfigFile struct {
	Enabled          *bool                            `yaml:"enabled"`
	FrontendURL      string                           `yaml:"frontend_url"`
	DefaultTenantID  string                           `yaml:"default_tenant_id"`
	DefaultClusterID string                           `yaml:"default_cluster_id"`
	OAuth            *runtimeOAuthConfigFile          `yaml:"oauth"`
	Persistence      *runtimeAuthPersistenceConfig    `yaml:"persistence"`
	Session          *runtimeAuthSessionConfigFile    `yaml:"session"`
	Membership       *runtimeMembershipPolicyConfig   `yaml:"membership"`
	Clustering       *runtimeClusterPerformanceConfig `yaml:"clustering"`
}

type runtimeOAuthConfigFile struct {
	GitHub *runtimeGitHubOAuthConfig `yaml:"github"`
}

type runtimeAuthSessionConfigFile struct {
	Transport string                  `yaml:"transport"`
	Cookie    *runtimeAuthCookieConfig `yaml:"cookie"`
}

type runtimePersistedSession struct {
	ID                string
	UserID            string
	TenantID          string
	Token             string
	RefreshToken      string
	ExpiresAt         time.Time
	CreatedAt         time.Time
	LastAccessedAt    time.Time
	RevokedAt         *time.Time
	SelectedClusterID string
	ClientIP          string
	UserAgent         string
	Metadata          map[string]interface{}
}

type runtimeClusterRecord struct {
	ID                         string                 `json:"id"`
	Name                       string                 `json:"name"`
	Tier                       string                 `json:"tier"`
	InterconnectLatencyMS      float64                `json:"interconnectLatencyMs"`
	InterconnectBandwidthMBPS  float64                `json:"interconnectBandwidthMbps"`
	GrowthLatencyPenaltyMS     float64                `json:"growthLatencyPenaltyMs"`
	GrowthBandwidthPenaltyMBPS float64                `json:"growthBandwidthPenaltyMbps"`
	CurrentNodeCount           int                    `json:"currentNodeCount"`
	MaxSupportedNodeCount      int                    `json:"maxSupportedNodeCount"`
	PerformanceScore           float64                `json:"performanceScore"`
	GrowthState                string                 `json:"growthState"`
	FederationState            string                 `json:"federationState"`
	Degraded                   bool                   `json:"degraded"`
	LastEvaluatedAt            time.Time              `json:"lastEvaluatedAt"`
	Metadata                   map[string]interface{} `json:"metadata,omitempty"`
}

type runtimeClusterMembership struct {
	ID         string    `json:"id"`
	UserID     string    `json:"userId"`
	TenantID   string    `json:"tenantId"`
	ClusterID  string    `json:"clusterId"`
	State      string    `json:"state"`
	Role       string    `json:"role"`
	Source     string    `json:"source"`
	CreatedAt  time.Time `json:"createdAt"`
	UpdatedAt  time.Time `json:"updatedAt"`
	Selected   bool      `json:"selected,omitempty"`
	Cluster    *runtimeClusterRecord `json:"cluster,omitempty"`
}

type runtimeEdgeMetric struct {
	UserID          string    `json:"userId"`
	ClusterID       string    `json:"clusterId"`
	LatencyMS       float64   `json:"latencyMs"`
	BandwidthMBPS   float64   `json:"bandwidthMbps"`
	RecordedAt      time.Time `json:"recordedAt"`
}

type runtimeAuthPersistence struct {
	db             *sql.DB
	users          *runtimeUserStore
	tenants        *runtimeTenantStore
	sessions       *runtimeSessionStore
	clusters       *runtimeClusterStore
	memberships    *runtimeMembershipStore
	edgeMetrics    *runtimeEdgeMetricsStore
	revocations    coreauth.TokenRevocationService
}

func defaultRuntimeAuthConfig() runtimeAuthConfig {
	config := runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      getenvFirst("NOVACRON_AUTH_FRONTEND_URL", "NOVACRON_FRONTEND_URL"),
		DefaultTenantID:  getenvFirst("NOVACRON_AUTH_DEFAULT_TENANT_ID"),
		DefaultClusterID: getenvFirst("NOVACRON_CLUSTER_ID"),
		OAuth: runtimeOAuthConfig{
			GitHub: runtimeGitHubOAuthConfig{
				ClientID:     getenvFirst("NOVACRON_GITHUB_CLIENT_ID"),
				ClientSecret: getenvFirst("NOVACRON_GITHUB_CLIENT_SECRET"),
				RedirectURL:  getenvFirst("NOVACRON_GITHUB_REDIRECT_URL"),
			},
		},
		Persistence: runtimeAuthPersistenceConfig{
			PostgresURL: getenvFirst("NOVACRON_AUTH_POSTGRES_URL", "NOVACRON_DATABASE_URL"),
			RedisURL:    getenvFirst("NOVACRON_AUTH_REDIS_URL", "NOVACRON_REDIS_URL"),
		},
		Session: runtimeAuthSessionConfig{
			Transport: "dual",
			Cookie: runtimeAuthCookieConfig{
				AccessTokenName:  "novacron_access_token",
				RefreshTokenName: "novacron_refresh_token",
				Path:             "/",
				SameSite:         "lax",
			},
		},
		Membership: runtimeMembershipPolicyConfig{
			AutoAdmit:    true,
			DefaultState: "pending",
		},
		Clustering: runtimeClusterPerformanceConfig{
			ReevaluationIntervalSeconds: 30,
			GrowthLatencyPenaltyMS:      0.5,
			GrowthBandwidthPenaltyMBPS:  250,
			Tiers: []runtimeClusterTierConfig{
				{Name: "platinum", MaxLatencyMS: 2, MinBandwidthMBPS: 5000},
				{Name: "gold", MaxLatencyMS: 10, MinBandwidthMBPS: 1000},
				{Name: "silver", MaxLatencyMS: 30, MinBandwidthMBPS: 250},
				{Name: "bronze", MaxLatencyMS: 80, MinBandwidthMBPS: 50},
			},
		},
	}

	applyRuntimeAuthDefaults(&config)
	return config
}

func mergeRuntimeAuthConfig(config *runtimeAuthConfig, fileConfig *authConfigFile) {
	if config == nil || fileConfig == nil {
		return
	}

	if fileConfig.Enabled != nil {
		config.Enabled = *fileConfig.Enabled
	}
	if fileConfig.FrontendURL != "" {
		config.FrontendURL = fileConfig.FrontendURL
	}
	if fileConfig.DefaultTenantID != "" {
		config.DefaultTenantID = fileConfig.DefaultTenantID
	}
	if fileConfig.DefaultClusterID != "" {
		config.DefaultClusterID = fileConfig.DefaultClusterID
	}
	if fileConfig.OAuth != nil && fileConfig.OAuth.GitHub != nil {
		if fileConfig.OAuth.GitHub.ClientID != "" {
			config.OAuth.GitHub.ClientID = fileConfig.OAuth.GitHub.ClientID
		}
		if fileConfig.OAuth.GitHub.ClientSecret != "" {
			config.OAuth.GitHub.ClientSecret = fileConfig.OAuth.GitHub.ClientSecret
		}
		if fileConfig.OAuth.GitHub.RedirectURL != "" {
			config.OAuth.GitHub.RedirectURL = fileConfig.OAuth.GitHub.RedirectURL
		}
	}
	if fileConfig.Persistence != nil {
		if fileConfig.Persistence.PostgresURL != "" {
			config.Persistence.PostgresURL = fileConfig.Persistence.PostgresURL
		}
		if fileConfig.Persistence.RedisURL != "" {
			config.Persistence.RedisURL = fileConfig.Persistence.RedisURL
		}
	}
	if fileConfig.Session != nil {
		if fileConfig.Session.Transport != "" {
			config.Session.Transport = fileConfig.Session.Transport
		}
		if fileConfig.Session.Cookie != nil {
			if fileConfig.Session.Cookie.AccessTokenName != "" {
				config.Session.Cookie.AccessTokenName = fileConfig.Session.Cookie.AccessTokenName
			}
			if fileConfig.Session.Cookie.RefreshTokenName != "" {
				config.Session.Cookie.RefreshTokenName = fileConfig.Session.Cookie.RefreshTokenName
			}
			if fileConfig.Session.Cookie.Domain != "" {
				config.Session.Cookie.Domain = fileConfig.Session.Cookie.Domain
			}
			if fileConfig.Session.Cookie.Path != "" {
				config.Session.Cookie.Path = fileConfig.Session.Cookie.Path
			}
			if fileConfig.Session.Cookie.SameSite != "" {
				config.Session.Cookie.SameSite = fileConfig.Session.Cookie.SameSite
			}
			config.Session.Cookie.Secure = fileConfig.Session.Cookie.Secure
		}
	}
	if fileConfig.Membership != nil {
		config.Membership.AutoAdmit = fileConfig.Membership.AutoAdmit
		if fileConfig.Membership.DefaultState != "" {
			config.Membership.DefaultState = fileConfig.Membership.DefaultState
		}
	}
	if fileConfig.Clustering != nil {
		if fileConfig.Clustering.ReevaluationIntervalSeconds != 0 {
			config.Clustering.ReevaluationIntervalSeconds = fileConfig.Clustering.ReevaluationIntervalSeconds
		}
		if fileConfig.Clustering.GrowthLatencyPenaltyMS != 0 {
			config.Clustering.GrowthLatencyPenaltyMS = fileConfig.Clustering.GrowthLatencyPenaltyMS
		}
		if fileConfig.Clustering.GrowthBandwidthPenaltyMBPS != 0 {
			config.Clustering.GrowthBandwidthPenaltyMBPS = fileConfig.Clustering.GrowthBandwidthPenaltyMBPS
		}
		if len(fileConfig.Clustering.Tiers) > 0 {
			config.Clustering.Tiers = append([]runtimeClusterTierConfig(nil), fileConfig.Clustering.Tiers...)
		}
	}
}

func applyRuntimeAuthDefaults(config *runtimeAuthConfig) {
	if config == nil {
		return
	}
	if config.FrontendURL == "" {
		config.FrontendURL = defaultRuntimeFrontendURL
	}
	if config.DefaultTenantID == "" {
		config.DefaultTenantID = defaultRuntimeTenantID
	}
	if config.DefaultClusterID == "" {
		config.DefaultClusterID = defaultRuntimeClusterID
	}
	if config.OAuth.GitHub.RedirectURL == "" {
		config.OAuth.GitHub.RedirectURL = defaultRuntimeGitHubRedirectURL
	}
	if config.Session.Transport == "" {
		config.Session.Transport = "dual"
	}
	if config.Session.Cookie.AccessTokenName == "" {
		config.Session.Cookie.AccessTokenName = "novacron_access_token"
	}
	if config.Session.Cookie.RefreshTokenName == "" {
		config.Session.Cookie.RefreshTokenName = "novacron_refresh_token"
	}
	if config.Session.Cookie.Path == "" {
		config.Session.Cookie.Path = "/"
	}
	if config.Session.Cookie.SameSite == "" {
		config.Session.Cookie.SameSite = "lax"
	}
	if !config.Membership.AutoAdmit && config.Membership.DefaultState == "" {
		config.Membership.AutoAdmit = true
	}
	if config.Membership.DefaultState == "" {
		config.Membership.DefaultState = "pending"
	}
	if len(config.Clustering.Tiers) == 0 {
		config.Clustering = defaultRuntimeAuthConfig().Clustering
	}
	if config.Clustering.GrowthLatencyPenaltyMS <= 0 {
		config.Clustering.GrowthLatencyPenaltyMS = 0.5
	}
	if config.Clustering.GrowthBandwidthPenaltyMBPS <= 0 {
		config.Clustering.GrowthBandwidthPenaltyMBPS = 250
	}
}

func newRuntimeAuthPersistence(config runtimeAuthConfig) (*runtimeAuthPersistence, error) {
	persistence := &runtimeAuthPersistence{}
	if strings.TrimSpace(config.Persistence.PostgresURL) != "" {
		db, err := sql.Open("postgres", config.Persistence.PostgresURL)
		if err != nil {
			return nil, fmt.Errorf("open auth postgres database: %w", err)
		}
		if err := db.Ping(); err != nil {
			db.Close()
			return nil, fmt.Errorf("ping auth postgres database: %w", err)
		}
		if err := ensureRuntimeAuthSchema(db); err != nil {
			db.Close()
			return nil, fmt.Errorf("ensure runtime auth schema: %w", err)
		}
		persistence.db = db
	}

	persistence.users = newRuntimeUserStore(persistence.db)
	persistence.tenants = newRuntimeTenantStore(persistence.db)
	persistence.sessions = newRuntimeSessionStore(persistence.db)
	persistence.clusters = newRuntimeClusterStore(persistence.db, config.Clustering)
	persistence.memberships = newRuntimeMembershipStore(persistence.db)
	persistence.edgeMetrics = newRuntimeEdgeMetricsStore(persistence.db)

	if strings.TrimSpace(config.Persistence.RedisURL) != "" {
		revocations, err := coreauth.NewRedisTokenRevocationFromURL(config.Persistence.RedisURL)
		if err != nil {
			return nil, fmt.Errorf("initialize auth redis revocation backend: %w", err)
		}
		persistence.revocations = revocations
	} else {
		persistence.revocations = coreauth.NewInMemoryTokenRevocation()
	}

	if err := persistence.tenants.EnsureDefaultTenant(config.DefaultTenantID); err != nil {
		return nil, fmt.Errorf("ensure default tenant: %w", err)
	}
	if err := persistence.clusters.EnsureDefaultCluster(config.DefaultClusterID); err != nil {
		return nil, fmt.Errorf("ensure default cluster: %w", err)
	}

	return persistence, nil
}

func (p *runtimeAuthPersistence) Close() error {
	if p == nil || p.db == nil {
		return nil
	}
	return p.db.Close()
}

func ensureRuntimeAuthSchema(db *sql.DB) error {
	statements := []string{
		`ALTER TABLE sessions ALTER COLUMN token TYPE TEXT`,
		`ALTER TABLE sessions ALTER COLUMN refresh_token TYPE TEXT`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tenant_id UUID`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ DEFAULT NOW()`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS selected_cluster_id TEXT`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb`,
		`CREATE TABLE IF NOT EXISTS runtime_clusters (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			tier TEXT NOT NULL,
			interconnect_latency_ms DOUBLE PRECISION NOT NULL,
			interconnect_bandwidth_mbps DOUBLE PRECISION NOT NULL,
			growth_latency_penalty_ms DOUBLE PRECISION NOT NULL,
			growth_bandwidth_penalty_mbps DOUBLE PRECISION NOT NULL,
			current_node_count INTEGER NOT NULL DEFAULT 1,
			max_supported_node_count INTEGER NOT NULL DEFAULT 1,
			performance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
			growth_state TEXT NOT NULL DEFAULT 'expandable',
			federation_state TEXT NOT NULL DEFAULT 'cluster-local',
			degraded BOOLEAN NOT NULL DEFAULT FALSE,
			last_evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			metadata JSONB NOT NULL DEFAULT '{}'::jsonb
		)`,
		`CREATE TABLE IF NOT EXISTS runtime_cluster_memberships (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			tenant_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
			cluster_id TEXT NOT NULL REFERENCES runtime_clusters(id) ON DELETE CASCADE,
			state TEXT NOT NULL,
			role TEXT NOT NULL DEFAULT 'member',
			source TEXT NOT NULL DEFAULT 'runtime',
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			UNIQUE(user_id, cluster_id)
		)`,
		`CREATE TABLE IF NOT EXISTS runtime_user_cluster_edges (
			user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			cluster_id TEXT NOT NULL REFERENCES runtime_clusters(id) ON DELETE CASCADE,
			latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
			bandwidth_mbps DOUBLE PRECISION NOT NULL DEFAULT 0,
			recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY (user_id, cluster_id)
		)`,
	}

	for _, statement := range statements {
		if _, err := db.Exec(statement); err != nil {
			return err
		}
	}

	_, _ = db.Exec(`UPDATE sessions SET tenant_id = users.organization_id FROM users WHERE sessions.user_id = users.id AND sessions.tenant_id IS NULL`)
	return nil
}

type runtimeUserStore struct {
	db     *sql.DB
	memory *coreauth.UserMemoryStore
}

func newRuntimeUserStore(db *sql.DB) *runtimeUserStore {
	return &runtimeUserStore{db: db, memory: coreauth.NewUserMemoryStore()}
}

func (s *runtimeUserStore) Create(user *coreauth.User, password string) error {
	if s.db == nil {
		normalizeRuntimeUser(user)
		return s.memory.Create(user, password)
	}
	if user == nil {
		return fmt.Errorf("user is required")
	}
	normalizeRuntimeUser(user)
	salt, err := coreauth.GenerateRandomSalt()
	if err != nil {
		return fmt.Errorf("generate password salt: %w", err)
	}
	hash := coreauth.HashPassword(password, salt)
	user.PasswordSalt = salt
	user.PasswordHash = hash
	role := runtimeUserRole(user)
	orgID, err := runtimeLookupOrganizationUUID(s.db, user.TenantID)
	if err != nil {
		return err
	}

	_, err = s.db.Exec(
		`INSERT INTO users (id, email, username, password_hash, first_name, last_name, organization_id, role, status, last_login, created_at, updated_at)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
		user.ID,
		strings.ToLower(strings.TrimSpace(user.Email)),
		strings.ToLower(strings.TrimSpace(user.Username)),
		fmt.Sprintf("%s$%s", salt, hash),
		user.FirstName,
		user.LastName,
		orgID,
		role,
		runtimeDBUserStatus(user.Status),
		nullTime(user.LastLogin),
		zeroToNow(user.CreatedAt),
		zeroToNow(user.UpdatedAt),
	)
	return err
}

func (s *runtimeUserStore) Get(id string) (*coreauth.User, error) {
	if s.db == nil {
		return s.memory.Get(id)
	}
	return s.getByQuery(`SELECT u.id, u.email, u.username, u.password_hash, u.first_name, u.last_name, o.slug, u.role, u.status, u.last_login, u.created_at, u.updated_at FROM users u LEFT JOIN organizations o ON o.id = u.organization_id WHERE u.id = $1`, id)
}

func (s *runtimeUserStore) GetByUsername(username string) (*coreauth.User, error) {
	if s.db == nil {
		return s.memory.GetByUsername(username)
	}
	return s.getByQuery(`SELECT u.id, u.email, u.username, u.password_hash, u.first_name, u.last_name, o.slug, u.role, u.status, u.last_login, u.created_at, u.updated_at FROM users u LEFT JOIN organizations o ON o.id = u.organization_id WHERE lower(u.username) = lower($1)`, username)
}

func (s *runtimeUserStore) GetByEmail(email string) (*coreauth.User, error) {
	if s.db == nil {
		return s.memory.GetByEmail(email)
	}
	return s.getByQuery(`SELECT u.id, u.email, u.username, u.password_hash, u.first_name, u.last_name, o.slug, u.role, u.status, u.last_login, u.created_at, u.updated_at FROM users u LEFT JOIN organizations o ON o.id = u.organization_id WHERE lower(u.email) = lower($1)`, email)
}

func (s *runtimeUserStore) List(filter map[string]interface{}) ([]*coreauth.User, error) {
	if s.db == nil {
		return s.memory.List(filter)
	}
	rows, err := s.db.Query(`SELECT u.id, u.email, u.username, u.password_hash, u.first_name, u.last_name, o.slug, u.role, u.status, u.last_login, u.created_at, u.updated_at FROM users u LEFT JOIN organizations o ON o.id = u.organization_id ORDER BY u.email`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []*coreauth.User
	for rows.Next() {
		user, err := scanRuntimeUser(rows)
		if err != nil {
			return nil, err
		}
		users = append(users, user)
	}
	return users, rows.Err()
}

func (s *runtimeUserStore) Update(user *coreauth.User) error {
	if s.db == nil {
		return s.memory.Update(user)
	}
	normalizeRuntimeUser(user)
	orgID, err := runtimeLookupOrganizationUUID(s.db, user.TenantID)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(
		`UPDATE users SET email=$2, username=$3, first_name=$4, last_name=$5, organization_id=$6, role=$7, status=$8, last_login=$9, updated_at=$10 WHERE id=$1`,
		user.ID,
		strings.ToLower(strings.TrimSpace(user.Email)),
		strings.ToLower(strings.TrimSpace(user.Username)),
		user.FirstName,
		user.LastName,
		orgID,
		runtimeUserRole(user),
		runtimeDBUserStatus(user.Status),
		nullTime(user.LastLogin),
		time.Now().UTC(),
	)
	return err
}

func (s *runtimeUserStore) Delete(id string) error {
	if s.db == nil {
		return s.memory.Delete(id)
	}
	_, err := s.db.Exec(`DELETE FROM users WHERE id = $1`, id)
	return err
}

func (s *runtimeUserStore) SetPassword(id string, password string) error {
	if s.db == nil {
		return s.memory.SetPassword(id, password)
	}
	salt, err := coreauth.GenerateRandomSalt()
	if err != nil {
		return err
	}
	hash := coreauth.HashPassword(password, salt)
	_, err = s.db.Exec(`UPDATE users SET password_hash = $2, updated_at = $3 WHERE id = $1`, id, fmt.Sprintf("%s$%s", salt, hash), time.Now().UTC())
	return err
}

func (s *runtimeUserStore) VerifyPassword(id string, password string) (bool, error) {
	if s.db == nil {
		return s.memory.VerifyPassword(id, password)
	}
	var stored string
	if err := s.db.QueryRow(`SELECT password_hash FROM users WHERE id = $1`, id).Scan(&stored); err != nil {
		return false, err
	}
	parts := strings.SplitN(stored, "$", 2)
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid stored password hash")
	}
	return coreauth.VerifyHashedPassword(password, parts[1], parts[0]), nil
}

func (s *runtimeUserStore) UpdateStatus(id string, status coreauth.UserStatus) error {
	if s.db == nil {
		return s.memory.UpdateStatus(id, status)
	}
	_, err := s.db.Exec(`UPDATE users SET status = $2, updated_at = $3 WHERE id = $1`, id, runtimeDBUserStatus(status), time.Now().UTC())
	return err
}

func (s *runtimeUserStore) AddRole(userID string, roleID string) error {
	if s.db == nil {
		return s.memory.AddRole(userID, roleID)
	}
	roleID = runtimeNormalizeRoleID(roleID)
	_, err := s.db.Exec(`UPDATE users SET role = $2, updated_at = $3 WHERE id = $1`, userID, roleID, time.Now().UTC())
	return err
}

func (s *runtimeUserStore) RemoveRole(userID string, roleID string) error {
	if s.db == nil {
		return s.memory.RemoveRole(userID, roleID)
	}
	_, err := s.db.Exec(`UPDATE users SET role = 'viewer', updated_at = $2 WHERE id = $1 AND role = $3`, userID, time.Now().UTC(), runtimeNormalizeRoleID(roleID))
	return err
}

func (s *runtimeUserStore) GetRoles(userID string) ([]*coreauth.Role, error) {
	if s.db == nil {
		return s.memory.GetRoles(userID)
	}
	user, err := s.Get(userID)
	if err != nil {
		return nil, err
	}
	roles := make([]*coreauth.Role, 0, len(user.RoleIDs))
	for _, roleID := range user.RoleIDs {
		roleID = runtimeNormalizeRoleID(roleID)
		switch roleID {
		case "admin":
			roles = append(roles, coreauth.SystemRoles["admin"])
		case "operator":
			roles = append(roles, &coreauth.Role{
				ID:          "operator",
				Name:        "Operator",
				Description: "Operator role with cluster control permissions",
				IsSystem:    true,
				Permissions: []coreauth.Permission{
					{Resource: "cluster", Action: "read", Effect: "allow"},
					{Resource: "cluster", Action: "select", Effect: "allow"},
					{Resource: "federation", Action: "read", Effect: "allow"},
				},
			})
		default:
			roles = append(roles, &coreauth.Role{
				ID:          "viewer",
				Name:        "Viewer",
				Description: "Viewer role with read-only cluster access",
				IsSystem:    true,
				Permissions: []coreauth.Permission{
					{Resource: "cluster", Action: "read", Effect: "allow"},
					{Resource: "cluster", Action: "select", Effect: "allow"},
				},
			})
		}
	}
	return roles, nil
}

func (s *runtimeUserStore) getByQuery(query string, arg interface{}) (*coreauth.User, error) {
	row := s.db.QueryRow(query, arg)
	return scanRuntimeUser(row)
}

type runtimeTenantStore struct {
	db     *sql.DB
	memory *coreauth.TenantMemoryStore
}

func newRuntimeTenantStore(db *sql.DB) *runtimeTenantStore {
	return &runtimeTenantStore{db: db, memory: coreauth.NewTenantMemoryStore()}
}

func (s *runtimeTenantStore) EnsureDefaultTenant(id string) error {
	if id == "" {
		id = defaultRuntimeTenantID
	}
	if _, err := s.Get(id); err == nil {
		return nil
	}
	return s.Create(&coreauth.Tenant{
		ID:        id,
		Name:      strings.Title(strings.ReplaceAll(id, "-", " ")),
		Status:    coreauth.TenantStatusActive,
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	})
}

func (s *runtimeTenantStore) Create(tenant *coreauth.Tenant) error {
	if s.db == nil {
		return s.memory.Create(tenant)
	}
	if tenant == nil {
		return fmt.Errorf("tenant is required")
	}
	now := time.Now().UTC()
	if tenant.CreatedAt.IsZero() {
		tenant.CreatedAt = now
	}
	if tenant.UpdatedAt.IsZero() {
		tenant.UpdatedAt = now
	}
	_, err := s.db.Exec(`INSERT INTO organizations (id, name, slug, created_at, updated_at) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (slug) DO NOTHING`, uuid.NewString(), tenant.Name, tenant.ID, tenant.CreatedAt, tenant.UpdatedAt)
	return err
}

func (s *runtimeTenantStore) Get(id string) (*coreauth.Tenant, error) {
	if s.db == nil {
		return s.memory.Get(id)
	}
	var tenant coreauth.Tenant
	err := s.db.QueryRow(`SELECT slug, name, created_at, updated_at FROM organizations WHERE slug = $1`, id).Scan(&tenant.ID, &tenant.Name, &tenant.CreatedAt, &tenant.UpdatedAt)
	if err != nil {
		return nil, err
	}
	tenant.Status = coreauth.TenantStatusActive
	return &tenant, nil
}

func (s *runtimeTenantStore) List(filter map[string]interface{}) ([]*coreauth.Tenant, error) {
	if s.db == nil {
		return s.memory.List(filter)
	}
	rows, err := s.db.Query(`SELECT slug, name, created_at, updated_at FROM organizations ORDER BY slug`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tenants []*coreauth.Tenant
	for rows.Next() {
		var tenant coreauth.Tenant
		if err := rows.Scan(&tenant.ID, &tenant.Name, &tenant.CreatedAt, &tenant.UpdatedAt); err != nil {
			return nil, err
		}
		tenant.Status = coreauth.TenantStatusActive
		tenants = append(tenants, &tenant)
	}
	return tenants, rows.Err()
}

func (s *runtimeTenantStore) Update(tenant *coreauth.Tenant) error {
	if s.db == nil {
		return s.memory.Update(tenant)
	}
	_, err := s.db.Exec(`UPDATE organizations SET name = $2, updated_at = $3 WHERE slug = $1`, tenant.ID, tenant.Name, time.Now().UTC())
	return err
}

func (s *runtimeTenantStore) Delete(id string) error {
	if s.db == nil {
		return s.memory.Delete(id)
	}
	_, err := s.db.Exec(`DELETE FROM organizations WHERE slug = $1`, id)
	return err
}

func (s *runtimeTenantStore) UpdateStatus(id string, status coreauth.TenantStatus) error {
	if s.db == nil {
		return s.memory.UpdateStatus(id, status)
	}
	return nil
}

func (s *runtimeTenantStore) SetResourceQuota(id string, resource string, quota int64) error {
	if s.db == nil {
		return s.memory.SetResourceQuota(id, resource, quota)
	}
	return nil
}

func (s *runtimeTenantStore) GetResourceQuota(id string, resource string) (int64, error) {
	if s.db == nil {
		return s.memory.GetResourceQuota(id, resource)
	}
	return coreauth.DefaultResourceQuotas[resource], nil
}

func (s *runtimeTenantStore) GetResourceQuotas(id string) (map[string]int64, error) {
	if s.db == nil {
		return s.memory.GetResourceQuotas(id)
	}
	return coreauth.DefaultResourceQuotas, nil
}

type runtimeSessionStore struct {
	db      *sql.DB
	records map[string]*runtimePersistedSession
}

func newRuntimeSessionStore(db *sql.DB) *runtimeSessionStore {
	return &runtimeSessionStore{db: db, records: make(map[string]*runtimePersistedSession)}
}

func (s *runtimeSessionStore) Upsert(session *runtimePersistedSession) error {
	if session == nil {
		return fmt.Errorf("session is required")
	}
	if s.db == nil {
		s.records[session.ID] = cloneRuntimeSession(session)
		return nil
	}
	metadata, err := json.Marshal(session.Metadata)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(
		`INSERT INTO sessions (id, user_id, tenant_id, token, refresh_token, ip_address, user_agent, expires_at, created_at, last_accessed_at, revoked_at, selected_cluster_id, metadata)
		 VALUES ($1,$2,$3,$4,$5,NULLIF($6,'')::inet,$7,$8,$9,$10,$11,$12,$13)
		 ON CONFLICT (id) DO UPDATE SET token = EXCLUDED.token, refresh_token = EXCLUDED.refresh_token, expires_at = EXCLUDED.expires_at, last_accessed_at = EXCLUDED.last_accessed_at, revoked_at = EXCLUDED.revoked_at, selected_cluster_id = EXCLUDED.selected_cluster_id, metadata = EXCLUDED.metadata, user_agent = EXCLUDED.user_agent`,
		session.ID,
		session.UserID,
		nullUUID(session.TenantID),
		session.Token,
		session.RefreshToken,
		session.ClientIP,
		session.UserAgent,
		session.ExpiresAt,
		zeroToNow(session.CreatedAt),
		zeroToNow(session.LastAccessedAt),
		session.RevokedAt,
		nullString(session.SelectedClusterID),
		string(metadata),
	)
	return err
}

func (s *runtimeSessionStore) Get(id string) (*runtimePersistedSession, error) {
	if s.db == nil {
		session, ok := s.records[id]
		if !ok {
			return nil, fmt.Errorf("session not found")
		}
		return cloneRuntimeSession(session), nil
	}
	row := s.db.QueryRow(`SELECT id, user_id, COALESCE(o.slug,''), token, COALESCE(refresh_token,''), expires_at, created_at, COALESCE(last_accessed_at, created_at), revoked_at, COALESCE(selected_cluster_id,''), COALESCE(host(ip_address),'') as ip_address, COALESCE(user_agent,''), metadata FROM sessions s LEFT JOIN organizations o ON o.id = s.tenant_id WHERE s.id = $1`, id)
	return scanRuntimeSession(row)
}

func (s *runtimeSessionStore) GetByRefreshToken(refreshToken string) (*runtimePersistedSession, error) {
	if s.db == nil {
		for _, session := range s.records {
			if session.RefreshToken == refreshToken {
				return cloneRuntimeSession(session), nil
			}
		}
		return nil, fmt.Errorf("session not found")
	}
	row := s.db.QueryRow(`SELECT id, user_id, COALESCE(o.slug,''), token, COALESCE(refresh_token,''), expires_at, created_at, COALESCE(last_accessed_at, created_at), revoked_at, COALESCE(selected_cluster_id,''), COALESCE(host(ip_address),''), COALESCE(user_agent,''), metadata FROM sessions s LEFT JOIN organizations o ON o.id = s.tenant_id WHERE s.refresh_token = $1`, refreshToken)
	return scanRuntimeSession(row)
}

func (s *runtimeSessionStore) ListByUser(userID string) ([]runtimePersistedSession, error) {
	if s.db == nil {
		results := make([]runtimePersistedSession, 0)
		for _, session := range s.records {
			if session.UserID == userID {
				results = append(results, *cloneRuntimeSession(session))
			}
		}
		return results, nil
	}
	rows, err := s.db.Query(`SELECT id, user_id, COALESCE(o.slug,''), token, COALESCE(refresh_token,''), expires_at, created_at, COALESCE(last_accessed_at, created_at), revoked_at, COALESCE(selected_cluster_id,''), COALESCE(host(ip_address),''), COALESCE(user_agent,''), metadata FROM sessions s LEFT JOIN organizations o ON o.id = s.tenant_id WHERE user_id = $1 ORDER BY created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]runtimePersistedSession, 0)
	for rows.Next() {
		session, err := scanRuntimeSession(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, *session)
	}
	return results, rows.Err()
}

func (s *runtimeSessionStore) Revoke(id string) error {
	now := time.Now().UTC()
	if s.db == nil {
		if session, ok := s.records[id]; ok {
			session.RevokedAt = &now
			s.records[id] = session
			return nil
		}
		return fmt.Errorf("session not found")
	}
	_, err := s.db.Exec(`UPDATE sessions SET revoked_at = $2 WHERE id = $1`, id, now)
	return err
}

func (s *runtimeSessionStore) SetSelectedCluster(id string, clusterID string) error {
	if s.db == nil {
		session, ok := s.records[id]
		if !ok {
			return fmt.Errorf("session not found")
		}
		session.SelectedClusterID = clusterID
		session.LastAccessedAt = time.Now().UTC()
		s.records[id] = session
		return nil
	}
	_, err := s.db.Exec(`UPDATE sessions SET selected_cluster_id = $2, last_accessed_at = $3 WHERE id = $1`, id, nullString(clusterID), time.Now().UTC())
	return err
}

type runtimeClusterStore struct {
	db      *sql.DB
	config  runtimeClusterPerformanceConfig
	records map[string]*runtimeClusterRecord
}

func newRuntimeClusterStore(db *sql.DB, config runtimeClusterPerformanceConfig) *runtimeClusterStore {
	return &runtimeClusterStore{db: db, config: config, records: make(map[string]*runtimeClusterRecord)}
}

func (s *runtimeClusterStore) EnsureDefaultCluster(clusterID string) error {
	if clusterID == "" {
		clusterID = defaultRuntimeClusterID
	}
	if _, err := s.Get(clusterID); err == nil {
		return nil
	}
	cluster := &runtimeClusterRecord{
		ID:                         clusterID,
		Name:                       "Local Cluster",
		InterconnectLatencyMS:      1,
		InterconnectBandwidthMBPS:  10000,
		GrowthLatencyPenaltyMS:     s.config.GrowthLatencyPenaltyMS,
		GrowthBandwidthPenaltyMBPS: s.config.GrowthBandwidthPenaltyMBPS,
		CurrentNodeCount:           1,
		FederationState:            "cluster-local",
	}
	s.evaluate(cluster)
	return s.Upsert(cluster)
}

func (s *runtimeClusterStore) Get(id string) (*runtimeClusterRecord, error) {
	if s.db == nil {
		cluster, ok := s.records[id]
		if !ok {
			return nil, fmt.Errorf("cluster not found")
		}
		cloned := *cluster
		return &cloned, nil
	}
	row := s.db.QueryRow(`SELECT id, name, tier, interconnect_latency_ms, interconnect_bandwidth_mbps, growth_latency_penalty_ms, growth_bandwidth_penalty_mbps, current_node_count, max_supported_node_count, performance_score, growth_state, federation_state, degraded, last_evaluated_at, metadata FROM runtime_clusters WHERE id = $1`, id)
	return scanRuntimeCluster(row)
}

func (s *runtimeClusterStore) List() ([]runtimeClusterRecord, error) {
	if s.db == nil {
		results := make([]runtimeClusterRecord, 0, len(s.records))
		for _, cluster := range s.records {
			results = append(results, *cluster)
		}
		return results, nil
	}
	rows, err := s.db.Query(`SELECT id, name, tier, interconnect_latency_ms, interconnect_bandwidth_mbps, growth_latency_penalty_ms, growth_bandwidth_penalty_mbps, current_node_count, max_supported_node_count, performance_score, growth_state, federation_state, degraded, last_evaluated_at, metadata FROM runtime_clusters ORDER BY performance_score DESC, id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]runtimeClusterRecord, 0)
	for rows.Next() {
		cluster, err := scanRuntimeCluster(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, *cluster)
	}
	return results, rows.Err()
}

func (s *runtimeClusterStore) Upsert(cluster *runtimeClusterRecord) error {
	if cluster == nil {
		return fmt.Errorf("cluster is required")
	}
	s.evaluate(cluster)
	if s.db == nil {
		cloned := *cluster
		s.records[cluster.ID] = &cloned
		return nil
	}
	metadata, err := json.Marshal(cluster.Metadata)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(`INSERT INTO runtime_clusters (id, name, tier, interconnect_latency_ms, interconnect_bandwidth_mbps, growth_latency_penalty_ms, growth_bandwidth_penalty_mbps, current_node_count, max_supported_node_count, performance_score, growth_state, federation_state, degraded, last_evaluated_at, metadata)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
		ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, tier=EXCLUDED.tier, interconnect_latency_ms=EXCLUDED.interconnect_latency_ms, interconnect_bandwidth_mbps=EXCLUDED.interconnect_bandwidth_mbps, growth_latency_penalty_ms=EXCLUDED.growth_latency_penalty_ms, growth_bandwidth_penalty_mbps=EXCLUDED.growth_bandwidth_penalty_mbps, current_node_count=EXCLUDED.current_node_count, max_supported_node_count=EXCLUDED.max_supported_node_count, performance_score=EXCLUDED.performance_score, growth_state=EXCLUDED.growth_state, federation_state=EXCLUDED.federation_state, degraded=EXCLUDED.degraded, last_evaluated_at=EXCLUDED.last_evaluated_at, metadata=EXCLUDED.metadata`,
		cluster.ID, cluster.Name, cluster.Tier, cluster.InterconnectLatencyMS, cluster.InterconnectBandwidthMBPS, cluster.GrowthLatencyPenaltyMS, cluster.GrowthBandwidthPenaltyMBPS, cluster.CurrentNodeCount, cluster.MaxSupportedNodeCount, cluster.PerformanceScore, cluster.GrowthState, cluster.FederationState, cluster.Degraded, cluster.LastEvaluatedAt, string(metadata))
	return err
}

func (s *runtimeClusterStore) evaluate(cluster *runtimeClusterRecord) {
	if cluster.Name == "" {
		cluster.Name = cluster.ID
	}
	if cluster.GrowthLatencyPenaltyMS <= 0 {
		cluster.GrowthLatencyPenaltyMS = s.config.GrowthLatencyPenaltyMS
	}
	if cluster.GrowthBandwidthPenaltyMBPS <= 0 {
		cluster.GrowthBandwidthPenaltyMBPS = s.config.GrowthBandwidthPenaltyMBPS
	}
	if cluster.CurrentNodeCount <= 0 {
		cluster.CurrentNodeCount = 1
	}
	selectedTier := s.config.Tiers[len(s.config.Tiers)-1]
	for _, tier := range s.config.Tiers {
		if cluster.InterconnectLatencyMS <= tier.MaxLatencyMS && cluster.InterconnectBandwidthMBPS >= tier.MinBandwidthMBPS {
			selectedTier = tier
			break
		}
	}
	cluster.Tier = selectedTier.Name
	cluster.Degraded = cluster.InterconnectLatencyMS > selectedTier.MaxLatencyMS || cluster.InterconnectBandwidthMBPS < selectedTier.MinBandwidthMBPS

	latencyHeadroom := math.Max(0, selectedTier.MaxLatencyMS-cluster.InterconnectLatencyMS)
	bandwidthHeadroom := math.Max(0, cluster.InterconnectBandwidthMBPS-selectedTier.MinBandwidthMBPS)
	latencyCapacity := cluster.CurrentNodeCount
	if cluster.GrowthLatencyPenaltyMS > 0 {
		latencyCapacity += int(math.Floor(latencyHeadroom / cluster.GrowthLatencyPenaltyMS))
	}
	bandwidthCapacity := cluster.CurrentNodeCount
	if cluster.GrowthBandwidthPenaltyMBPS > 0 {
		bandwidthCapacity += int(math.Floor(bandwidthHeadroom / cluster.GrowthBandwidthPenaltyMBPS))
	}
	if bandwidthCapacity < latencyCapacity {
		cluster.MaxSupportedNodeCount = maxInt(1, bandwidthCapacity)
	} else {
		cluster.MaxSupportedNodeCount = maxInt(1, latencyCapacity)
	}
	if cluster.MaxSupportedNodeCount < cluster.CurrentNodeCount {
		cluster.MaxSupportedNodeCount = cluster.CurrentNodeCount
	}
	cluster.PerformanceScore = math.Round((((selectedTier.MaxLatencyMS-cluster.InterconnectLatencyMS)*10)+(cluster.InterconnectBandwidthMBPS/100))/10*100) / 100
	if cluster.CurrentNodeCount >= cluster.MaxSupportedNodeCount {
		cluster.GrowthState = "at_capacity"
	} else {
		cluster.GrowthState = "expandable"
	}
	if cluster.FederationState == "" {
		cluster.FederationState = "cluster-local"
	}
	cluster.LastEvaluatedAt = time.Now().UTC()
}

type runtimeMembershipStore struct {
	db      *sql.DB
	records map[string]runtimeClusterMembership
}

func newRuntimeMembershipStore(db *sql.DB) *runtimeMembershipStore {
	return &runtimeMembershipStore{db: db, records: make(map[string]runtimeClusterMembership)}
}

func (s *runtimeMembershipStore) Upsert(membership runtimeClusterMembership) (runtimeClusterMembership, error) {
	if membership.ID == "" {
		membership.ID = uuid.NewString()
	}
	if membership.CreatedAt.IsZero() {
		membership.CreatedAt = time.Now().UTC()
	}
	membership.UpdatedAt = time.Now().UTC()
	if s.db == nil {
		s.records[membership.UserID+"::"+membership.ClusterID] = membership
		return membership, nil
	}
	tenantID, err := runtimeLookupOrganizationUUID(s.db, membership.TenantID)
	if err != nil {
		return membership, err
	}
	var id string
	err = s.db.QueryRow(
		`INSERT INTO runtime_cluster_memberships (user_id, tenant_id, cluster_id, state, role, source, created_at, updated_at)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		 ON CONFLICT (user_id, cluster_id)
		 DO UPDATE SET tenant_id=EXCLUDED.tenant_id, state=EXCLUDED.state, role=EXCLUDED.role, source=EXCLUDED.source, updated_at=EXCLUDED.updated_at
		 RETURNING id`,
		membership.UserID, tenantID, membership.ClusterID, membership.State, membership.Role, membership.Source, membership.CreatedAt, membership.UpdatedAt,
	).Scan(&id)
	if err != nil {
		return membership, err
	}
	membership.ID = id
	return membership, nil
}

func (s *runtimeMembershipStore) ListByUser(userID string) ([]runtimeClusterMembership, error) {
	if s.db == nil {
		results := make([]runtimeClusterMembership, 0)
		for _, membership := range s.records {
			if membership.UserID == userID {
				results = append(results, membership)
			}
		}
		return results, nil
	}
	rows, err := s.db.Query(`SELECT m.id, m.user_id, COALESCE(o.slug,''), m.cluster_id, m.state, m.role, m.source, m.created_at, m.updated_at FROM runtime_cluster_memberships m LEFT JOIN organizations o ON o.id = m.tenant_id WHERE m.user_id = $1 ORDER BY m.created_at`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]runtimeClusterMembership, 0)
	for rows.Next() {
		var membership runtimeClusterMembership
		if err := rows.Scan(&membership.ID, &membership.UserID, &membership.TenantID, &membership.ClusterID, &membership.State, &membership.Role, &membership.Source, &membership.CreatedAt, &membership.UpdatedAt); err != nil {
			return nil, err
		}
		results = append(results, membership)
	}
	return results, rows.Err()
}

type runtimeEdgeMetricsStore struct {
	db      *sql.DB
	records map[string]runtimeEdgeMetric
}

func newRuntimeEdgeMetricsStore(db *sql.DB) *runtimeEdgeMetricsStore {
	return &runtimeEdgeMetricsStore{db: db, records: make(map[string]runtimeEdgeMetric)}
}

func (s *runtimeEdgeMetricsStore) Upsert(metric runtimeEdgeMetric) error {
	metric.RecordedAt = time.Now().UTC()
	if s.db == nil {
		s.records[metric.UserID+"::"+metric.ClusterID] = metric
		return nil
	}
	_, err := s.db.Exec(`INSERT INTO runtime_user_cluster_edges (user_id, cluster_id, latency_ms, bandwidth_mbps, recorded_at) VALUES ($1,$2,$3,$4,$5)
		ON CONFLICT (user_id, cluster_id) DO UPDATE SET latency_ms=EXCLUDED.latency_ms, bandwidth_mbps=EXCLUDED.bandwidth_mbps, recorded_at=EXCLUDED.recorded_at`,
		metric.UserID, metric.ClusterID, metric.LatencyMS, metric.BandwidthMBPS, metric.RecordedAt)
	return err
}

func (s *runtimeEdgeMetricsStore) ListByUser(userID string) ([]runtimeEdgeMetric, error) {
	if s.db == nil {
		results := make([]runtimeEdgeMetric, 0)
		for _, metric := range s.records {
			if metric.UserID == userID {
				results = append(results, metric)
			}
		}
		return results, nil
	}
	rows, err := s.db.Query(`SELECT user_id, cluster_id, latency_ms, bandwidth_mbps, recorded_at FROM runtime_user_cluster_edges WHERE user_id = $1 ORDER BY recorded_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]runtimeEdgeMetric, 0)
	for rows.Next() {
		var metric runtimeEdgeMetric
		if err := rows.Scan(&metric.UserID, &metric.ClusterID, &metric.LatencyMS, &metric.BandwidthMBPS, &metric.RecordedAt); err != nil {
			return nil, err
		}
		results = append(results, metric)
	}
	return results, rows.Err()
}

func scanRuntimeUser(scanner interface{ Scan(dest ...interface{}) error }) (*coreauth.User, error) {
	var (
		id        string
		email     string
		username  string
		password  string
		firstName sql.NullString
		lastName  sql.NullString
		tenantID  sql.NullString
		role      string
		status    string
		lastLogin sql.NullTime
		createdAt time.Time
		updatedAt time.Time
	)
	if err := scanner.Scan(&id, &email, &username, &password, &firstName, &lastName, &tenantID, &role, &status, &lastLogin, &createdAt, &updatedAt); err != nil {
		return nil, err
	}
	user := &coreauth.User{
		ID:        id,
		Email:     email,
		Username:  username,
		FirstName: firstName.String,
		LastName:  lastName.String,
		TenantID:  tenantID.String,
		RoleIDs:   []string{runtimeNormalizeRoleID(role)},
		Status:    runtimeCoreUserStatus(status),
		CreatedAt: createdAt,
		UpdatedAt: updatedAt,
		Metadata:  map[string]interface{}{},
	}
	if lastLogin.Valid {
		user.LastLogin = lastLogin.Time
	}
	if parts := strings.SplitN(password, "$", 2); len(parts) == 2 {
		user.PasswordSalt = parts[0]
		user.PasswordHash = parts[1]
	}
	return user, nil
}

func scanRuntimeSession(scanner interface{ Scan(dest ...interface{}) error }) (*runtimePersistedSession, error) {
	var (
		session         runtimePersistedSession
		revokedAt       sql.NullTime
		selectedCluster sql.NullString
		metadataRaw     []byte
	)
	if err := scanner.Scan(&session.ID, &session.UserID, &session.TenantID, &session.Token, &session.RefreshToken, &session.ExpiresAt, &session.CreatedAt, &session.LastAccessedAt, &revokedAt, &selectedCluster, &session.ClientIP, &session.UserAgent, &metadataRaw); err != nil {
		return nil, err
	}
	session.SelectedClusterID = selectedCluster.String
	if revokedAt.Valid {
		session.RevokedAt = &revokedAt.Time
	}
	if len(metadataRaw) > 0 {
		_ = json.Unmarshal(metadataRaw, &session.Metadata)
	}
	if session.Metadata == nil {
		session.Metadata = map[string]interface{}{}
	}
	return &session, nil
}

func scanRuntimeCluster(scanner interface{ Scan(dest ...interface{}) error }) (*runtimeClusterRecord, error) {
	var (
		cluster     runtimeClusterRecord
		metadataRaw []byte
	)
	if err := scanner.Scan(&cluster.ID, &cluster.Name, &cluster.Tier, &cluster.InterconnectLatencyMS, &cluster.InterconnectBandwidthMBPS, &cluster.GrowthLatencyPenaltyMS, &cluster.GrowthBandwidthPenaltyMBPS, &cluster.CurrentNodeCount, &cluster.MaxSupportedNodeCount, &cluster.PerformanceScore, &cluster.GrowthState, &cluster.FederationState, &cluster.Degraded, &cluster.LastEvaluatedAt, &metadataRaw); err != nil {
		return nil, err
	}
	if len(metadataRaw) > 0 {
		_ = json.Unmarshal(metadataRaw, &cluster.Metadata)
	}
	if cluster.Metadata == nil {
		cluster.Metadata = map[string]interface{}{}
	}
	return &cluster, nil
}

func runtimeLookupOrganizationUUID(db *sql.DB, slug string) (sql.NullString, error) {
	if strings.TrimSpace(slug) == "" {
		return sql.NullString{}, nil
	}
	var id string
	if err := db.QueryRow(`SELECT id FROM organizations WHERE slug = $1`, slug).Scan(&id); err != nil {
		return sql.NullString{}, err
	}
	return sql.NullString{String: id, Valid: true}, nil
}

func runtimeNormalizeRoleID(roleID string) string {
	switch strings.ToLower(strings.TrimSpace(roleID)) {
	case "admin":
		return "admin"
	case "operator":
		return "operator"
	case "user", "member", "readonly", "viewer":
		return "viewer"
	default:
		return "viewer"
	}
}

func runtimeUserRole(user *coreauth.User) string {
	if user == nil || len(user.RoleIDs) == 0 {
		return "viewer"
	}
	return runtimeNormalizeRoleID(user.RoleIDs[0])
}

func normalizeRuntimeUser(user *coreauth.User) {
	if user == nil {
		return
	}
	if _, err := uuid.Parse(user.ID); err != nil {
		user.ID = uuid.NewString()
	}
	if user.Username == "" {
		user.Username = strings.ToLower(strings.TrimSpace(user.Email))
	}
	if user.Status == "" {
		user.Status = coreauth.UserStatusActive
	}
	if len(user.RoleIDs) == 0 {
		user.RoleIDs = []string{"viewer"}
	}
	user.RoleIDs[0] = runtimeNormalizeRoleID(user.RoleIDs[0])
	if user.Metadata == nil {
		user.Metadata = map[string]interface{}{}
	}
}

func runtimeCoreUserStatus(status string) coreauth.UserStatus {
	switch strings.ToLower(status) {
	case "active":
		return coreauth.UserStatusActive
	case "pending":
		return coreauth.UserStatusPending
	case "suspended":
		return coreauth.UserStatusLocked
	default:
		return coreauth.UserStatusInactive
	}
}

func runtimeDBUserStatus(status coreauth.UserStatus) string {
	switch status {
	case coreauth.UserStatusActive:
		return "active"
	case coreauth.UserStatusPending:
		return "pending"
	case coreauth.UserStatusLocked:
		return "suspended"
	default:
		return "inactive"
	}
}

func zeroToNow(value time.Time) time.Time {
	if value.IsZero() {
		return time.Now().UTC()
	}
	return value.UTC()
}

func nullTime(value time.Time) interface{} {
	if value.IsZero() {
		return nil
	}
	return value.UTC()
}

func nullString(value string) interface{} {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func nullUUID(slug string) interface{} {
	if strings.TrimSpace(slug) == "" {
		return nil
	}
	return slug
}

func cloneRuntimeSession(session *runtimePersistedSession) *runtimePersistedSession {
	if session == nil {
		return nil
	}
	cloned := *session
	if session.Metadata != nil {
		cloned.Metadata = make(map[string]interface{}, len(session.Metadata))
		for key, value := range session.Metadata {
			cloned.Metadata[key] = value
		}
	}
	return &cloned
}
