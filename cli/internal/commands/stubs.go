package commands

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
	"github.com/spf13/cobra"
)

// Stub commands - to be implemented

func NewLoginCommand() *cobra.Command {
	var (
		loginCluster  string
		server        string
		email         string
		password      string
		namespace     string
		loginInsecure bool
	)

	cmd := &cobra.Command{
		Use:   "login",
		Short: "Authenticate with a NovaCron cluster",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			cluster := strings.TrimSpace(loginCluster)
			if cluster == "" {
				cluster = strings.TrimSpace(clusterName)
			}
			if cluster == "" {
				current, err := manager.GetCurrentCluster()
				if err == nil {
					cluster = current.Name
				}
			}
			if cluster == "" {
				return fmt.Errorf("cluster is required")
			}

			if strings.TrimSpace(server) == "" {
				if existing, ok := manager.Get().Clusters[cluster]; ok {
					server = existing.Server
					if namespace == "" {
						namespace = existing.Namespace
					}
					loginInsecure = loginInsecure || existing.Insecure
				}
			}
			server = strings.TrimRight(strings.TrimSpace(server), "/")
			if server == "" {
				return fmt.Errorf("server is required")
			}

			email = strings.TrimSpace(firstNonEmpty(email, os.Getenv("NOVACRON_EMAIL")))
			password = firstNonEmpty(password, os.Getenv("NOVACRON_PASSWORD"))
			if email == "" {
				return fmt.Errorf("email is required")
			}
			if password == "" {
				return fmt.Errorf("password is required")
			}

			token, err := loginToCluster(server, email, password, loginInsecure || insecure)
			if err != nil {
				return err
			}

			if namespace == "" {
				namespace = "default"
			}

			if err := manager.AddCluster(config.Cluster{
				Name:      cluster,
				Server:    server,
				Insecure:  loginInsecure || insecure,
				Namespace: namespace,
				AuthType:  "token",
				AuthData:  "token-store",
			}); err != nil {
				return err
			}
			if err := manager.SetCurrentCluster(cluster); err != nil {
				return err
			}

			token.RefreshURL = authEndpoint(server, "/api/auth/refresh")
			store, err := auth.NewTokenStore()
			if err != nil {
				return err
			}
			if err := store.Save(cluster, token); err != nil {
				return err
			}

			fmt.Fprintf(cmd.OutOrStdout(), "Logged in to cluster %s\n", cluster)
			return nil
		},
	}

	cmd.Flags().StringVar(&loginCluster, "cluster", "", "cluster name")
	cmd.Flags().StringVar(&server, "server", "", "NovaCron API server URL")
	cmd.Flags().StringVar(&email, "email", "", "login email")
	cmd.Flags().StringVar(&password, "password", "", "login password")
	cmd.Flags().StringVar(&namespace, "namespace", "", "default namespace")
	cmd.Flags().BoolVar(&loginInsecure, "insecure", false, "skip TLS certificate verification")

	return cmd
}

func loginToCluster(server string, email string, password string, allowInsecure bool) (*auth.TokenAuth, error) {
	body, err := json.Marshal(map[string]string{
		"email":    email,
		"password": password,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal login request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, authEndpoint(server, "/api/auth/login"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create login request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	if allowInsecure {
		client.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send login request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("login failed with status %d", resp.StatusCode)
	}

	var loginResp struct {
		Token        string    `json:"token"`
		RefreshToken string    `json:"refreshToken"`
		ExpiresAt    time.Time `json:"expiresAt"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&loginResp); err != nil {
		return nil, fmt.Errorf("decode login response: %w", err)
	}
	if loginResp.Token == "" {
		return nil, fmt.Errorf("login response missing token")
	}
	if loginResp.ExpiresAt.IsZero() {
		return nil, fmt.Errorf("login response missing expiresAt")
	}

	return &auth.TokenAuth{
		Token:        loginResp.Token,
		RefreshToken: loginResp.RefreshToken,
		ExpiresAt:    loginResp.ExpiresAt,
	}, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func NewConfigCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "config",
		Short: "Manage CLI configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("config command not yet implemented")
		},
	}
}

func NewNodeCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "node",
		Short: "Manage cluster nodes",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("node command not yet implemented")
		},
	}
}

func NewClusterCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "cluster",
		Short: "Manage NovaCron clusters",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("cluster command not yet implemented")
		},
	}
}

func NewMigrateCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "migrate",
		Short: "Migrate VMs between nodes",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("migrate command not yet implemented")
		},
	}
}

func NewSnapshotCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "snapshot",
		Short: "Manage VM snapshots",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("snapshot command not yet implemented")
		},
	}
}

func NewMonitorCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "monitor",
		Short: "Monitor cluster resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("monitor command not yet implemented")
		},
	}
}

func NewLogsCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "logs",
		Short: "View VM logs",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("logs command not yet implemented")
		},
	}
}

func NewExecCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "exec",
		Short: "Execute command in VM",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("exec command not yet implemented")
		},
	}
}

func NewCopyCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "copy",
		Short: "Copy files to/from VMs",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("copy command not yet implemented")
		},
	}
}

func NewPortForwardCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "port-forward",
		Short: "Forward ports to/from VMs",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("port-forward command not yet implemented")
		},
	}
}

func NewTopCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "top",
		Short: "Display resource usage",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("top command not yet implemented")
		},
	}
}

func NewApplyCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "apply",
		Short: "Apply configuration from file",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("apply command not yet implemented")
		},
	}
}

func NewDeleteCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "delete",
		Short: "Delete resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("delete command not yet implemented")
		},
	}
}

func NewDescribeCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "describe",
		Short: "Describe resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("describe command not yet implemented")
		},
	}
}

func NewGetCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "get",
		Short: "Get resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("get command not yet implemented")
		},
	}
}

func NewCreateCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "create",
		Short: "Create resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("create command not yet implemented")
		},
	}
}

func NewUpdateCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "update",
		Short: "Update resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("update command not yet implemented")
		},
	}
}

func NewScaleCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "scale",
		Short: "Scale VM resources",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("scale command not yet implemented")
		},
	}
}

func NewRolloutCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "rollout",
		Short: "Manage rollouts",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("rollout command not yet implemented")
		},
	}
}

func NewCompletionCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "completion",
		Short: "Generate shell completion scripts",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("completion command not yet implemented")
		},
	}
}

func NewPluginCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "plugin",
		Short: "Manage CLI plugins",
		RunE: func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("plugin command not yet implemented")
		},
	}
}
