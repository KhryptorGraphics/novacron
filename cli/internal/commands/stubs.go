package commands

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
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

func NewAuthCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "auth",
		Short: "Inspect CLI authentication state",
	}

	cmd.AddCommand(newAuthInfoCommand())
	return cmd
}

func newAuthInfoCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "info",
		Short: "Show token status for the current cluster",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			cluster, err := manager.GetCurrentCluster()
			if err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintf(writer, "CLUSTER\t%s\n", cluster.Name)
			fmt.Fprintf(writer, "SERVER\t%s\n", cluster.Server)
			fmt.Fprintf(writer, "AUTH TYPE\t%s\n", cluster.AuthType)

			if !strings.EqualFold(cluster.AuthType, "token") {
				fmt.Fprintln(writer, "TOKEN STATUS\tnot configured")
				return writer.Flush()
			}

			store, err := auth.NewTokenStore()
			if err != nil {
				return err
			}
			tokenAuth, err := store.Load(cluster.Name)
			if err != nil {
				return fmt.Errorf("load auth token for cluster %s: %w", cluster.Name, err)
			}

			now := time.Now()
			status := "valid"
			switch {
			case tokenAuth.Token == "":
				status = "missing"
			case !tokenAuth.ExpiresAt.IsZero() && now.After(tokenAuth.ExpiresAt):
				status = "expired"
			case tokenAuth.ExpiresAt.IsZero():
				status = "unknown"
			}

			fmt.Fprintf(writer, "TOKEN STATUS\t%s\n", status)
			if tokenAuth.ExpiresAt.IsZero() {
				fmt.Fprintln(writer, "EXPIRES AT\tunknown")
			} else {
				fmt.Fprintf(writer, "EXPIRES AT\t%s\n", tokenAuth.ExpiresAt.UTC().Format(time.RFC3339))
			}
			if tokenAuth.RefreshToken == "" {
				fmt.Fprintln(writer, "REFRESH TOKEN\tmissing")
			} else {
				fmt.Fprintln(writer, "REFRESH TOKEN\tavailable")
			}
			if tokenAuth.RefreshURL != "" {
				fmt.Fprintf(writer, "REFRESH URL\t%s\n", tokenAuth.RefreshURL)
			}
			return writer.Flush()
		},
	}
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
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage CLI configuration",
	}

	cmd.AddCommand(
		newConfigSetClusterCommand(),
		newConfigUseContextCommand(),
		newConfigViewCommand(),
		newConfigGetClustersCommand(),
		newConfigGetClusterCommand(),
	)

	return cmd
}

func newConfigSetClusterCommand() *cobra.Command {
	var (
		server    string
		namespace string
		authType  string
		insecure  bool
	)

	cmd := &cobra.Command{
		Use:   "set-cluster <name>",
		Short: "Create or update a cluster connection",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := strings.TrimSpace(args[0])
			if name == "" {
				return fmt.Errorf("cluster name is required")
			}

			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			existing := manager.Get().Clusters[name]
			if strings.TrimSpace(server) == "" {
				server = existing.Server
			}
			server = strings.TrimRight(strings.TrimSpace(server), "/")
			if server == "" {
				return fmt.Errorf("server is required")
			}

			if namespace == "" {
				namespace = existing.Namespace
			}
			if namespace == "" {
				namespace = "default"
			}
			if authType == "" {
				authType = existing.AuthType
			}
			if authType == "" {
				authType = "token"
			}
			insecure = insecure || existing.Insecure

			if err := manager.AddCluster(config.Cluster{
				Name:      name,
				Server:    server,
				Insecure:  insecure,
				Namespace: namespace,
				AuthType:  authType,
				AuthData:  existing.AuthData,
			}); err != nil {
				return err
			}

			fmt.Fprintf(cmd.OutOrStdout(), "Cluster %s saved\n", name)
			return nil
		},
	}

	cmd.Flags().StringVar(&server, "server", "", "NovaCron API server URL")
	cmd.Flags().StringVar(&namespace, "namespace", "", "default namespace")
	cmd.Flags().StringVar(&authType, "auth-type", "", "authentication type")
	cmd.Flags().BoolVar(&insecure, "insecure", false, "skip TLS certificate verification")

	return cmd
}

func newConfigUseContextCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "use-context <name>",
		Short: "Set the current cluster",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}
			if err := manager.SetCurrentCluster(args[0]); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "Current cluster set to %s\n", args[0])
			return nil
		},
	}
}

func newConfigViewCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "view",
		Short: "Print CLI configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}
			data, err := yaml.Marshal(manager.Get())
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}
}

func newConfigGetClustersCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "get-clusters",
		Short: "List configured clusters",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			cfg := manager.Get()
			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "CURRENT\tNAME\tSERVER\tNAMESPACE\tAUTH")
			for name, cluster := range cfg.Clusters {
				current := ""
				if name == cfg.CurrentCluster {
					current = "*"
				}
				fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\n", current, cluster.Name, cluster.Server, cluster.Namespace, cluster.AuthType)
			}
			return writer.Flush()
		},
	}
}

func newConfigGetClusterCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "get-cluster [name]",
		Short: "Print a cluster configuration",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			var cluster config.Cluster
			if len(args) == 1 {
				found, ok := manager.Get().Clusters[args[0]]
				if !ok {
					return fmt.Errorf("cluster %s not found", args[0])
				}
				cluster = found
			} else {
				current, err := manager.GetCurrentCluster()
				if err != nil {
					return err
				}
				cluster = *current
			}

			data, err := yaml.Marshal(cluster)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}
}

func NewNodeCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "node",
		Short: "Manage cluster nodes",
	}

	cmd.AddCommand(
		newNodeListCommand(),
		newNodeGetCommand(),
	)

	return cmd
}

type clusterNode struct {
	ID                 string            `json:"id" yaml:"id"`
	Address            string            `json:"address,omitempty" yaml:"address,omitempty"`
	Status             string            `json:"status" yaml:"status"`
	CPU                int               `json:"cpu,omitempty" yaml:"cpu,omitempty"`
	Memory             int64             `json:"memory,omitempty" yaml:"memory,omitempty"`
	Disk               int64             `json:"disk,omitempty" yaml:"disk,omitempty"`
	UsedCPU            int               `json:"used_cpu,omitempty" yaml:"used_cpu,omitempty"`
	RemainingCPU       int               `json:"remaining_cpu,omitempty" yaml:"remaining_cpu,omitempty"`
	UsedMemoryMB       int64             `json:"used_memory_mb,omitempty" yaml:"used_memory_mb,omitempty"`
	RemainingMemoryMB  int64             `json:"remaining_memory_mb,omitempty" yaml:"remaining_memory_mb,omitempty"`
	UsedDiskGB         int64             `json:"used_disk_gb,omitempty" yaml:"used_disk_gb,omitempty"`
	RemainingDiskGB    int64             `json:"remaining_disk_gb,omitempty" yaml:"remaining_disk_gb,omitempty"`
	CPUUsagePercent    float64           `json:"cpu_usage_percent,omitempty" yaml:"cpu_usage_percent,omitempty"`
	MemoryUsagePercent float64           `json:"memory_usage_percent,omitempty" yaml:"memory_usage_percent,omitempty"`
	DiskUsagePercent   float64           `json:"disk_usage_percent,omitempty" yaml:"disk_usage_percent,omitempty"`
	VMCount            int               `json:"vm_count,omitempty" yaml:"vm_count,omitempty"`
	Schedulable        bool              `json:"schedulable" yaml:"schedulable"`
	Labels             map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

func newNodeListCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List cluster nodes",
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var nodes []clusterNode
			if err := client.Get(cmd.Context(), "/api/cluster/nodes", &nodes); err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "NAME\tSTATUS\tCPU\tMEMORY(MB)\tDISK(GB)\tVMS\tSCHEDULABLE")
			for _, node := range nodes {
				fmt.Fprintf(writer, "%s\t%s\t%d\t%d\t%d\t%d\t%t\n", node.ID, node.Status, node.CPU, node.Memory, node.Disk, node.VMCount, node.Schedulable)
			}
			return writer.Flush()
		},
	}
}

func newNodeGetCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "get <name>",
		Short: "Get cluster node details",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var node clusterNode
			if err := client.Get(cmd.Context(), "/api/cluster/nodes/"+args[0], &node); err != nil {
				return err
			}

			data, err := yaml.Marshal(node)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}
}

func currentClusterAPIClient() (*api.Client, error) {
	manager, err := config.NewManager(cfgFile)
	if err != nil {
		return nil, err
	}
	cluster, err := manager.GetCurrentCluster()
	if err != nil {
		return nil, err
	}
	return newClusterAPIClient(cluster)
}

func NewClusterCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Manage NovaCron clusters",
	}

	cmd.AddCommand(
		newClusterListCommand(),
		newClusterInfoCommand(),
	)

	return cmd
}

func newClusterListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List configured clusters",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}

			cfg := manager.Get()
			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "CURRENT\tNAME\tSERVER\tNAMESPACE\tAUTH")
			for name, cluster := range cfg.Clusters {
				current := ""
				if name == cfg.CurrentCluster {
					current = "*"
				}
				fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\n", current, cluster.Name, cluster.Server, cluster.Namespace, cluster.AuthType)
			}
			return writer.Flush()
		},
	}
}

type clusterHealthResponse struct {
	Status       string    `json:"status"`
	TotalNodes   int       `json:"total_nodes"`
	HealthyNodes int       `json:"healthy_nodes"`
	HasQuorum    bool      `json:"has_quorum"`
	Leader       string    `json:"leader"`
	LastUpdated  time.Time `json:"last_updated"`
}

func newClusterInfoCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "info",
		Short: "Show current cluster health",
		RunE: func(cmd *cobra.Command, args []string) error {
			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}
			cluster, err := manager.GetCurrentCluster()
			if err != nil {
				return err
			}
			client, err := newClusterAPIClient(cluster)
			if err != nil {
				return err
			}

			var health clusterHealthResponse
			if err := client.Get(cmd.Context(), "/api/cluster/health", &health); err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintf(writer, "NAME\t%s\n", cluster.Name)
			fmt.Fprintf(writer, "SERVER\t%s\n", cluster.Server)
			fmt.Fprintf(writer, "STATUS\t%s\n", health.Status)
			fmt.Fprintf(writer, "TOTAL NODES\t%d\n", health.TotalNodes)
			fmt.Fprintf(writer, "HEALTHY NODES\t%d\n", health.HealthyNodes)
			fmt.Fprintf(writer, "QUORUM\t%t\n", health.HasQuorum)
			fmt.Fprintf(writer, "LEADER\t%s\n", health.Leader)
			if !health.LastUpdated.IsZero() {
				fmt.Fprintf(writer, "LAST UPDATED\t%s\n", health.LastUpdated.Format(time.RFC3339))
			}
			return writer.Flush()
		},
	}
}

func NewMigrateCommand() *cobra.Command {
	var (
		targetNode  string
		maxDowntime int
		bandwidth   int
		priority    string
	)

	cmd := &cobra.Command{
		Use:   "migrate <vm-id>",
		Short: "Migrate VMs between nodes",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(targetNode) == "" {
				return fmt.Errorf("target-node is required")
			}
			if maxDowntime < 0 {
				return fmt.Errorf("max-downtime must be non-negative")
			}
			if bandwidth < 0 {
				return fmt.Errorf("bandwidth must be non-negative")
			}

			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			req := liveMigrationRequest{
				TargetNode:  strings.TrimSpace(targetNode),
				MaxDowntime: maxDowntime,
				Bandwidth:   bandwidth,
				Priority:    strings.TrimSpace(priority),
			}

			var migration migrationResponse
			path := "/migration/live/" + url.PathEscape(strings.TrimSpace(args[0]))
			if err := client.Post(cmd.Context(), path, req, &migration); err != nil {
				return err
			}

			data, err := yaml.Marshal(migration)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVar(&targetNode, "target-node", "", "target node for migration")
	cmd.Flags().IntVar(&maxDowntime, "max-downtime", 0, "maximum allowed downtime in milliseconds")
	cmd.Flags().IntVar(&bandwidth, "bandwidth", 0, "migration bandwidth limit")
	cmd.Flags().StringVar(&priority, "priority", "", "migration priority")

	return cmd
}

type liveMigrationRequest struct {
	TargetNode  string `json:"targetNode"`
	MaxDowntime int    `json:"maxDowntime,omitempty"`
	Bandwidth   int    `json:"bandwidth,omitempty"`
	Priority    string `json:"priority,omitempty"`
}

type migrationResponse struct {
	MigrationID string `json:"migrationId,omitempty" yaml:"migration_id,omitempty"`
	Status      string `json:"status,omitempty" yaml:"status,omitempty"`
	VMID        string `json:"vmId,omitempty" yaml:"vm_id,omitempty"`
	TargetNode  string `json:"targetNode,omitempty" yaml:"target_node,omitempty"`
	Priority    string `json:"priority,omitempty" yaml:"priority,omitempty"`
	CreatedAt   string `json:"createdAt,omitempty" yaml:"created_at,omitempty"`
}

func NewSnapshotCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "snapshot",
		Short: "Manage VM snapshots",
	}

	cmd.AddCommand(newSnapshotCreateCommand())
	return cmd
}

func newSnapshotCreateCommand() *cobra.Command {
	var (
		description string
		memory      bool
		quiesce     bool
	)

	cmd := &cobra.Command{
		Use:   "create <vm-id> <snapshot-name>",
		Short: "Create a VM snapshot",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			req := createSnapshotRequest{
				Name:        strings.TrimSpace(args[1]),
				Description: strings.TrimSpace(description),
				Memory:      memory,
				Quiesce:     quiesce,
			}

			var snapshot snapshotResponse
			path := "/api/v1/vms/" + url.PathEscape(strings.TrimSpace(args[0])) + "/snapshot"
			if err := client.Post(cmd.Context(), path, req, &snapshot); err != nil {
				return err
			}

			data, err := yaml.Marshal(snapshot)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVar(&description, "description", "", "Snapshot description")
	cmd.Flags().BoolVar(&memory, "memory", false, "Include VM memory in the snapshot")
	cmd.Flags().BoolVar(&quiesce, "quiesce", false, "Quiesce guest filesystems before snapshot")

	return cmd
}

type createSnapshotRequest struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Memory      bool   `json:"memory,omitempty"`
	Quiesce     bool   `json:"quiesce,omitempty"`
}

type snapshotResponse struct {
	SnapshotID string `json:"snapshot_id,omitempty" yaml:"snapshot_id,omitempty"`
	Status     string `json:"status,omitempty" yaml:"status,omitempty"`
	Message    string `json:"message,omitempty" yaml:"message,omitempty"`
	VMID       string `json:"vm_id,omitempty" yaml:"vm_id,omitempty"`
	Name       string `json:"name,omitempty" yaml:"name,omitempty"`
	CreatedAt  string `json:"created_at,omitempty" yaml:"created_at,omitempty"`
}

func NewMonitorCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "monitor",
		Short: "Monitor cluster resources",
	}

	cmd.AddCommand(
		newMonitorMetricsCommand(),
		newMonitorVMsCommand(),
	)

	return cmd
}

type vmMetricsResponse struct {
	ID          string  `json:"id" yaml:"id"`
	CPUUsage    float64 `json:"cpu_usage" yaml:"cpu_usage"`
	MemoryUsage float64 `json:"memory_usage" yaml:"memory_usage"`
}

type monitoringVM struct {
	VMID        string  `json:"vmId" yaml:"vm_id"`
	Name        string  `json:"name" yaml:"name"`
	Status      string  `json:"status" yaml:"status"`
	CPUUsage    float64 `json:"cpuUsage" yaml:"cpu_usage"`
	MemoryUsage float64 `json:"memoryUsage" yaml:"memory_usage"`
	DiskUsage   float64 `json:"diskUsage" yaml:"disk_usage"`
	NetworkRx   float64 `json:"networkRx" yaml:"network_rx"`
	NetworkTx   float64 `json:"networkTx" yaml:"network_tx"`
	IOPS        float64 `json:"iops" yaml:"iops"`
}

func newMonitorMetricsCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "metrics <vm-id>",
		Short: "Show current VM metrics",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var metrics vmMetricsResponse
			path := "/api/v1/vms/" + url.PathEscape(args[0]) + "/metrics"
			if err := client.Get(cmd.Context(), path, &metrics); err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintf(writer, "ID\t%s\n", metrics.ID)
			fmt.Fprintf(writer, "CPU USAGE\t%.2f%%\n", metrics.CPUUsage)
			fmt.Fprintf(writer, "MEMORY USAGE\t%.2f%%\n", metrics.MemoryUsage)
			return writer.Flush()
		},
	}
}

func newMonitorVMsCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "vms",
		Aliases: []string{"vm"},
		Short:   "List monitored VMs",
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var vms []monitoringVM
			if err := client.Get(cmd.Context(), "/api/monitoring/vms", &vms); err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "VM ID\tNAME\tSTATUS\tCPU\tMEMORY\tDISK\tRX\tTX\tIOPS")
			for _, vm := range vms {
				fmt.Fprintf(writer, "%s\t%s\t%s\t%.2f%%\t%.2f%%\t%.2f%%\t%.0f\t%.0f\t%.0f\n",
					vm.VMID,
					vm.Name,
					vm.Status,
					vm.CPUUsage,
					vm.MemoryUsage,
					vm.DiskUsage,
					vm.NetworkRx,
					vm.NetworkTx,
					vm.IOPS,
				)
			}
			return writer.Flush()
		},
	}
}

func NewLogsCommand() *cobra.Command {
	var (
		count      int
		level      string
		components string
		vmID       string
		timeout    time.Duration
	)

	cmd := &cobra.Command{
		Use:   "logs [source]",
		Short: "Stream cluster logs",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if count <= 0 {
				return fmt.Errorf("count must be greater than zero")
			}

			manager, err := config.NewManager(cfgFile)
			if err != nil {
				return err
			}
			cluster, err := manager.GetCurrentCluster()
			if err != nil {
				return err
			}
			client, err := newClusterAPIClient(cluster)
			if err != nil {
				return err
			}
			if strings.EqualFold(cluster.AuthType, "token") {
				store, err := auth.NewTokenStore()
				if err != nil {
					return fmt.Errorf("initialize token store: %w", err)
				}
				tokenAuth, err := store.Load(cluster.Name)
				if err != nil {
					return fmt.Errorf("load auth token for cluster %s: %w", cluster.Name, err)
				}
				client.SetToken(tokenAuth.Token)
			}

			source := "all"
			if len(args) == 1 {
				source = strings.TrimSpace(args[0])
			}
			path := "/api/ws/logs"
			if source != "" && source != "all" {
				path += "/" + url.PathEscape(source)
			}

			query := url.Values{}
			if strings.TrimSpace(level) != "" {
				query.Set("level", strings.TrimSpace(level))
			}
			if strings.TrimSpace(components) != "" {
				query.Set("components", strings.TrimSpace(components))
			}
			if strings.TrimSpace(vmID) != "" {
				query.Set("vm_id", strings.TrimSpace(vmID))
			}
			if encoded := query.Encode(); encoded != "" {
				path += "?" + encoded
			}

			ctx, cancel := context.WithTimeout(cmd.Context(), timeout)
			defer cancel()

			conn, err := client.WebSocket(ctx, path)
			if err != nil {
				return err
			}
			defer conn.Close()

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "TIME\tSOURCE\tLEVEL\tCOMPONENT\tVM\tMESSAGE")
			for i := 0; i < count; i++ {
				var msg logStreamMessage
				if err := conn.Receive(&msg); err != nil {
					return fmt.Errorf("receive log entry: %w", err)
				}
				fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\t%s\n",
					formatLogTimestamp(msg.Timestamp),
					msg.Source,
					msg.Level,
					msg.Component,
					msg.VMID,
					msg.Message,
				)
			}
			return writer.Flush()
		},
	}

	cmd.Flags().IntVar(&count, "count", 10, "Number of log entries to read before exiting")
	cmd.Flags().StringVar(&level, "level", "", "Comma-separated log levels to include")
	cmd.Flags().StringVar(&components, "components", "", "Comma-separated components to include")
	cmd.Flags().StringVar(&vmID, "vm-id", "", "VM ID filter for VM logs")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Second, "Maximum time to wait for log entries")

	return cmd
}

type logStreamMessage struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Timestamp time.Time              `json:"timestamp"`
	Component string                 `json:"component,omitempty"`
	VMID      string                 `json:"vm_id,omitempty"`
	Labels    map[string]string      `json:"labels,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

func formatLogTimestamp(timestamp time.Time) string {
	if timestamp.IsZero() {
		return ""
	}
	return timestamp.Format(time.RFC3339)
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
		Use:   "top [system]",
		Short: "Display resource usage",
		Args: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 {
				return nil
			}
			if len(args) == 1 && args[0] == "system" {
				return nil
			}
			return fmt.Errorf("unsupported resource %q", strings.Join(args, " "))
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var summary monitoringSummary
			if err := client.Get(cmd.Context(), "/api/monitoring/metrics", &summary); err != nil {
				return err
			}

			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
			fmt.Fprintln(writer, "RESOURCE\tUSAGE\tCHANGE\tANALYSIS")
			fmt.Fprintf(writer, "CPU\t%.2f%%\t%+.2f%%\t%s\n", summary.CurrentCPUUsage, summary.CPUChangePercentage, summary.CPUAnalysis)
			fmt.Fprintf(writer, "Memory\t%.2f%%\t%+.2f%%\t%s\n", summary.CurrentMemoryUsage, summary.MemoryChangePercentage, summary.MemoryAnalysis)
			fmt.Fprintf(writer, "Disk\t%.2f%%\t%+.2f%%\t\n", summary.CurrentDiskUsage, summary.DiskChangePercentage)
			fmt.Fprintf(writer, "Network\t%.2f%%\t%+.2f%%\t\n", summary.CurrentNetworkUsage, summary.NetworkChangePercentage)
			return writer.Flush()
		},
	}
}

type monitoringSummary struct {
	CurrentCPUUsage         float64  `json:"currentCpuUsage"`
	CurrentMemoryUsage      float64  `json:"currentMemoryUsage"`
	CurrentDiskUsage        float64  `json:"currentDiskUsage"`
	CurrentNetworkUsage     float64  `json:"currentNetworkUsage"`
	CPUChangePercentage     float64  `json:"cpuChangePercentage"`
	MemoryChangePercentage  float64  `json:"memoryChangePercentage"`
	DiskChangePercentage    float64  `json:"diskChangePercentage"`
	NetworkChangePercentage float64  `json:"networkChangePercentage"`
	TimeLabels              []string `json:"timeLabels"`
	CPUAnalysis             string   `json:"cpuAnalysis"`
	MemoryAnalysis          string   `json:"memoryAnalysis"`
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
	var (
		force bool
		vmID  string
	)

	cmd := &cobra.Command{
		Use:   "delete <resource> <name>",
		Short: "Delete resources",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			if !force {
				return fmt.Errorf("delete %s %s requires --force", args[0], args[1])
			}

			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			resource := strings.ToLower(strings.TrimSpace(args[0]))
			name := strings.TrimSpace(args[1])

			path := ""
			switch resource {
			case "vm", "vms":
				path = "/api/v1/vms/" + url.PathEscape(name)
			case "network", "networks", "net":
				path = "/api/v1/networks/" + url.PathEscape(name)
			case "interface", "interfaces", "nic":
				if strings.TrimSpace(vmID) == "" {
					return fmt.Errorf("vm is required for interface deletion")
				}
				path = "/api/v1/vms/" + url.PathEscape(strings.TrimSpace(vmID)) + "/interfaces/" + url.PathEscape(name)
			default:
				return fmt.Errorf("unsupported resource %q", args[0])
			}

			var response deleteResponse
			if err := client.Delete(cmd.Context(), path); err != nil {
				return err
			}
			response.ID = name
			response.Status = "deleted"
			if resource == "interface" || resource == "interfaces" || resource == "nic" {
				response.VMID = strings.TrimSpace(vmID)
				response.Status = "detached"
			}

			data, err := yaml.Marshal(response)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().BoolVar(&force, "force", false, "Delete without interactive confirmation")
	cmd.Flags().StringVar(&vmID, "vm", "", "VM ID for interface deletion")

	return cmd
}

func NewDescribeCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "describe <resource> <name>",
		Short: "Describe resources",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			resource := strings.ToLower(strings.TrimSpace(args[0]))
			name := strings.TrimSpace(args[1])

			switch resource {
			case "vm", "vms":
				client, err := currentClusterAPIClient()
				if err != nil {
					return err
				}
				return describeVM(cmd, client, name)
			case "network", "networks", "net":
				client, err := currentClusterAPIClient()
				if err != nil {
					return err
				}
				return describeNetwork(cmd, client, name)
			default:
				return fmt.Errorf("unsupported resource %q", args[0])
			}
		},
	}
}

func describeVM(cmd *cobra.Command, client *api.Client, name string) error {
	var vm coreVM
	if err := client.Get(cmd.Context(), "/api/v1/vms/"+url.PathEscape(name), &vm); err != nil {
		return err
	}

	writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintf(writer, "Name:\t%s\n", vm.Name)
	fmt.Fprintf(writer, "ID:\t%s\n", vm.ID)
	fmt.Fprintf(writer, "Status:\t%s\n", firstNonEmpty(vm.Status, vm.State))
	fmt.Fprintf(writer, "Node:\t%s\n", vm.NodeID)
	fmt.Fprintf(writer, "Tenant:\t%s\n", vm.TenantID)
	if vm.CreatedAt != "" {
		fmt.Fprintf(writer, "Created:\t%s\n", vm.CreatedAt)
	}
	if vm.UpdatedAt != "" {
		fmt.Fprintf(writer, "Updated:\t%s\n", vm.UpdatedAt)
	}
	return writer.Flush()
}

func describeNetwork(cmd *cobra.Command, client *api.Client, name string) error {
	var network coreNetwork
	if err := client.Get(cmd.Context(), "/api/v1/networks/"+url.PathEscape(name), &network); err != nil {
		return err
	}

	writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintf(writer, "Name:\t%s\n", network.Name)
	fmt.Fprintf(writer, "ID:\t%s\n", network.ID)
	fmt.Fprintf(writer, "Type:\t%s\n", network.Type)
	fmt.Fprintf(writer, "Subnet:\t%s\n", network.Subnet)
	fmt.Fprintf(writer, "Gateway:\t%s\n", network.Gateway)
	fmt.Fprintf(writer, "Status:\t%s\n", network.Status)
	if network.CreatedAt != "" {
		fmt.Fprintf(writer, "Created:\t%s\n", network.CreatedAt)
	}
	if network.UpdatedAt != "" {
		fmt.Fprintf(writer, "Updated:\t%s\n", network.UpdatedAt)
	}
	return writer.Flush()
}

type deleteResponse struct {
	ID     string `json:"id" yaml:"id"`
	VMID   string `json:"vm_id,omitempty" yaml:"vm_id,omitempty"`
	Status string `json:"status" yaml:"status"`
}

func NewGetCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "get <resource> [name]",
		Short: "Get resources",
		Args:  cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			resource := strings.ToLower(strings.TrimSpace(args[0]))
			name := ""
			if len(args) == 2 {
				name = strings.TrimSpace(args[1])
			}

			switch resource {
			case "vm", "vms":
				client, err := currentClusterAPIClient()
				if err != nil {
					return err
				}
				if name == "" {
					return printVMList(cmd, client)
				}
				return printVMDetail(cmd, client, name)
			case "network", "networks", "net":
				client, err := currentClusterAPIClient()
				if err != nil {
					return err
				}
				if name == "" {
					return printNetworkList(cmd, client)
				}
				return printNetworkDetail(cmd, client, name)
			default:
				return fmt.Errorf("unsupported resource %q", args[0])
			}
		},
	}
}

type coreVM struct {
	ID        string `json:"id" yaml:"id"`
	Name      string `json:"name" yaml:"name"`
	State     string `json:"state,omitempty" yaml:"state,omitempty"`
	Status    string `json:"status,omitempty" yaml:"status,omitempty"`
	NodeID    string `json:"node_id,omitempty" yaml:"node_id,omitempty"`
	TenantID  string `json:"tenant_id,omitempty" yaml:"tenant_id,omitempty"`
	CreatedAt string `json:"created_at,omitempty" yaml:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty" yaml:"updated_at,omitempty"`
}

type coreNetwork struct {
	ID        string `json:"id" yaml:"id"`
	Name      string `json:"name" yaml:"name"`
	Type      string `json:"type" yaml:"type"`
	Subnet    string `json:"subnet" yaml:"subnet"`
	Gateway   string `json:"gateway,omitempty" yaml:"gateway,omitempty"`
	Status    string `json:"status" yaml:"status"`
	CreatedAt string `json:"created_at,omitempty" yaml:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty" yaml:"updated_at,omitempty"`
}

func printVMList(cmd *cobra.Command, client *api.Client) error {
	var vms []coreVM
	if err := client.Get(cmd.Context(), "/api/v1/vms", &vms); err != nil {
		return err
	}

	writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(writer, "ID\tNAME\tSTATUS\tNODE\tTENANT")
	for _, vm := range vms {
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\n", vm.ID, vm.Name, firstNonEmpty(vm.Status, vm.State), vm.NodeID, vm.TenantID)
	}
	return writer.Flush()
}

func printVMDetail(cmd *cobra.Command, client *api.Client, name string) error {
	var vm coreVM
	if err := client.Get(cmd.Context(), "/api/v1/vms/"+url.PathEscape(name), &vm); err != nil {
		return err
	}
	data, err := yaml.Marshal(vm)
	if err != nil {
		return err
	}
	_, err = cmd.OutOrStdout().Write(data)
	return err
}

func printNetworkList(cmd *cobra.Command, client *api.Client) error {
	var networks []coreNetwork
	if err := client.Get(cmd.Context(), "/api/v1/networks", &networks); err != nil {
		return err
	}

	writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(writer, "ID\tNAME\tTYPE\tSUBNET\tSTATUS")
	for _, network := range networks {
		fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\n", network.ID, network.Name, network.Type, network.Subnet, network.Status)
	}
	return writer.Flush()
}

func printNetworkDetail(cmd *cobra.Command, client *api.Client, name string) error {
	var network coreNetwork
	if err := client.Get(cmd.Context(), "/api/v1/networks/"+url.PathEscape(name), &network); err != nil {
		return err
	}
	data, err := yaml.Marshal(network)
	if err != nil {
		return err
	}
	_, err = cmd.OutOrStdout().Write(data)
	return err
}

func NewCreateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "create",
		Short: "Create resources",
	}

	cmd.AddCommand(
		newCreateVMCommand(),
		newCreateNetworkCommand(),
	)

	return cmd
}

type createVMRequest struct {
	Name      string                 `json:"name"`
	State     string                 `json:"state,omitempty"`
	NodeID    string                 `json:"node_id,omitempty"`
	Tags      map[string]interface{} `json:"tags,omitempty"`
	CPUShares int                    `json:"cpu_shares,omitempty"`
	MemoryMB  int                    `json:"memory_mb,omitempty"`
}

type createNetworkRequest struct {
	Name    string `json:"name"`
	Type    string `json:"type,omitempty"`
	Subnet  string `json:"subnet"`
	Gateway string `json:"gateway,omitempty"`
}

func newCreateVMCommand() *cobra.Command {
	var (
		nodeID    string
		cpuShares int
		memoryMB  int
	)

	cmd := &cobra.Command{
		Use:   "vm <name>",
		Short: "Create a VM",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var vm coreVM
			req := createVMRequest{
				Name:      args[0],
				NodeID:    strings.TrimSpace(nodeID),
				CPUShares: cpuShares,
				MemoryMB:  memoryMB,
			}
			if err := client.Post(cmd.Context(), "/api/v1/vms", req, &vm); err != nil {
				return err
			}

			data, err := yaml.Marshal(vm)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVar(&nodeID, "node", "", "Target node ID")
	cmd.Flags().IntVar(&cpuShares, "cpu", 0, "CPU shares")
	cmd.Flags().IntVar(&memoryMB, "memory", 0, "Memory in MB")

	return cmd
}

func newCreateNetworkCommand() *cobra.Command {
	var (
		networkType string
		subnet      string
		gateway     string
	)

	cmd := &cobra.Command{
		Use:     "network <name>",
		Aliases: []string{"net"},
		Short:   "Create a network",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(subnet) == "" {
				return fmt.Errorf("subnet is required")
			}

			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			var network coreNetwork
			req := createNetworkRequest{
				Name:    args[0],
				Type:    strings.TrimSpace(networkType),
				Subnet:  strings.TrimSpace(subnet),
				Gateway: strings.TrimSpace(gateway),
			}
			if err := client.Post(cmd.Context(), "/api/v1/networks", req, &network); err != nil {
				return err
			}

			data, err := yaml.Marshal(network)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVar(&networkType, "type", "bridged", "Network type")
	cmd.Flags().StringVar(&subnet, "subnet", "", "Network subnet in CIDR notation")
	cmd.Flags().StringVar(&gateway, "gateway", "", "Network gateway")

	return cmd
}

func NewUpdateCommand() *cobra.Command {
	var (
		vmID      string
		networkID string
		name      string
		ipAddress string
		status    string
	)

	cmd := &cobra.Command{
		Use:   "update <resource> <name>",
		Short: "Update resources",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			resource := strings.ToLower(strings.TrimSpace(args[0]))
			if resource != "interface" && resource != "interfaces" && resource != "nic" {
				return fmt.Errorf("unsupported resource %q", args[0])
			}
			if strings.TrimSpace(vmID) == "" {
				return fmt.Errorf("vm is required for interface update")
			}
			if strings.TrimSpace(networkID) == "" && strings.TrimSpace(name) == "" && strings.TrimSpace(ipAddress) == "" && strings.TrimSpace(status) == "" {
				return fmt.Errorf("at least one update field is required")
			}

			client, err := currentClusterAPIClient()
			if err != nil {
				return err
			}

			req := updateVMInterfaceRequest{
				NetworkID: strings.TrimSpace(networkID),
				Name:      strings.TrimSpace(name),
				IPAddress: strings.TrimSpace(ipAddress),
				Status:    strings.TrimSpace(status),
			}

			var updated coreVMInterface
			path := "/api/v1/vms/" + url.PathEscape(strings.TrimSpace(vmID)) + "/interfaces/" + url.PathEscape(strings.TrimSpace(args[1]))
			if err := client.Put(cmd.Context(), path, req, &updated); err != nil {
				return err
			}

			data, err := yaml.Marshal(updated)
			if err != nil {
				return err
			}
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVar(&vmID, "vm", "", "VM ID that owns the interface")
	cmd.Flags().StringVar(&networkID, "network", "", "Network ID to attach")
	cmd.Flags().StringVar(&name, "name", "", "Interface name")
	cmd.Flags().StringVar(&ipAddress, "ip", "", "Interface IP address")
	cmd.Flags().StringVar(&status, "status", "", "Interface status")

	return cmd
}

type updateVMInterfaceRequest struct {
	NetworkID string `json:"network_id,omitempty"`
	Name      string `json:"name,omitempty"`
	IPAddress string `json:"ip_address,omitempty"`
	Status    string `json:"status,omitempty"`
}

type coreVMInterface struct {
	ID         string `json:"id" yaml:"id"`
	VMID       string `json:"vm_id" yaml:"vm_id"`
	NetworkID  string `json:"network_id,omitempty" yaml:"network_id,omitempty"`
	Name       string `json:"name,omitempty" yaml:"name,omitempty"`
	MACAddress string `json:"mac_address,omitempty" yaml:"mac_address,omitempty"`
	IPAddress  string `json:"ip_address,omitempty" yaml:"ip_address,omitempty"`
	Status     string `json:"status,omitempty" yaml:"status,omitempty"`
	CreatedAt  string `json:"created_at,omitempty" yaml:"created_at,omitempty"`
	UpdatedAt  string `json:"updated_at,omitempty" yaml:"updated_at,omitempty"`
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
		Use:   "completion [bash|zsh|fish|powershell]",
		Short: "Generate shell completion scripts",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			root := cmd.Root()
			out := cmd.OutOrStdout()

			switch strings.ToLower(args[0]) {
			case "bash":
				return root.GenBashCompletion(out)
			case "zsh":
				return root.GenZshCompletion(out)
			case "fish":
				return root.GenFishCompletion(out, true)
			case "powershell":
				return root.GenPowerShellCompletion(out)
			default:
				return fmt.Errorf("unsupported shell %q", args[0])
			}
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
