package commands

import (
	"bytes"
	"strings"
	"testing"

	"github.com/novacron/cli/pkg/config"
)

func TestConfigSetClusterPersistsConnection(t *testing.T) {
	withTempHome(t)

	var output bytes.Buffer
	cmd := NewConfigCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{
		"set-cluster", "prod",
		"--server", "https://api.novacron.example.com/",
		"--namespace", "production",
		"--insecure",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("config set-cluster failed: %v", err)
	}

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	cluster, err := manager.GetCurrentCluster()
	if err != nil {
		t.Fatalf("get current cluster: %v", err)
	}
	if cluster.Name != "prod" || cluster.Server != "https://api.novacron.example.com" {
		t.Fatalf("unexpected cluster identity: %#v", cluster)
	}
	if cluster.Namespace != "production" || !cluster.Insecure || cluster.AuthType != "token" {
		t.Fatalf("unexpected cluster settings: %#v", cluster)
	}
	if !strings.Contains(output.String(), "Cluster prod saved") {
		t.Fatalf("expected success output, got %q", output.String())
	}
}

func TestConfigUseContextSwitchesCurrentCluster(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com"}); err != nil {
		t.Fatalf("add prod: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "stage", Server: "https://stage.example.com"}); err != nil {
		t.Fatalf("add stage: %v", err)
	}

	cmd := NewConfigCommand()
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetArgs([]string{"use-context", "stage"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("config use-context failed: %v", err)
	}

	manager, err = config.NewManager("")
	if err != nil {
		t.Fatalf("reload config manager: %v", err)
	}
	if got := manager.Get().CurrentCluster; got != "stage" {
		t.Fatalf("expected current cluster stage, got %q", got)
	}
}

func TestConfigViewPrintsCurrentConfig(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com"}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}

	var output bytes.Buffer
	cmd := NewConfigCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"view"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("config view failed: %v", err)
	}

	for _, expected := range []string{"currentCluster: prod", "https://prod.example.com"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected config view to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestConfigGetClustersMarksCurrentCluster(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com"}); err != nil {
		t.Fatalf("add prod: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "stage", Server: "https://stage.example.com"}); err != nil {
		t.Fatalf("add stage: %v", err)
	}

	var output bytes.Buffer
	cmd := NewConfigCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"get-clusters"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("config get-clusters failed: %v", err)
	}

	for _, expected := range []string{"CURRENT", "NAME", "SERVER", "*", "prod", "stage"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected cluster list to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestConfigGetClusterPrintsSelectedCluster(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{
		Name:      "prod",
		Server:    "https://prod.example.com",
		Namespace: "production",
		AuthType:  "token",
	}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}

	var output bytes.Buffer
	cmd := NewConfigCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"get-cluster"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("config get-cluster failed: %v", err)
	}

	for _, expected := range []string{"name: prod", "server: https://prod.example.com", "namespace: production"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected cluster output to contain %q, got:\n%s", expected, output.String())
		}
	}
}
