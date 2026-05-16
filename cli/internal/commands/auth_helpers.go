package commands

import (
	"fmt"
	"strings"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
)

func newClusterAPIClient(cluster *config.Cluster) (*api.Client, error) {
	if cluster == nil {
		return nil, fmt.Errorf("cluster configuration is required")
	}

	options := []api.Option{
		api.WithInsecure(cluster.Insecure),
	}

	if strings.EqualFold(cluster.AuthType, "token") {
		store, err := auth.NewTokenStore()
		if err != nil {
			return nil, fmt.Errorf("initialize token store: %w", err)
		}

		tokenAuth, err := store.Load(cluster.Name)
		if err != nil {
			return nil, fmt.Errorf("load auth token for cluster %s: %w", cluster.Name, err)
		}
		if strings.TrimSpace(tokenAuth.RefreshURL) == "" {
			tokenAuth.RefreshURL = authEndpoint(cluster.Server, "/api/auth/refresh")
		}

		options = append(options, api.WithAuth(tokenAuth))
	}

	return api.NewClient(cluster.Server, options...)
}

func authEndpoint(server string, path string) string {
	return strings.TrimRight(server, "/") + path
}
