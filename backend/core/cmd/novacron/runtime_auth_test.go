package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestRuntimeAuthRegisterLoginAndAdmissionFlow(t *testing.T) {
	t.Parallel()

	apiServer, manager := newRuntimeAuthTestServer(t, runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      "http://localhost:3000",
		DefaultTenantID:  "default",
		DefaultClusterID: "cluster-local",
	})
	defer manager.Stop()
	defer apiServer.Shutdown(context.Background())

	registeredUser := postJSONResponse[runtimeAuthUserResponse](t, apiServer, "/api/auth/register", runtimeAuthRegisterRequest{
		Email:     "user@example.com",
		Password:  "ValidPassword123!",
		FirstName: "Edge",
		LastName:  "User",
	}, http.StatusCreated)

	if got, want := registeredUser.Email, "user@example.com"; got != want {
		t.Fatalf("registered user email = %q, want %q", got, want)
	}
	if got, want := registeredUser.TenantID, "default"; got != want {
		t.Fatalf("registered tenant id = %q, want %q", got, want)
	}

	authResponse := postJSONResponse[runtimeAuthResponse](t, apiServer, "/api/auth/login", runtimeAuthLoginRequest{
		Email:    "user@example.com",
		Password: "ValidPassword123!",
	}, http.StatusOK)

	if authResponse.Token == "" {
		t.Fatal("expected login token to be issued")
	}
	if got, want := authResponse.User.Email, "user@example.com"; got != want {
		t.Fatalf("login user email = %q, want %q", got, want)
	}
	if !authResponse.Admission.Admitted {
		t.Fatal("expected authenticated user to be admitted into the cluster")
	}
	if got, want := authResponse.Admission.ClusterID, "cluster-local"; got != want {
		t.Fatalf("admission cluster id = %q, want %q", got, want)
	}

	me := getAuthorizedJSONResponse[runtimeAuthCurrentUserResponse](t, apiServer, "/api/auth/me", authResponse.Token)
	if got, want := me.User.Email, "user@example.com"; got != want {
		t.Fatalf("current user email = %q, want %q", got, want)
	}
	if !me.Admission.Admitted {
		t.Fatal("expected /api/auth/me to include cluster admission")
	}

	admission := getAuthorizedJSONResponse[runtimeAdmissionResponse](t, apiServer, "/api/cluster/admission", authResponse.Token)
	if !admission.Admitted {
		t.Fatal("expected /api/cluster/admission to report admitted user")
	}
	if got, want := admission.ClusterID, "cluster-local"; got != want {
		t.Fatalf("cluster admission id = %q, want %q", got, want)
	}

	health := getAuthorizedJSONResponse[runtimeClusterHealth](t, apiServer, "/api/cluster/health", authResponse.Token)
	if health.Auth == nil || !health.Auth.Enabled {
		t.Fatal("expected protected cluster health to include enabled auth metadata")
	}
	if health.Federation == nil || health.Federation.TotalClusters < 1 {
		t.Fatal("expected protected cluster health to include federation cluster summaries")
	}
}

func TestRuntimeAuthRefreshLogoutAndClusterSelectionFlow(t *testing.T) {
	t.Parallel()

	apiServer, manager := newRuntimeAuthTestServer(t, runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      "http://localhost:3000",
		DefaultTenantID:  "default",
		DefaultClusterID: "cluster-local",
	})
	defer manager.Stop()
	defer apiServer.Shutdown(context.Background())

	if err := apiServer.runtimeAuth.persistence.clusters.Upsert(&runtimeClusterRecord{
		ID:                         "cluster-premium",
		Name:                       "Premium Cluster",
		InterconnectLatencyMS:      0.8,
		InterconnectBandwidthMBPS:  25000,
		GrowthLatencyPenaltyMS:     apiServer.runtimeAuth.config.Clustering.GrowthLatencyPenaltyMS,
		GrowthBandwidthPenaltyMBPS: apiServer.runtimeAuth.config.Clustering.GrowthBandwidthPenaltyMBPS,
		CurrentNodeCount:           2,
		FederationState:            "tiered-federation",
	}); err != nil {
		t.Fatalf("upsert premium cluster: %v", err)
	}

	postJSONResponse[runtimeAuthUserResponse](t, apiServer, "/api/auth/register", runtimeAuthRegisterRequest{
		Email:     "operator@example.com",
		Password:  "ValidPassword123!",
		FirstName: "Op",
		LastName:  "User",
	}, http.StatusCreated)

	loginResponse := postJSONResponse[runtimeAuthResponse](t, apiServer, "/api/auth/login", runtimeAuthLoginRequest{
		Email:    "operator@example.com",
		Password: "ValidPassword123!",
	}, http.StatusOK)

	if len(loginResponse.Memberships) != 2 {
		t.Fatalf("login memberships = %d, want 2", len(loginResponse.Memberships))
	}

	unauthorizedHealth := getRawResponse(t, apiServer, "/api/cluster/health", "", http.StatusUnauthorized)
	if !strings.Contains(unauthorizedHealth, "authorization required") {
		t.Fatalf("unauthorized cluster health response = %q, want auth failure", unauthorizedHealth)
	}

	selected := postAuthorizedJSONResponse[runtimeAuthCurrentUserResponse](t, apiServer, "/api/cluster/admissions/select", loginResponse.Token, runtimeSelectClusterRequest{
		ClusterID: "cluster-premium",
	}, http.StatusOK)
	if selected.SelectedCluster == nil || selected.SelectedCluster.ID != "cluster-premium" {
		t.Fatalf("selected cluster = %+v, want cluster-premium", selected.SelectedCluster)
	}

	refreshed := postJSONResponse[runtimeAuthResponse](t, apiServer, "/api/auth/refresh", runtimeRefreshRequest{
		RefreshToken: loginResponse.RefreshToken,
	}, http.StatusOK)
	if refreshed.SelectedCluster == nil || refreshed.SelectedCluster.ID != "cluster-premium" {
		t.Fatalf("refreshed selected cluster = %+v, want cluster-premium", refreshed.SelectedCluster)
	}

	federation := getAuthorizedJSONResponse[runtimeFederationResponse](t, apiServer, "/api/cluster/federation", refreshed.Token)
	if federation.Federation.TotalClusters != 2 {
		t.Fatalf("federation total clusters = %d, want 2", federation.Federation.TotalClusters)
	}
	if federation.Federation.SelectedClusterID != "cluster-premium" {
		t.Fatalf("federation selected cluster id = %q, want cluster-premium", federation.Federation.SelectedClusterID)
	}

	postAuthorizedJSONResponse[map[string]bool](t, apiServer, "/api/auth/logout", refreshed.Token, map[string]string{}, http.StatusOK)

	loggedOut := getRawResponse(t, apiServer, "/api/auth/me", refreshed.Token, http.StatusUnauthorized)
	if !strings.Contains(loggedOut, "revoked") {
		t.Fatalf("post-logout me response = %q, want token revocation", loggedOut)
	}
}

func TestRuntimeAuthGitHubAuthorizationURLRequiresConfiguration(t *testing.T) {
	t.Parallel()

	apiServer, manager := newRuntimeAuthTestServer(t, runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      "http://localhost:3000",
		DefaultTenantID:  "default",
		DefaultClusterID: "cluster-local",
	})
	defer manager.Stop()
	defer apiServer.Shutdown(context.Background())

	response := getRawResponse(t, apiServer, "/api/auth/oauth/github/url?redirect_to=/dashboard", "", http.StatusServiceUnavailable)
	if !strings.Contains(response, "github oauth is not configured") {
		t.Fatalf("unconfigured github oauth response = %q, want configuration error", response)
	}
}

func TestRuntimeAuthGitHubAuthorizationURLConfigured(t *testing.T) {
	t.Parallel()

	apiServer, manager := newRuntimeAuthTestServer(t, runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      "http://localhost:3000",
		DefaultTenantID:  "default",
		DefaultClusterID: "cluster-local",
		OAuth: runtimeOAuthConfig{
			GitHub: runtimeGitHubOAuthConfig{
				ClientID:     "github-client",
				ClientSecret: "github-secret",
				RedirectURL:  "http://localhost:8090/api/auth/oauth/github/callback",
			},
		},
	})
	defer manager.Stop()
	defer apiServer.Shutdown(context.Background())

	response := getJSONResponse[runtimeAuthProviderURLResponse](t, apiServer, "/api/auth/oauth/github/url?redirect_to=/dashboard")
	if got, want := response.Provider, "github"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if !strings.Contains(response.AuthorizationURL, "https://github.com/login/oauth/authorize") {
		t.Fatalf("authorization url = %q, want github authorize endpoint", response.AuthorizationURL)
	}
	if !strings.Contains(response.AuthorizationURL, "client_id=github-client") {
		t.Fatalf("authorization url = %q, want github client id", response.AuthorizationURL)
	}
}

func newRuntimeAuthTestServer(t *testing.T, authConfig runtimeAuthConfig) (*APIServer, *vm.VMManager) {
	t.Helper()

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "qemu-system-x86_64"))
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}

	if err := registerLocalSchedulerNode(manager, "test-node", t.TempDir()); err != nil {
		manager.Stop()
		t.Fatalf("registerLocalSchedulerNode returned error: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	apiServer, err := initializeAPI(ctx, runtimeConfig{Auth: authConfig}, "127.0.0.1:0", manager, nil, nil, nil, nil)
	if err != nil {
		manager.Stop()
		t.Fatalf("initializeAPI returned error: %v", err)
	}

	return apiServer, manager
}

func postJSONResponse[T any](t *testing.T, apiServer *APIServer, path string, body interface{}, expectedStatus int) T {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}

	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://%s%s", apiServer.address, path), bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("create POST request for %s: %v", path, err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("POST %s returned error: %v", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode != expectedStatus {
		bodyBytes, _ := io.ReadAll(response.Body)
		t.Fatalf("POST %s status = %d, want %d: %s", path, response.StatusCode, expectedStatus, strings.TrimSpace(string(bodyBytes)))
	}

	var decoded T
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode POST %s response: %v", path, err)
	}

	return decoded
}

func postAuthorizedJSONResponse[T any](t *testing.T, apiServer *APIServer, path string, token string, body interface{}, expectedStatus int) T {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}

	request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://%s%s", apiServer.address, path), bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("create POST request for %s: %v", path, err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+token)

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("POST %s returned error: %v", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode != expectedStatus {
		bodyBytes, _ := io.ReadAll(response.Body)
		t.Fatalf("POST %s status = %d, want %d: %s", path, response.StatusCode, expectedStatus, strings.TrimSpace(string(bodyBytes)))
	}

	var decoded T
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode POST %s response: %v", path, err)
	}

	return decoded
}

func getAuthorizedJSONResponse[T any](t *testing.T, apiServer *APIServer, path string, token string) T {
	t.Helper()

	request, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://%s%s", apiServer.address, path), nil)
	if err != nil {
		t.Fatalf("create GET request for %s: %v", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+token)

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("GET %s returned error: %v", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(response.Body)
		t.Fatalf("GET %s status = %d, want %d: %s", path, response.StatusCode, http.StatusOK, strings.TrimSpace(string(bodyBytes)))
	}

	var decoded T
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode GET %s response: %v", path, err)
	}

	return decoded
}

func getRawResponse(t *testing.T, apiServer *APIServer, path string, token string, expectedStatus int) string {
	t.Helper()

	request, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://%s%s", apiServer.address, path), nil)
	if err != nil {
		t.Fatalf("create GET request for %s: %v", path, err)
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("GET %s returned error: %v", path, err)
	}
	defer response.Body.Close()

	bodyBytes, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("read GET %s response body: %v", path, err)
	}
	if response.StatusCode != expectedStatus {
		t.Fatalf("GET %s status = %d, want %d: %s", path, response.StatusCode, expectedStatus, strings.TrimSpace(string(bodyBytes)))
	}

	return strings.TrimSpace(string(bodyBytes))
}
