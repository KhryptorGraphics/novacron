package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequestErrorBodies(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{
			name:   "json error body from the fabric handlers",
			status: http.StatusUnauthorized,
			body:   `{"error":"authorization header required"}`,
			want:   "authorization header required",
		},
		{
			name:   "coded error body from the DWCP handlers",
			status: http.StatusBadRequest,
			body:   `{"code":"bad_request","message":"command is required","details":"submit a command"}`,
			want:   "bad_request: command is required (submit a command)",
		},
		{
			name:   "non-JSON body is surfaced verbatim",
			status: http.StatusInternalServerError,
			body:   "internal error",
			want:   "request failed with status 500: internal error",
		},
		{
			name:   "empty body falls back to the status",
			status: http.StatusNotFound,
			body:   "",
			want:   "request failed with status 404 (Not Found)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client, err := NewClient(server.URL)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}

			_, err = client.Request(context.Background(), http.MethodGet, "/api/cluster/nodes", nil)
			if err == nil {
				t.Fatalf("expected status %d to fail the request", tt.status)
			}
			if err.Error() != tt.want {
				t.Errorf("error = %q, want %q", err, tt.want)
			}
		})
	}
}

func TestRequestSendsBearerToken(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"nodes":[]}`))
	}))
	defer server.Close()

	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	client.SetToken("jwt-token")

	var result map[string]interface{}
	if err := client.Get(context.Background(), "/api/cluster/nodes", &result); err != nil {
		t.Fatalf("Get: %v", err)
	}

	if authorization != "Bearer jwt-token" {
		t.Errorf("Authorization = %q, want %q", authorization, "Bearer jwt-token")
	}
}
