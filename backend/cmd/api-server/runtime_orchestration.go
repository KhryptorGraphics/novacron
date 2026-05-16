//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	canonicalRuntimeOrchestrationEnv     = "CANONICAL_RUNTIME_ORCHESTRATION"
	canonicalRuntimeOrchestrationTimeout = 2 * time.Second
)

type runtimeOrchestrationClient struct {
	baseURL string
	client  *http.Client
}

func newRuntimeOrchestrationClientFromEnv() *runtimeOrchestrationClient {
	if !envBool(canonicalRuntimeOrchestrationEnv) {
		return nil
	}

	baseURL := strings.TrimSpace(os.Getenv(canonicalRuntimeBaseURLEnv))
	if baseURL == "" {
		return nil
	}

	return &runtimeOrchestrationClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: canonicalRuntimeOrchestrationTimeout,
		},
	}
}

func (c *runtimeOrchestrationClient) proxy(w http.ResponseWriter, source *http.Request, internalPath string) bool {
	if c == nil {
		return false
	}

	var body []byte
	if source.Body != nil && source.Body != http.NoBody {
		var err error
		body, err = io.ReadAll(source.Body)
		_ = source.Body.Close()
		if err != nil {
			source.Body = io.NopCloser(bytes.NewReader(nil))
			return false
		}
		source.Body = io.NopCloser(bytes.NewReader(body))
	}

	requestCtx, cancel := context.WithTimeout(source.Context(), canonicalRuntimeOrchestrationTimeout)
	defer cancel()

	targetURL := c.baseURL + internalPath
	if rawQuery := source.URL.RawQuery; rawQuery != "" {
		targetURL += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(requestCtx, source.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return false
	}

	req.Header.Set("Accept", "application/json")
	if source.Header.Get("Content-Type") != "" {
		req.Header.Set("Content-Type", source.Header.Get("Content-Type"))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	w.Header().Set(novaCronReadSourceHeader, novaCronReadSourceRuntime)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
	return true
}
