//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	canonicalRuntimeInventoryReadsEnv = "CANONICAL_RUNTIME_INVENTORY_READS"
	canonicalRuntimeInventoryTimeout  = 2 * time.Second
)

type runtimeInventoryReadClient struct {
	baseURL string
	client  *http.Client
}

func newRuntimeInventoryReadClientFromEnv() *runtimeInventoryReadClient {
	if !envBool(canonicalRuntimeInventoryReadsEnv) {
		return nil
	}

	baseURL := strings.TrimSpace(os.Getenv(canonicalRuntimeBaseURLEnv))
	if baseURL == "" {
		return nil
	}

	return &runtimeInventoryReadClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: canonicalRuntimeInventoryTimeout,
		},
	}
}

func (c *runtimeInventoryReadClient) proxy(w http.ResponseWriter, source *http.Request, internalPath string) bool {
	if c == nil {
		return false
	}

	requestCtx, cancel := context.WithTimeout(source.Context(), canonicalRuntimeInventoryTimeout)
	defer cancel()

	targetURL := c.baseURL + internalPath
	if rawQuery := source.URL.RawQuery; rawQuery != "" {
		targetURL += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, targetURL, nil)
	if err != nil {
		return false
	}

	req.Header.Set("Accept", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false
	}

	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	w.Header().Set(novaCronReadSourceHeader, novaCronReadSourceRuntime)
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, resp.Body)
	return true
}

func setRuntimeSQLFallbackHeader(w http.ResponseWriter) {
	w.Header().Set(novaCronReadSourceHeader, novaCronReadSourceSQLFallback)
}
