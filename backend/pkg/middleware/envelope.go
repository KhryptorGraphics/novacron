package middleware

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// ResponseEnvelopeMiddleware wraps JSON responses into a standard envelope
// { data, error, pagination? }
func ResponseEnvelopeMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap := &captureResponseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(cap, r)

		// Only wrap JSON responses
		ct := cap.Header().Get("Content-Type")
		if ct == "" {
			ct = w.Header().Get("Content-Type")
		}
		if ct == "" || (ct != "application/json" && ct != "application/json; charset=utf-8") {
			// passthrough
			w.WriteHeader(cap.status)
			w.Write(cap.buf.Bytes())
			return
		}

		// Preserve pagination header for clients while wrapping body
		if xp := cap.Header().Get("X-Pagination"); xp != "" {
			w.Header().Set("X-Pagination", xp)
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(cap.status)

		var payload interface{}
		_ = json.Unmarshal(cap.buf.Bytes(), &payload)

		// Attempt to read pagination info from header set by handlers
		var pagination interface{}
		if pjson := cap.Header().Get("X-Pagination"); pjson != "" {
			_ = json.Unmarshal([]byte(pjson), &pagination)
		}

		env := map[string]interface{}{
			"data":  nil,
			"error": nil,
		}
		if cap.status >= 400 {
			// Error response
			msg := string(cap.buf.Bytes())
			_ = json.Unmarshal(cap.buf.Bytes(), &payload)
			code := "unknown"
			if cap.status == http.StatusBadRequest { code = "invalid_argument" }
			if cap.status == http.StatusUnauthorized { code = "unauthorized" }
			if cap.status == http.StatusForbidden { code = "forbidden" }
			if cap.status == http.StatusNotFound { code = "not_found" }
			if cap.status == http.StatusConflict { code = "conflict" }
			if cap.status == http.StatusUnprocessableEntity { code = "validation_failed" }
			if m, ok := payload.(map[string]interface{}); ok {
				if em, ok2 := m["error"].(map[string]interface{}); ok2 {
					if c, ok3 := em["code"].(string); ok3 { code = c }
					if m2, ok3 := em["message"].(string); ok3 { msg = m2 }
				}
			}
			env["error"] = map[string]interface{}{"code": code, "message": msg}
		} else {
			// Success
			env["data"] = payload
			if pagination != nil {
				env["pagination"] = pagination
			}
		}
		_ = json.NewEncoder(w).Encode(env)
	})
}

type captureResponseWriter struct{
	http.ResponseWriter
	status int
	buf    bytes.Buffer
}

func (c *captureResponseWriter) WriteHeader(statusCode int) {
	c.status = statusCode
	// Do not write yet; defer until after handler returns
}

func (c *captureResponseWriter) Write(b []byte) (int, error) {
	return c.buf.Write(b)
}

