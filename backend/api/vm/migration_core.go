//go:build !experimental

package vm

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
)

// Minimal core-mode migration endpoints: allow listing empty and reject unsupported

// RegisterCoreMigrationRoutes registers core-mode migration routes that compile
func RegisterCoreMigrationRoutes(router *mux.Router) {
	router.HandleFunc("/migrations", func(w http.ResponseWriter, r *http.Request){
		w.Header().Set("Content-Type","application/json")
		w.Write([]byte(`[]`))
	}).Methods("GET")

	router.HandleFunc("/migrations", func(w http.ResponseWriter, r *http.Request){
		w.Header().Set("Content-Type","application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{"code":"unsupported_feature","message":"migrations not enabled in core mode"},
		})
	}).Methods("POST")

	router.HandleFunc("/migrations/{id}", func(w http.ResponseWriter, r *http.Request){
		w.Header().Set("Content-Type","application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"code":"not_found","message":"unknown migration"}})
	}).Methods("GET")

	router.HandleFunc("/migrations/{id}", func(w http.ResponseWriter, r *http.Request){
		w.Header().Set("Content-Type","application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"code":"unsupported_feature","message":"cancel not supported in core mode"}})
	}).Methods("DELETE")
}

