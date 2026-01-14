//go:build !experimental

package vm

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
)

// RegisterCoreNodeRoutes registers core-mode node endpoints
func RegisterCoreNodeRoutes(router *mux.Router, engine *orchestration.DefaultOrchestrationEngine) {
	router.HandleFunc("/nodes", func(w http.ResponseWriter, r *http.Request){
		statuses := engine.GetNodeStatuses()
		list := make([]map[string]interface{}, 0, len(statuses))
		for id, st := range statuses {
			list = append(list, map[string]interface{}{"id": id, "healthy": st.Healthy, "lastChange": st.LastChange})
		}
		w.Header().Set("Content-Type","application/json")
		json.NewEncoder(w).Encode(list)
	}).Methods("GET")

	router.HandleFunc("/nodes/{id}", func(w http.ResponseWriter, r *http.Request){
		vars := mux.Vars(r)
		nodeID := vars["id"]
		st, ok := engine.GetNodeStatuses()[nodeID]
		if !ok { w.WriteHeader(http.StatusNotFound); json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]interface{}{"code":"not_found","message":"node not found"}}); return }
		w.Header().Set("Content-Type","application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"id": nodeID, "healthy": st.Healthy, "lastChange": st.LastChange})
	}).Methods("GET")
}

