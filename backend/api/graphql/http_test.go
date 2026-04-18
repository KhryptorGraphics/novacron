package graphql

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	corestorage "github.com/khryptorgraphics/novacron/backend/core/storage"
)

func TestNewVolumeHTTPHandlerSupportsVolumeLifecycle(t *testing.T) {
	t.Parallel()

	store, err := corestorage.NewStorageManager(corestorage.StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	handler := NewVolumeHTTPHandler(NewResolverWithVolumeStore(nil, nil, store))

	createBody := requestEnvelope{
		Query: "mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name size tier vmId } }",
		Variables: map[string]interface{}{
			"input": map[string]interface{}{
				"name": "phase3-volume",
				"size": 8,
				"tier": "HOT",
				"vmId": "vm-42",
			},
		},
	}

	createReq := httptest.NewRequest(http.MethodPost, "/graphql", marshalEnvelope(t, createBody))
	createRec := httptest.NewRecorder()
	handler.ServeHTTP(createRec, createReq)

	if createRec.Code != http.StatusOK {
		t.Fatalf("expected create volume 200, got %d (%s)", createRec.Code, createRec.Body.String())
	}

	var createResp responseEnvelope
	if err := json.NewDecoder(createRec.Body).Decode(&createResp); err != nil {
		t.Fatalf("decode create response: %v", err)
	}

	created, ok := createResp.Data["createVolume"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected createVolume object, got %#v", createResp.Data["createVolume"])
	}

	createdID, ok := created["id"].(string)
	if !ok || createdID == "" {
		t.Fatalf("expected created volume id, got %#v", created["id"])
	}

	listBody := requestEnvelope{
		Query: "query Volumes { volumes { id name tier } }",
	}

	listReq := httptest.NewRequest(http.MethodPost, "/graphql", marshalEnvelope(t, listBody))
	listRec := httptest.NewRecorder()
	handler.ServeHTTP(listRec, listReq)

	if listRec.Code != http.StatusOK {
		t.Fatalf("expected list volumes 200, got %d (%s)", listRec.Code, listRec.Body.String())
	}

	var listResp responseEnvelope
	if err := json.NewDecoder(listRec.Body).Decode(&listResp); err != nil {
		t.Fatalf("decode list response: %v", err)
	}

	volumes, ok := listResp.Data["volumes"].([]interface{})
	if !ok || len(volumes) != 1 {
		t.Fatalf("expected one listed volume, got %#v", listResp.Data["volumes"])
	}

	volumeEntry, ok := volumes[0].(map[string]interface{})
	if !ok || volumeEntry["id"] != createdID {
		t.Fatalf("expected listed volume %s, got %#v", createdID, volumes[0])
	}

	changeBody := requestEnvelope{
		Query: "mutation ChangeVolumeTier($id: ID!, $tier: StorageTier!) { changeVolumeTier(id: $id, tier: $tier) { id tier } }",
		Variables: map[string]interface{}{
			"id":   createdID,
			"tier": "COLD",
		},
	}

	changeReq := httptest.NewRequest(http.MethodPost, "/graphql", marshalEnvelope(t, changeBody))
	changeRec := httptest.NewRecorder()
	handler.ServeHTTP(changeRec, changeReq)

	if changeRec.Code != http.StatusOK {
		t.Fatalf("expected change volume tier 200, got %d (%s)", changeRec.Code, changeRec.Body.String())
	}

	var changeResp responseEnvelope
	if err := json.NewDecoder(changeRec.Body).Decode(&changeResp); err != nil {
		t.Fatalf("decode change response: %v", err)
	}

	updated, ok := changeResp.Data["changeVolumeTier"].(map[string]interface{})
	if !ok || updated["tier"] != "COLD" {
		t.Fatalf("expected changeVolumeTier tier COLD, got %#v", changeResp.Data["changeVolumeTier"])
	}
}

func TestNewVolumeHTTPHandlerRejectsUnsupportedOperations(t *testing.T) {
	t.Parallel()

	handler := NewVolumeHTTPHandler(NewResolver(nil, nil))

	body := requestEnvelope{
		Query: "query Unsupported { vm(id: \"1\") { id } }",
	}

	req := httptest.NewRequest(http.MethodPost, "/graphql", marshalEnvelope(t, body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for unsupported operation, got %d", rec.Code)
	}

	var resp responseEnvelope
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode unsupported response: %v", err)
	}
	if len(resp.Errors) == 0 {
		t.Fatalf("expected graphql errors, got %#v", resp)
	}
}

func marshalEnvelope(t *testing.T, payload requestEnvelope) *bytes.Reader {
	t.Helper()

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal request envelope: %v", err)
	}
	return bytes.NewReader(raw)
}
