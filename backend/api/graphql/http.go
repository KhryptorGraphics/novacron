package graphql

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type requestEnvelope struct {
	Query         string                 `json:"query"`
	Variables     map[string]interface{} `json:"variables"`
	OperationName string                 `json:"operationName"`
}

type responseEnvelope struct {
	Data   map[string]interface{} `json:"data,omitempty"`
	Errors []map[string]string    `json:"errors,omitempty"`
}

func NewVolumeHTTPHandler(resolver *Resolver) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeGraphQLResponse(w, http.StatusMethodNotAllowed, responseEnvelope{
				Errors: []map[string]string{{"message": "only POST is supported"}},
			})
			return
		}

		var request requestEnvelope
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			writeGraphQLResponse(w, http.StatusBadRequest, responseEnvelope{
				Errors: []map[string]string{{"message": "invalid GraphQL request body"}},
			})
			return
		}

		data, err := executeVolumeOperation(r.Context(), resolver, request)
		if err != nil {
			writeGraphQLResponse(w, http.StatusBadRequest, responseEnvelope{
				Errors: []map[string]string{{"message": err.Error()}},
			})
			return
		}

		writeGraphQLResponse(w, http.StatusOK, responseEnvelope{Data: data})
	})
}

func executeVolumeOperation(ctx context.Context, resolver *Resolver, request requestEnvelope) (map[string]interface{}, error) {
	query := strings.TrimSpace(request.Query)
	switch {
	case strings.Contains(query, "createVolume"):
		inputValue, ok := request.Variables["input"].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("createVolume requires variables.input")
		}

		input, err := decodeCreateVolumeInput(inputValue)
		if err != nil {
			return nil, err
		}

		volume, err := resolver.CreateVolume(ctx, struct{ Input CreateVolumeInput }{Input: input})
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"createVolume": volume}, nil

	case strings.Contains(query, "changeVolumeTier"):
		id, _ := request.Variables["id"].(string)
		tier, _ := request.Variables["tier"].(string)
		if strings.TrimSpace(id) == "" || strings.TrimSpace(tier) == "" {
			return nil, fmt.Errorf("changeVolumeTier requires variables.id and variables.tier")
		}

		volume, err := resolver.ChangeVolumeTier(ctx, struct {
			ID   string
			Tier string
		}{ID: id, Tier: tier})
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"changeVolumeTier": volume}, nil

	case strings.Contains(query, "volumes"):
		var pagination *PaginationInput
		if rawPagination, ok := request.Variables["pagination"].(map[string]interface{}); ok {
			decodedPagination, err := decodePaginationInput(rawPagination)
			if err != nil {
				return nil, err
			}
			pagination = decodedPagination
		}

		volumes, err := resolver.Volumes(ctx, struct{ Pagination *PaginationInput }{Pagination: pagination})
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"volumes": volumes}, nil

	default:
		return nil, fmt.Errorf("unsupported GraphQL operation on canonical backend")
	}
}

func decodeCreateVolumeInput(raw map[string]interface{}) (CreateVolumeInput, error) {
	input := CreateVolumeInput{}

	name, _ := raw["name"].(string)
	tier, _ := raw["tier"].(string)
	if strings.TrimSpace(name) == "" {
		return input, fmt.Errorf("createVolume input.name is required")
	}
	if strings.TrimSpace(tier) == "" {
		return input, fmt.Errorf("createVolume input.tier is required")
	}

	sizeFloat, ok := raw["size"].(float64)
	if !ok || int(sizeFloat) <= 0 {
		return input, fmt.Errorf("createVolume input.size must be a positive integer")
	}

	input.Name = name
	input.Size = int(sizeFloat)
	input.Tier = tier
	if vmID, ok := raw["vmId"].(string); ok && strings.TrimSpace(vmID) != "" {
		input.VMID = &vmID
	}

	return input, nil
}

func decodePaginationInput(raw map[string]interface{}) (*PaginationInput, error) {
	pagination := &PaginationInput{}

	if pageValue, ok := raw["page"].(float64); ok {
		pagination.Page = int(pageValue)
	}
	if pageSizeValue, ok := raw["pageSize"].(float64); ok {
		pagination.PageSize = int(pageSizeValue)
	}
	if pagination.Page < 0 || pagination.PageSize < 0 {
		return nil, fmt.Errorf("pagination values must be non-negative")
	}

	return pagination, nil
}

func writeGraphQLResponse(w http.ResponseWriter, status int, payload responseEnvelope) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
