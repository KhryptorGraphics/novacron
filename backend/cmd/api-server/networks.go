//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

// Networks catalog (novacron-ed7). A catalog network is a NovaCron-managed
// Linux bridge on this node (backend/core/network/provision): creating one
// inserts the networks row and provisions the bridge in one step, deleting
// one removes both. KVM guests created with network_id (POST /vms) get their
// primary NIC bridged onto it, recorded in vms.network_id, which blocks
// deleting a network that still has VMs.
//
//	GET    /networks       any authenticated caller (vm_count is org-scoped)
//	POST   /networks       admin / super-admin
//	GET    /networks/{id}  any authenticated caller
//	DELETE /networks/{id}  admin / super-admin
//
// The catalog is node-local: each node's database lists the bridges of that
// node, so a VM on a catalog network is always placed on this node.

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/network/provision"
	"github.com/lib/pq"
)

// networkNameRE: 1-63 characters, letters/digits/'.'/'_'/'-', starting with a
// letter or digit (mirrors the networks_name_format CHECK in 000018).
var networkNameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`)

// hostNetworkTimeout bounds the host-side rollback/reconcile calls that must
// run even when the request context is already gone.
const hostNetworkTimeout = 30 * time.Second

// newNetworkProvisioner builds this node's provisioner from the environment:
//
//	NOVACRON_NETWORK_UPLINK       host interface VLAN networks are tagged onto
//	                              (unset: VLAN networks are refused)
//	NOVACRON_QEMU_BRIDGE_ACL_DIR  qemu-bridge-helper config dir kept in sync
//	                              (default /etc/qemu; "none" = the operator
//	                              manages bridge.conf, e.g. "allow all")
func newNetworkProvisioner() provision.Provisioner {
	p := &provision.Linux{Uplink: strings.TrimSpace(os.Getenv("NOVACRON_NETWORK_UPLINK"))}
	switch dir := strings.TrimSpace(os.Getenv("NOVACRON_QEMU_BRIDGE_ACL_DIR")); dir {
	case "none":
	case "":
		p.ACL = &provision.BridgeACL{Dir: "/etc/qemu"}
	default:
		p.ACL = &provision.BridgeACL{Dir: dir}
	}
	return p
}

func registerNetworkRoutes(router *mux.Router, db *sql.DB, prov provision.Provisioner) {
	adminOnly := requireAnyRole("admin", "super-admin")
	router.HandleFunc("/networks", listNetworksHandler(db)).Methods(http.MethodGet)
	router.Handle("/networks", adminOnly(createNetworkHandler(db, prov))).Methods(http.MethodPost)
	router.HandleFunc("/networks/{id}", getNetworkHandler(db)).Methods(http.MethodGet)
	router.Handle("/networks/{id}", adminOnly(deleteNetworkHandler(db, prov))).Methods(http.MethodDelete)
}

// networkColumns is the SELECT list scanned by scanNetwork; the vm_count
// column is appended by networkVMCount.
const networkColumns = `n.id, n.name, n.bridge, n.cidr::text, host(n.gateway), n.vlan_id, n.mtu, n.created_by, n.created_at, n.updated_at`

type networkRecord struct {
	ID, Name, Bridge, CIDR string
	Gateway, CreatedBy     sql.NullString
	VLANID                 sql.NullInt64
	MTU                    int
	CreatedAt, UpdatedAt   time.Time
	VMCount                int64
}

type rowScanner interface {
	Scan(dest ...interface{}) error
}

func scanNetwork(row rowScanner) (networkRecord, error) {
	var n networkRecord
	err := row.Scan(&n.ID, &n.Name, &n.Bridge, &n.CIDR, &n.Gateway, &n.VLANID, &n.MTU, &n.CreatedBy, &n.CreatedAt, &n.UpdatedAt, &n.VMCount)
	return n, err
}

func (n networkRecord) JSON() map[string]interface{} {
	var vlan interface{}
	if n.VLANID.Valid {
		vlan = n.VLANID.Int64
	}
	return map[string]interface{}{
		"id":         n.ID,
		"name":       n.Name,
		"bridge":     n.Bridge,
		"cidr":       n.CIDR,
		"gateway":    nullableString(n.Gateway),
		"vlan_id":    vlan,
		"mtu":        n.MTU,
		"created_by": nullableString(n.CreatedBy),
		"vm_count":   n.VMCount,
		"created_at": n.CreatedAt.UTC().Format(time.RFC3339),
		"updated_at": n.UpdatedAt.UTC().Format(time.RFC3339),
	}
}

// spec rebuilds the host-side description of a stored network.
func (n networkRecord) spec() (provision.Spec, error) {
	return provision.NewSpec(n.ID, n.CIDR, n.Gateway.String, int(n.VLANID.Int64), n.MTU)
}

// networkVMCount returns the vm_count select expression for the caller's
// tenancy scope (see requireOrgScope): admins count every VM on the network,
// everyone else only the VMs of their own org, so the catalog never reveals
// another org's footprint. The expression's parameter is $argIndex.
func networkVMCount(ctx context.Context, argIndex int) (string, []interface{}) {
	scopeOrg, isAdmin, _ := requireOrgScope(ctx, nil, "")
	switch {
	case isAdmin:
		return `(SELECT count(*) FROM vms v WHERE v.network_id = n.id)`, nil
	case scopeOrg != "":
		return fmt.Sprintf(`(SELECT count(*) FROM vms v WHERE v.network_id = n.id AND v.organization_id = $%d)`, argIndex), []interface{}{scopeOrg}
	default:
		return `(SELECT count(*) FROM vms v WHERE v.network_id = n.id AND v.organization_id IS NULL)`, nil
	}
}

func listNetworksHandler(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		countExpr, args := networkVMCount(r.Context(), 1)
		rows, err := db.QueryContext(r.Context(), `SELECT `+networkColumns+`, `+countExpr+` FROM networks n ORDER BY n.created_at DESC, n.name`, args...)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query networks")
			return
		}
		defer rows.Close()
		networks := make([]map[string]interface{}, 0)
		for rows.Next() {
			n, err := scanNetwork(rows)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to read networks")
				return
			}
			networks = append(networks, n.JSON())
		}
		if err := rows.Err(); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to read networks")
			return
		}
		writeJSON(w, http.StatusOK, networks)
	}
}

func getNetworkHandler(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, ok := networkIDParam(r)
		if !ok {
			writeJSONError(w, http.StatusNotFound, "network not found")
			return
		}
		countExpr, args := networkVMCount(r.Context(), 2)
		n, err := scanNetwork(db.QueryRowContext(r.Context(), `SELECT `+networkColumns+`, `+countExpr+` FROM networks n WHERE n.id = $1`, append([]interface{}{id}, args...)...))
		if errors.Is(err, sql.ErrNoRows) {
			writeJSONError(w, http.StatusNotFound, "network not found")
			return
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query network")
			return
		}
		writeJSON(w, http.StatusOK, n.JSON())
	}
}

// networkIDParam returns the canonical form of the {id} path variable; a
// non-UUID id cannot name a network.
func networkIDParam(r *http.Request) (string, bool) {
	id, err := uuid.Parse(mux.Vars(r)["id"])
	if err != nil {
		return "", false
	}
	return id.String(), true
}

const sqlInsertNetwork = `
	INSERT INTO networks (id, name, bridge, cidr, gateway, vlan_id, mtu, created_by, created_at, updated_at)
	VALUES ($1, $2, $3, $4::cidr, NULLIF($5, '')::inet, $6, $7,
		(SELECT u.id FROM users u WHERE u.id = NULLIF($8, '')::uuid), NOW(), NOW())
	RETURNING created_by, created_at, updated_at`

// createNetworkHandler validates the request, inserts the row and provisions
// the bridge inside one transaction: the row commits only after the host
// side is in place, and a failed provision or commit removes whatever part of
// the bridge was created.
func createNetworkHandler(db *sql.DB, prov provision.Provisioner) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Name    string `json:"name"`
			CIDR    string `json:"cidr"`
			Gateway string `json:"gateway"`
			VLANID  *int   `json:"vlan_id"`
			MTU     *int   `json:"mtu"`
		}
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
			return
		}
		name := strings.TrimSpace(req.Name)
		if !networkNameRE.MatchString(name) {
			writeJSONError(w, http.StatusBadRequest, "name must be 1-63 characters of letters, digits, '.', '_' or '-', starting with a letter or digit")
			return
		}
		vlan := 0
		if req.VLANID != nil {
			if *req.VLANID < 1 || *req.VLANID > 4094 {
				writeJSONError(w, http.StatusBadRequest, "vlan_id must be between 1 and 4094 (omit it for an untagged network)")
				return
			}
			vlan = *req.VLANID
		}
		mtu := provision.DefaultMTU
		if req.MTU != nil {
			mtu = *req.MTU
		}
		id := uuid.NewString()
		spec, err := provision.NewSpec(id, req.CIDR, req.Gateway, vlan, mtu)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}
		gateway := ""
		if spec.Gateway.IsValid() {
			gateway = spec.Gateway.String()
		}
		var vlanArg interface{}
		if vlan != 0 {
			vlanArg = vlan
		}
		userID, _ := r.Context().Value("user_id").(string)
		if _, err := uuid.Parse(userID); err != nil {
			userID = ""
		}

		ctx := r.Context()
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create network")
			return
		}
		defer tx.Rollback() // no-op after Commit

		n := networkRecord{ID: id, Name: name, Bridge: spec.Bridge, CIDR: spec.CIDR.String(), MTU: mtu,
			Gateway: sql.NullString{String: gateway, Valid: gateway != ""}, VLANID: sql.NullInt64{Int64: int64(vlan), Valid: vlan != 0}}
		err = tx.QueryRowContext(ctx, sqlInsertNetwork, id, name, spec.Bridge, n.CIDR, gateway, vlanArg, mtu, userID).
			Scan(&n.CreatedBy, &n.CreatedAt, &n.UpdatedAt)
		if err != nil {
			switch pgErrorCode(err) {
			case "23505": // unique_violation (networks_name_lower_key)
				writeJSONError(w, http.StatusConflict, fmt.Sprintf("a network named %q already exists", name))
			case "23P01": // exclusion_violation (networks_cidr_no_overlap)
				writeJSONError(w, http.StatusConflict, fmt.Sprintf("cidr %s overlaps an existing network", n.CIDR))
			default:
				writeJSONError(w, http.StatusInternalServerError, "failed to create network")
			}
			return
		}

		if err := prov.Ensure(ctx, spec); err != nil {
			_ = tx.Rollback()
			undoHostNetwork(prov.Remove, spec, "create rollback")
			writeProvisionError(w, err)
			return
		}
		if err := tx.Commit(); err != nil {
			undoHostNetwork(prov.Remove, spec, "create rollback")
			writeJSONError(w, http.StatusInternalServerError, "failed to create network")
			return
		}
		writeJSON(w, http.StatusCreated, n.JSON())
	}
}

// deleteNetworkHandler refuses while any VM is attached (409 with their ids),
// then deletes the row and removes the bridge in one transaction: the row is
// only gone once the host side is, and a failed commit re-provisions it.
func deleteNetworkHandler(db *sql.DB, prov provision.Provisioner) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, ok := networkIDParam(r)
		if !ok {
			writeJSONError(w, http.StatusNotFound, "network not found")
			return
		}
		ctx := r.Context()
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}
		defer tx.Rollback() // no-op after Commit

		// FOR UPDATE conflicts with the FK KEY SHARE lock a concurrent VM
		// create takes on this row, so no VM can attach between the check
		// below and the DELETE.
		n, err := scanNetwork(tx.QueryRowContext(ctx, `SELECT `+networkColumns+`, 0 FROM networks n WHERE n.id = $1 FOR UPDATE`, id))
		if errors.Is(err, sql.ErrNoRows) {
			writeJSONError(w, http.StatusNotFound, "network not found")
			return
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}
		vmIDs, err := attachedVMIDs(ctx, tx, id)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}
		if len(vmIDs) > 0 {
			writeNetworkInUse(w, n.Name, vmIDs)
			return
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM networks WHERE id = $1`, id); err != nil {
			if pgErrorCode(err) == "23503" { // foreign_key_violation: vms.network_id
				writeJSONError(w, http.StatusConflict, fmt.Sprintf("network %q is in use by a VM", n.Name))
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}

		spec, specErr := n.spec()
		if specErr != nil { // unreachable for rows the CHECKs admitted; Remove needs only the id
			spec = provision.Spec{NetworkID: n.ID, Bridge: n.Bridge}
		}
		if err := prov.Remove(ctx, spec); err != nil {
			writeProvisionError(w, err)
			return
		}
		if err := tx.Commit(); err != nil {
			if specErr == nil {
				undoHostNetwork(prov.Ensure, spec, "delete rollback")
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"id": id, "name": n.Name, "status": "deleted"})
	}
}

func attachedVMIDs(ctx context.Context, tx *sql.Tx, networkID string) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id::text FROM vms WHERE network_id = $1 ORDER BY created_at, id`, networkID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var vmID string
		if err := rows.Scan(&vmID); err != nil {
			return nil, err
		}
		ids = append(ids, vmID)
	}
	return ids, rows.Err()
}

func writeNetworkInUse(w http.ResponseWriter, name string, vmIDs []string) {
	writeJSON(w, http.StatusConflict, map[string]interface{}{
		"error":  fmt.Sprintf("network %q is in use by %d VM(s); delete them first", name, len(vmIDs)),
		"vm_ids": vmIDs,
	})
}

// writeProvisionError maps host provisioning failures: conditions the caller
// can act on are 4xx, anything else (permissions, kernel errors) is a 500
// whose message names the host-side cause for the admin.
func writeProvisionError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, provision.ErrNoUplink):
		writeJSONError(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, provision.ErrAddressConflict), errors.Is(err, provision.ErrForeignLink), errors.Is(err, provision.ErrBridgeInUse):
		writeJSONError(w, http.StatusConflict, err.Error())
	default:
		writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("network provisioning failed: %v", err))
	}
}

// undoHostNetwork runs a compensating provisioner call on its own deadline
// (the request may already be cancelled); a failure is logged because the
// response already reports the original error.
func undoHostNetwork(op func(context.Context, provision.Spec) error, spec provision.Spec, what string) {
	ctx, cancel := context.WithTimeout(context.Background(), hostNetworkTimeout)
	defer cancel()
	if err := op(ctx, spec); err != nil {
		log.Printf("networks: %s of %s failed: %v", what, spec.Bridge, err)
	}
}

// reconcileNetworks re-provisions every catalog network at boot: bridges do
// not survive a host reboot, and guests on them cannot start until they are
// back. Failures are logged per network and do not stop the server.
func reconcileNetworks(db *sql.DB, prov provision.Provisioner) {
	ctx, cancel := context.WithTimeout(context.Background(), hostNetworkTimeout)
	defer cancel()
	rows, err := db.QueryContext(ctx, `SELECT `+networkColumns+`, 0 FROM networks n ORDER BY n.created_at`)
	if err != nil {
		log.Printf("networks: boot reconcile skipped: %v", err)
		return
	}
	var records []networkRecord
	for rows.Next() {
		n, err := scanNetwork(rows)
		if err != nil {
			log.Printf("networks: boot reconcile: %v", err)
			continue
		}
		records = append(records, n)
	}
	rows.Close()
	for _, n := range records {
		spec, err := n.spec()
		if err == nil {
			err = prov.Ensure(ctx, spec)
		}
		if err != nil {
			log.Printf("networks: boot reconcile of %s (%s) failed: %v", n.Name, n.Bridge, err)
		}
	}
}

// errUnknownNetwork: a VM create named a network_id that is not in this
// node's catalog (a client error, unlike a failed lookup).
var errUnknownNetwork = errors.New("unknown network_id")

// resolveCatalogNetwork validates a VM create's network_id: it must name a
// catalog network on this node. Returns the canonical id.
func resolveCatalogNetwork(ctx context.Context, db *sql.DB, networkID string) (string, error) {
	id, err := uuid.Parse(strings.TrimSpace(networkID))
	if err != nil {
		return "", fmt.Errorf("%w: %q is not a network id", errUnknownNetwork, networkID)
	}
	var exists bool
	if err := db.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM networks WHERE id = $1)`, id.String()).Scan(&exists); err != nil {
		return "", fmt.Errorf("failed to look up network_id: %w", err)
	}
	if !exists {
		return "", fmt.Errorf("%w: no network %s on this node", errUnknownNetwork, id)
	}
	return id.String(), nil
}

func pgErrorCode(err error) string {
	var pqErr *pq.Error
	if errors.As(err, &pqErr) {
		return string(pqErr.Code)
	}
	return ""
}
