// Command loadtest drives the canonical NovaCron api-server's real HTTP API
// with concurrent VM create/migrate/delete operations and reports pass/fail
// against configurable latency and error-rate thresholds.
//
// It is a load-test CLIENT only -- it does not start any NovaCron process
// itself. Start a real api-server first (see backend/cmd/loadtest/README.md),
// then point this at it:
//
//	go run . -api-url http://localhost:8090 -db-url "postgresql://postgres:postgres@localhost:5432/novacron?sslmode=disable" -concurrency 5 -creates 20
//
// If -db-url is given, it seeds (or reuses) a real "operator"-role user
// directly in Postgres -- self-registration via /auth/register only grants
// role "user", which the VM routes' RBAC (require("operator", ...)) rejects,
// so a load test that only self-registers can never legitimately exercise
// create/migrate/delete. Seeding uses the same bcrypt format the server's own
// SimpleAuthManager uses, so the seeded user logs in through the real
// /auth/login endpoint like any other account.
package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/lib/pq"
	"golang.org/x/crypto/bcrypt"
)

func main() {
	apiURL := flag.String("api-url", envOr("API_URL", "http://localhost:8090"), "base URL of a running api-server")
	dbURL := flag.String("db-url", envOr("DB_URL", ""), "if set, seed a real operator-role test user directly in this Postgres DB before running")
	username := flag.String("username", "loadtest-operator", "test account username (created if -db-url is set)")
	email := flag.String("email", "loadtest-operator@example.test", "test account email")
	password := flag.String("password", "LoadTest123!", "test account password")

	concurrency := flag.Int("concurrency", 5, "concurrent workers for the create/delete phases")
	creates := flag.Int("creates", 20, "total VM create operations across all workers")
	image := flag.String("image", defaultCirrosImage(), "guest disk image path passed to each VM create")
	memoryMB := flag.Int("memory-mb", 256, "VM memory size for created VMs")
	diskGB := flag.Int("disk-size-gb", 1, "VM disk size for created VMs")

	withMigrate := flag.Bool("with-migrate", false, "also run a migrate phase (needs -migrate-target, a peer node NOVACRON_PEERS entry)")
	migrateTarget := flag.String("migrate-target", "", "target_node id for the migrate phase (see NOVACRON_PEERS on the server)")
	migrateCount := flag.Int("migrate-count", 3, "how many of the created VMs to start+migrate")
	peerAPIURL := flag.String("peer-api-url", "", "base URL of the migrate-target peer's own api-server -- needed so this tool can delete/query VMs the cluster scheduler's best-fit dispatch placed on the peer at create time, or that a successful migrate moved there (cluster dispatch places VMs across nodes at create, but does not proxy delete/get -- only the owning node's own API can act on them)")
	peerDBURL := flag.String("peer-db-url", "", "if set, also seed the same operator user (matching id) in the peer's own Postgres DB -- required for cross-node cluster-dispatch creates to pass the dest's owner_id foreign key")

	reqTimeout := flag.Duration("request-timeout", 30*time.Second, "per-request HTTP timeout")
	maxErrorRate := flag.Float64("max-error-rate", 0.05, "fail if any phase's error rate exceeds this fraction")
	maxP95CreateMS := flag.Int("max-p95-create-ms", 8000, "fail if create p95 latency exceeds this many ms")
	maxP95DeleteMS := flag.Int("max-p95-delete-ms", 4000, "fail if delete p95 latency exceeds this many ms")
	maxP95MigrateMS := flag.Int("max-p95-migrate-ms", 30000, "fail if migrate p95 latency exceeds this many ms")
	flag.Parse()

	if *dbURL != "" {
		if err := seedOperatorUser(*dbURL, *username, *email, *password); err != nil {
			fatalf("seed operator user: %v", err)
		}
		fmt.Printf("seeded/confirmed operator user %q in %s\n", *username, redactDBURL(*dbURL))
	}
	if *peerDBURL != "" {
		if err := seedOperatorUser(*peerDBURL, *username, *email, *password); err != nil {
			fatalf("seed operator user on peer: %v", err)
		}
		fmt.Printf("seeded/confirmed operator user %q in peer %s\n", *username, redactDBURL(*peerDBURL))
	}

	client := &http.Client{Timeout: *reqTimeout}
	token, err := login(client, *apiURL, *username, *password)
	if err != nil {
		fatalf("login as %q failed: %v (did you mean to pass -db-url so this tool can seed an operator account?)", *username, err)
	}
	fmt.Printf("authenticated as %q (operator role expected)\n", *username)

	report := &Report{}

	// ---- Phase 1: concurrent create ----
	// vmNode tracks which node's API currently owns each VM: the cluster
	// scheduler's best-fit placement can dispatch a create straight to a peer
	// (see cluster.go's clusteredCreateHandler), and a later migrate moves
	// ownership again -- delete/start/get are NOT cluster-dispatched (only
	// create is), so this tool must call the right node directly.
	ids, vmNode := runCreatePhase(client, *apiURL, token, *concurrency, *creates, *image, *memoryMB, *diskGB, report)
	fmt.Printf("created %d/%d VMs\n", len(ids), *creates)

	// ---- Phase 2: optional migrate ----
	if *withMigrate {
		if *migrateTarget == "" {
			fmt.Println("skip migrate phase: -with-migrate set but -migrate-target is empty")
		} else {
			// Only VMs the scheduler placed on THIS node (api-url) can be
			// started+migrated through this node's API; ones best-fit already
			// sent to the peer are excluded from this phase (not a failure --
			// they're just already elsewhere).
			var local []string
			for _, id := range ids {
				if vmNode[id] == "" {
					local = append(local, id)
				}
			}
			n := *migrateCount
			if n > len(local) {
				n = len(local)
			}
			runMigratePhase(client, *apiURL, token, local[:n], *migrateTarget, report, vmNode)
		}
	}

	// ---- Phase 3: concurrent delete (clean up everything we created) ----
	runDeletePhase(client, *apiURL, *peerAPIURL, token, ids, vmNode, *concurrency, report)

	// ---- Report + thresholds ----
	pass := report.Print(*maxErrorRate, *maxP95CreateMS, *maxP95DeleteMS, *maxP95MigrateMS)
	if !pass {
		os.Exit(1)
	}
}

func fatalf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "loadtest: "+format+"\n", args...)
	os.Exit(2)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// defaultCirrosImage guesses the same real cirros test image the backend/core/vm
// real-qemu tests use, so this tool works out of the box on a dev box that has
// already run those tests (see findCirrosImage in driver_kvm_migrate_test.go).
func defaultCirrosImage() string {
	home, _ := os.UserHomeDir()
	arch := "aarch64"
	// best-effort: most dev/CI boxes running this are the same arch either way;
	// override with -image if not.
	for _, dir := range []string{"novacron-run", "novacron-e2e"} {
		for _, a := range []string{arch, "x86_64"} {
			p := home + "/" + dir + "/images/cirros-0.6.2-" + a + "-disk.img"
			if fi, err := os.Stat(p); err == nil && fi.Size() > 0 {
				return p
			}
		}
	}
	return ""
}

func redactDBURL(u string) string {
	if i := strings.Index(u, "@"); i > 0 {
		if j := strings.Index(u, "://"); j > 0 && j < i {
			return u[:j+3] + "***@" + u[i+1:]
		}
	}
	return u
}

// seedOperatorUser ensures a real operator-role user exists, using the exact
// schema and bcrypt format backend/cmd/api-server/main.go's runMigrations +
// SimpleAuthManager use, so the account is indistinguishable from one created
// through normal admin provisioning.
func seedOperatorUser(dbURL, username, email, password string) error {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	defer db.Close()
	if err := db.Ping(); err != nil {
		return fmt.Errorf("ping db (is the api-server's Postgres up and reachable at this URL?): %w", err)
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("hash password: %w", err)
	}

	// Match the api-server's own inline schema (runMigrations in main.go):
	// users(id SERIAL, username, email, password_hash, role VARCHAR DEFAULT
	// 'user', active BOOLEAN, tenant_id VARCHAR). Upsert so reruns are safe.
	_, err = db.Exec(`
		INSERT INTO users (username, email, password_hash, role, active, tenant_id)
		VALUES ($1, $2, $3, 'operator', true, 'default')
		ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = 'operator', active = true
	`, username, email, string(hash))
	if err != nil {
		return fmt.Errorf("upsert operator user (has the api-server started at least once against this DB, to create the users table? runMigrations runs on server startup): %w", err)
	}
	return nil
}

// ---- HTTP helpers ----

func login(client *http.Client, apiURL, username, password string) (string, error) {
	body, _ := json.Marshal(map[string]string{"username": username, "password": password})
	resp, err := client.Post(apiURL+"/auth/login", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, string(data))
	}
	var parsed struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("decode login response: %w", err)
	}
	if parsed.Token == "" {
		return "", fmt.Errorf("login response had no token: %s", string(data))
	}
	return parsed.Token, nil
}

// ---- Result tracking ----

// Result is one timed HTTP operation's outcome.
type Result struct {
	Latency time.Duration
	Err     error
}

// PhaseStats holds the raw results for one operation type (create/migrate/delete).
type PhaseStats struct {
	mu      sync.Mutex
	results []Result
}

func (p *PhaseStats) add(r Result) {
	p.mu.Lock()
	p.results = append(p.results, r)
	p.mu.Unlock()
}

func (p *PhaseStats) summarize() (count, errCount int, errRate float64, min, avg, p50, p95, p99, max time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	count = len(p.results)
	if count == 0 {
		return
	}
	lat := make([]time.Duration, 0, count)
	var sum time.Duration
	for _, r := range p.results {
		if r.Err != nil {
			errCount++
			continue
		}
		lat = append(lat, r.Latency)
		sum += r.Latency
	}
	if len(lat) == 0 {
		errRate = 1.0
		return
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	errRate = float64(errCount) / float64(count)
	min = lat[0]
	max = lat[len(lat)-1]
	avg = sum / time.Duration(len(lat))
	p50 = percentile(lat, 0.50)
	p95 = percentile(lat, 0.95)
	p99 = percentile(lat, 0.99)
	return
}

func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// Report aggregates the three phases for the final printout + threshold check.
type Report struct {
	Create  PhaseStats
	Migrate PhaseStats
	Delete  PhaseStats
}

func (r *Report) Print(maxErrorRate float64, maxP95CreateMS, maxP95DeleteMS, maxP95MigrateMS int) bool {
	pass := true
	fmt.Println()
	fmt.Println("==================== LOAD TEST REPORT ====================")
	pass = printPhase("create", &r.Create, maxErrorRate, time.Duration(maxP95CreateMS)*time.Millisecond) && pass
	if len(r.Migrate.results) > 0 {
		pass = printPhase("migrate", &r.Migrate, maxErrorRate, time.Duration(maxP95MigrateMS)*time.Millisecond) && pass
	}
	pass = printPhase("delete", &r.Delete, maxErrorRate, time.Duration(maxP95DeleteMS)*time.Millisecond) && pass
	fmt.Println("============================================================")
	if pass {
		fmt.Println("RESULT: PASS (all phases within thresholds)")
	} else {
		fmt.Println("RESULT: FAIL (see phase(s) marked FAIL above)")
	}
	return pass
}

func printPhase(name string, p *PhaseStats, maxErrorRate float64, maxP95 time.Duration) bool {
	count, errCount, errRate, min, avg, p50, p95, p99, max := p.summarize()
	status := "PASS"
	ok := true
	if errRate > maxErrorRate {
		status = "FAIL"
		ok = false
	}
	if p95 > maxP95 {
		status = "FAIL"
		ok = false
	}
	fmt.Printf("\n[%s] %s\n", strings.ToUpper(name), status)
	fmt.Printf("  count=%d errors=%d error_rate=%.1f%% (threshold <%.1f%%)\n", count, errCount, errRate*100, maxErrorRate*100)
	fmt.Printf("  latency  min=%s avg=%s p50=%s p95=%s (threshold <%s) p99=%s max=%s\n",
		min.Round(time.Millisecond), avg.Round(time.Millisecond), p50.Round(time.Millisecond),
		p95.Round(time.Millisecond), maxP95, p99.Round(time.Millisecond), max.Round(time.Millisecond))
	return ok
}

// ---- Phases ----

// runCreatePhase fires `total` concurrent creates and returns every VM id
// created plus a node-ownership map: vmNode[id] == "" means the scheduler kept
// it on this node (apiURL); a non-empty value is the peer node id it
// best-fit-dispatched to instead (see clusteredCreateHandler in cluster.go).
func runCreatePhase(client *http.Client, apiURL, token string, concurrency, total int, image string, memoryMB, diskGB int, report *Report) ([]string, map[string]string) {
	var mu sync.Mutex
	var ids []string
	vmNode := make(map[string]string)
	var counter int64

	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for {
				n := atomic.AddInt64(&counter, 1)
				if n > int64(total) {
					return
				}
				name := fmt.Sprintf("loadtest-vm-%d-%d", time.Now().UnixNano(), n)
				body, _ := json.Marshal(map[string]interface{}{
					"name":         name,
					"type":         "kvm",
					"memory_mb":    memoryMB,
					"disk_size_gb": diskGB,
					"image":        image,
					"cpu_shares":   1,
					"tags":         map[string]string{"purpose": "loadtest"},
				})
				start := time.Now()
				id, placedOn, err := doCreate(client, apiURL, token, body)
				report.Create.add(Result{Latency: time.Since(start), Err: err})
				if err == nil {
					mu.Lock()
					ids = append(ids, id)
					if placedOn != "" && placedOn != "local" {
						vmNode[id] = placedOn
					}
					mu.Unlock()
				}
			}
		}(w)
	}
	wg.Wait()
	return ids, vmNode
}

func doCreate(client *http.Client, apiURL, token string, body []byte) (id, placedOn string, err error) {
	req, _ := http.NewRequest(http.MethodPost, apiURL+"/api/v1/vms", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(data), 300))
	}
	var parsed struct {
		ID       string `json:"id"`
		PlacedOn string `json:"placed_on"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil || parsed.ID == "" {
		return "", "", fmt.Errorf("no id in response: %s", truncate(string(data), 300))
	}
	return parsed.ID, parsed.PlacedOn, nil
}

// runDeletePhase deletes every id against whichever node currently owns it
// (per vmNode; see runCreatePhase's doc comment) -- cluster dispatch and
// migrate both move ownership, and only the owning node's own API can act on
// a VM (delete/get are not cluster-dispatched the way create is).
func runDeletePhase(client *http.Client, apiURL, peerAPIURL, token string, ids []string, vmNode map[string]string, concurrency int, report *Report) {
	work := make(chan string, len(ids))
	for _, id := range ids {
		work <- id
	}
	close(work)

	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range work {
				base := apiURL
				if vmNode[id] != "" {
					if peerAPIURL == "" {
						report.Delete.add(Result{Err: fmt.Errorf("%s was placed on peer %q but -peer-api-url was not given, cannot delete it", id, vmNode[id])})
						continue
					}
					base = peerAPIURL
				}
				start := time.Now()
				err := doDelete(client, base, token, id)
				report.Delete.add(Result{Latency: time.Since(start), Err: err})
			}
		}()
	}
	wg.Wait()
}

func doDelete(client *http.Client, apiURL, token, id string) error {
	req, _ := http.NewRequest(http.MethodDelete, apiURL+"/api/v1/vms/"+id, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(data), 300))
	}
	return nil
}

// runMigratePhase starts each of the given VMs, waits for it to report
// running, then issues a real migrate call to migrateTarget. Sequential (not
// fanned out with the create/delete concurrency) since each migration is
// itself a heavyweight real-qemu operation on the target. Updates vmNode on a
// successful migrate so the later delete phase routes to the new owner.
func runMigratePhase(client *http.Client, apiURL, token string, ids []string, migrateTarget string, report *Report, vmNode map[string]string) {
	for _, id := range ids {
		if err := doStart(client, apiURL, token, id); err != nil {
			report.Migrate.add(Result{Err: fmt.Errorf("start before migrate: %w", err)})
			continue
		}
		if err := waitRunning(client, apiURL, token, id, 30*time.Second); err != nil {
			report.Migrate.add(Result{Err: fmt.Errorf("wait running before migrate: %w", err)})
			continue
		}
		start := time.Now()
		err := doMigrate(client, apiURL, token, id, migrateTarget)
		report.Migrate.add(Result{Latency: time.Since(start), Err: err})
		if err == nil {
			vmNode[id] = migrateTarget
		}
	}
}

func doStart(client *http.Client, apiURL, token, id string) error {
	req, _ := http.NewRequest(http.MethodPost, apiURL+"/api/v1/vms/"+id+"/start", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(data), 300))
	}
	return nil
}

func waitRunning(client *http.Client, apiURL, token, id string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequest(http.MethodGet, apiURL+"/api/v1/vms/"+id, nil)
		req.Header.Set("Authorization", "Bearer "+token)
		resp, err := client.Do(req)
		if err == nil {
			data, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			var parsed struct {
				State string `json:"state"`
			}
			if json.Unmarshal(data, &parsed) == nil && strings.EqualFold(parsed.State, "running") {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %s to report running", id)
}

func doMigrate(client *http.Client, apiURL, token, id, targetNode string) error {
	body, _ := json.Marshal(map[string]string{"target_node": targetNode})
	req, _ := http.NewRequest(http.MethodPost, apiURL+"/api/v1/vms/"+id+"/migrate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(data), 300))
	}
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
