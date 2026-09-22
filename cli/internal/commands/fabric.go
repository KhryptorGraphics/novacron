package commands

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
	"github.com/novacron/cli/pkg/output"
	"github.com/novacron/cli/pkg/service"
	"github.com/spf13/cobra"
)

// Fabric request budgets. Submitting a job waits for the executor VM to be
// created and started (on a peer when placement picks one), so it gets a longer
// budget than the read paths.
const (
	fabricReadTimeout = 30 * time.Second
	fabricJobTimeout  = 2 * time.Minute
)

// newFabricService builds the fabric service for the current cluster. It is a
// package variable so tests can point the commands at a test server.
var newFabricService = fabricServiceFromConfig

// fabricServiceFromConfig resolves the current cluster from the CLI config and
// the token stored for it, then builds the fabric API client.
func fabricServiceFromConfig() (*service.FabricService, error) {
	cfg, err := config.NewManager("")
	if err != nil {
		return nil, err
	}

	cluster, err := cfg.GetCurrentCluster()
	if err != nil {
		return nil, fmt.Errorf("%w: set currentCluster in ~/.novacron/config.yaml (see `novacron config`)", err)
	}

	token, err := storedClusterToken(cluster.Name)
	if err != nil {
		return nil, err
	}

	client, err := api.NewClient(cluster.Server, api.WithInsecure(cluster.Insecure))
	if err != nil {
		return nil, err
	}
	client.SetToken(token)

	return service.NewFabricService(client), nil
}

// storedClusterToken loads the token saved for the cluster. It is set on the
// client directly rather than through auth.TokenAuth.Apply: the stored token is
// sent as-is, because TokenAuth re-checks expiry locally and its refresh path is
// not implemented.
func storedClusterToken(clusterName string) (string, error) {
	store, err := auth.NewTokenStore()
	if err != nil {
		return "", err
	}

	stored, err := store.Load(clusterName)
	if err != nil || stored == nil || stored.Token == "" {
		return "", fmt.Errorf("no stored token for cluster %q: run `novacron login` or add one to ~/.novacron/tokens.json", clusterName)
	}

	return stored.Token, nil
}

// NewFabricCommand creates the fabric command group
func NewFabricCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "fabric",
		Short: "Manage the bandwidth-aware compute fabric",
		Long: `Commands for the peer-to-peer compute fabric.

Jobs are placed on the node the bandwidth and locality cost picks, run there in
a Process VM, and their output is read back from that VM. Bulk data moves
between nodes as transfers with measured throughput.`,
	}

	cmd.AddCommand(
		newFabricNodesCommand(),
		newFabricJobsCommand(),
		newFabricJobCommand(),
		newFabricTransfersCommand(),
		newFabricTransferCommand(),
		newFabricUsageCommand(),
	)

	return cmd
}

// newFabricNodesCommand creates the fabric nodes command
func newFabricNodesCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "nodes",
		Short: "List fabric nodes with capacity and link state",
		Long:  "List every node in the fabric with its live capacity and measured link to the local node",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			nodes, err := svc.ListNodes(ctx)
			if err != nil {
				return err
			}

			return printFabricPayload(cmd, nodes, func(w io.Writer) error {
				return printFabricNodes(w, nodes)
			})
		},
	}
}

// newFabricJobsCommand creates the fabric jobs list command
func newFabricJobsCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "jobs",
		Short: "List fabric jobs",
		Long:  "List every job submitted to the fabric and the node each one runs on",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			jobs, err := svc.ListJobs(ctx)
			if err != nil {
				return err
			}

			return printFabricPayload(cmd, jobs, func(w io.Writer) error {
				return printFabricJobs(w, jobs)
			})
		},
	}
}

// newFabricJobCommand creates the fabric job command group
func newFabricJobCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "job",
		Short: "Submit and manage fabric jobs",
	}

	cmd.AddCommand(
		newFabricJobSubmitCommand(),
		newFabricJobStatusCommand(),
		newFabricJobCancelCommand(),
	)

	return cmd
}

// fabricJobFlags is the raw flag input for a job submission.
type fabricJobFlags struct {
	Name        string
	Command     string
	Args        []string
	Env         []string
	NodeID      string
	MemoryMB    int
	VCPUs       int
	BytesToMove int
}

// newFabricJobSubmitCommand creates the fabric job submit command
func newFabricJobSubmitCommand() *cobra.Command {
	var flags fabricJobFlags

	cmd := &cobra.Command{
		Use:   "submit",
		Short: "Submit a job to the fabric",
		Long: `Submit a job for placement and execution on a fabric node.

The job is placed on the node the bandwidth and locality cost picks; --node pins
it to a specific node instead, and --bytes-to-move tells the placement how much
input data would have to be moved to each candidate node.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			spec, err := buildFabricJobSpec(flags)
			if err != nil {
				return err
			}

			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricJobTimeout)
			defer cancel()

			submission, err := svc.SubmitJob(ctx, spec)
			if err != nil {
				return err
			}

			return printFabricJobSubmission(cmd.OutOrStdout(), submission)
		},
	}

	cmd.Flags().StringVar(&flags.Name, "name", "", "job name")
	cmd.Flags().StringVar(&flags.Command, "command", "", "command to run (required)")
	cmd.Flags().StringArrayVar(&flags.Args, "arg", nil, "argument to the command (repeatable)")
	cmd.Flags().StringArrayVar(&flags.Env, "env", nil, "environment variable as KEY=VALUE (repeatable)")
	cmd.Flags().StringVar(&flags.NodeID, "node", "", "pin the job to a node instead of letting placement choose")
	cmd.Flags().IntVar(&flags.MemoryMB, "memory-mb", 0, "memory to reserve for the job, in MB")
	cmd.Flags().IntVar(&flags.VCPUs, "vcpus", 0, "virtual CPUs to reserve for the job")
	cmd.Flags().IntVar(&flags.BytesToMove, "bytes-to-move", 0, "bytes of input the job needs moved to its node")
	_ = cmd.MarkFlagRequired("command")

	return cmd
}

// newFabricJobStatusCommand creates the fabric job status command
func newFabricJobStatusCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "status <job-id>",
		Short: "Show a job's status and log tails",
		Long:  "Show one job's status, executor node and the tail of its stdout/stderr",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			job, err := svc.GetJob(ctx, args[0])
			if err != nil {
				return err
			}

			return printFabricJobDetail(cmd.OutOrStdout(), job)
		},
	}
}

// newFabricJobCancelCommand creates the fabric job cancel command
func newFabricJobCancelCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "cancel <job-id>",
		Short: "Cancel a job",
		Long:  "Cancel a job by stopping its executor VM",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricJobTimeout)
			defer cancel()

			result, err := svc.CancelJob(ctx, args[0])
			if err != nil {
				return err
			}

			if !result.Cancelled {
				reason := result.Error
				if reason == "" {
					reason = result.Status
				}
				return fmt.Errorf("job %s was not cancelled: %s", args[0], reason)
			}

			fmt.Fprintf(cmd.OutOrStdout(), "Job %s cancelled (status %s)\n", args[0], result.Status)
			return nil
		},
	}
}

// newFabricTransfersCommand creates the fabric transfers list command
func newFabricTransfersCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "transfers",
		Short: "List bulk data transfers between nodes",
		Long:  "List the fabric's bulk data transfers and their measured progress",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			transfers, err := svc.ListTransfers(ctx)
			if err != nil {
				return err
			}

			return printFabricPayload(cmd, transfers, func(w io.Writer) error {
				return printFabricTransfers(w, transfers)
			})
		},
	}
}

// newFabricTransferCommand creates the fabric transfer detail command
func newFabricTransferCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "transfer <transfer-id>",
		Short: "Show one transfer's progress",
		Long:  "Show one transfer's progress, compression and the inputs its scheduling decision used",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			transfer, err := svc.GetTransfer(ctx, args[0])
			if err != nil {
				return err
			}

			return printFabricTransfer(cmd.OutOrStdout(), transfer)
		},
	}
}

// newFabricUsageCommand creates the fabric usage summary command. It reads
// the metered consumption the fabric already measures (transfer egress, job
// seconds, migrations) for the caller's organization.
func newFabricUsageCommand() *cobra.Command {
	var orgID string
	cmd := &cobra.Command{
		Use:   "usage",
		Short: "Show measured usage and estimated cost",
		Long: `Show the measured consumption the fabric recorded for an organization.

The metering is real: egress bytes come from completed transfers, job seconds
from finished jobs, migrations from successful moves. The cost estimate uses
the operator's rate card; a zero rate means unpriced, not free.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			svc, err := newFabricService()
			if err != nil {
				return err
			}

			ctx, cancel := context.WithTimeout(context.Background(), fabricReadTimeout)
			defer cancel()

			summary, err := svc.UsageSummary(ctx, orgID)
			if err != nil {
				return err
			}

			return printFabricPayload(cmd, summary, func(w io.Writer) error {
				return printFabricUsage(w, summary)
			})
		},
	}
	cmd.Flags().StringVar(&orgID, "org", "", "organization id (admin only; default: your own org)")
	return cmd
}

// printFabricUsage writes the usage summary as labeled fields.
func printFabricUsage(w io.Writer, s *api.FabricUsageSummary) error {
	if s == nil {
		return nil
	}
	fabricField(w, "Window", s.From+" -> "+s.To)
	if s.OrgID != "" {
		fabricField(w, "Org", s.OrgID)
	}
	fabricField(w, "Egress", fmt.Sprintf("%.2f GiB (%.0f bytes)", s.Totals.EgressGB, s.Totals.EgressBytes))
	fabricField(w, "Migrations", fmt.Sprintf("%.0f", s.Totals.Migrations))
	fabricField(w, "Job time", fmt.Sprintf("%.1fs", s.Totals.JobSeconds))
	fabricField(w, "vCPU time", fmt.Sprintf("%.2f hours", s.Totals.VCPUHours))
	fabricField(w, "Est. cost", fmt.Sprintf("$%.4f", s.Totals.EstimatedCost))
	fabricField(w, "Rates", fmt.Sprintf("egress $%.3f/GiB, vCPU $%.3f/h, job $%.4f/s, migration $%.2f",
		s.RateCard.PerGBEgress, s.RateCard.PerVCPUHour, s.RateCard.PerJobSecond, s.RateCard.PerMigration))
	if s.Note != "" {
		fmt.Fprintln(w)
		fmt.Fprintln(w, s.Note)
	}
	return nil
}

// buildFabricJobSpec validates the submit flags and turns them into the API
// payload.
func buildFabricJobSpec(flags fabricJobFlags) (api.FabricJobSpec, error) {
	command := strings.TrimSpace(flags.Command)
	if command == "" {
		return api.FabricJobSpec{}, fmt.Errorf("--command is required")
	}

	env, err := parseEnvVars(flags.Env)
	if err != nil {
		return api.FabricJobSpec{}, err
	}

	return api.FabricJobSpec{
		Name:        flags.Name,
		Command:     command,
		Args:        flags.Args,
		Env:         env,
		NodeID:      flags.NodeID,
		MemoryMB:    flags.MemoryMB,
		VCPUs:       flags.VCPUs,
		BytesToMove: flags.BytesToMove,
	}, nil
}

// parseEnvVars parses repeated KEY=VALUE flags. An empty value is allowed; a
// missing '=' or an empty key is not.
func parseEnvVars(pairs []string) (map[string]string, error) {
	if len(pairs) == 0 {
		return nil, nil
	}

	env := make(map[string]string, len(pairs))
	for _, pair := range pairs {
		key, value, ok := strings.Cut(pair, "=")
		key = strings.TrimSpace(key)
		if !ok || key == "" {
			return nil, fmt.Errorf("invalid --env %q: expected KEY=VALUE", pair)
		}
		env[key] = value
	}

	return env, nil
}

// printFabricPayload prints a fabric payload as a table, or in the requested
// structured format (-o json|yaml).
func printFabricPayload(cmd *cobra.Command, payload interface{}, table func(io.Writer) error) error {
	format := output.GetFormat()
	if format == output.FormatJSON || format == output.FormatYAML {
		return output.NewPrinter(format).WithWriter(cmd.OutOrStdout()).Print(payload)
	}

	return table(cmd.OutOrStdout())
}

// printFabricNodes writes the node inventory, one row per node.
func printFabricNodes(w io.Writer, nodes []api.FabricNode) error {
	if len(nodes) == 0 {
		_, err := fmt.Fprintln(w, "No fabric nodes found")
		return err
	}

	tw := newFabricTable(w)
	fmt.Fprintln(tw, "NODE\tADDR\tARCH\tCORES\tMEM FREE\tSTORAGE FREE\tREACHABLE\tRTT")
	for _, node := range nodes {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n",
			orDash(node.NodeID),
			orDash(node.Addr),
			orDash(node.Arch),
			node.Cores,
			formatBytes(memFreeBytes(node)),
			formatBytes(gibibytes(node.StorageFreeGB)),
			yesNo(node.Reachable),
			formatNodeLink(node.Link),
		)
	}

	return tw.Flush()
}

// printFabricJobs writes the job list, one row per job.
func printFabricJobs(w io.Writer, jobs []api.FabricJob) error {
	if len(jobs) == 0 {
		_, err := fmt.Fprintln(w, "No fabric jobs found")
		return err
	}

	tw := newFabricTable(w)
	fmt.Fprintln(tw, "JOB ID\tNAME\tSTATUS\tNODE\tVM\tCOMMAND\tCREATED")
	for _, job := range jobs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			orDash(job.JobID),
			orDash(job.Name),
			orDash(job.Status),
			orDash(job.NodeID),
			orDash(job.VMID),
			orDash(job.Command),
			orDash(job.CreatedAt),
		)
	}

	return tw.Flush()
}

// printFabricJobSubmission writes the accepted job and the placement decision
// that chose its node.
func printFabricJobSubmission(w io.Writer, submission *api.FabricJobSubmission) error {
	fabricField(w, "Job ID", orDash(submission.JobID))
	fabricField(w, "Status", orDash(submission.Status))
	fabricField(w, "Node", orDash(submission.NodeID))
	fabricField(w, "VM", orDash(submission.VMID))

	if placement := submission.Placement; placement != nil {
		fabricField(w, "Placement", orDash(placement.Decision))
		if placement.Reason != "" {
			fabricField(w, "Reason", placement.Reason)
		}
		if placement.CostEstimateS != nil {
			fabricField(w, "Cost", strconv.FormatFloat(*placement.CostEstimateS, 'f', -1, 64)+"s")
		}
	}

	return nil
}

// printFabricJobDetail writes one job's fields plus the tail of its streams.
func printFabricJobDetail(w io.Writer, job *api.FabricJobDetail) error {
	fabricField(w, "Job ID", orDash(job.JobID))
	if job.Name != "" {
		fabricField(w, "Name", job.Name)
	}
	fabricField(w, "Command", orDash(job.Command))
	fabricField(w, "Status", orDash(job.Status))
	fabricField(w, "Node", orDash(job.NodeID))
	fabricField(w, "VM", orDash(job.VMID))
	fabricField(w, "Created", orDash(job.CreatedAt))
	if job.Error != "" {
		fabricField(w, "Error", job.Error)
	}

	switch {
	case job.Logs != nil:
		writeFabricLogTail(w, "stdout", job.Logs.Stdout)
		writeFabricLogTail(w, "stderr", job.Logs.Stderr)
	case job.LogsError != "":
		fmt.Fprintf(w, "\nLogs unavailable: %s\n", job.LogsError)
	default:
		fmt.Fprintln(w, "\nLogs unavailable for this job")
	}

	return nil
}

// printFabricTransfers writes the transfer list, one row per transfer.
func printFabricTransfers(w io.Writer, transfers []api.FabricTransfer) error {
	if len(transfers) == 0 {
		_, err := fmt.Fprintln(w, "No transfers found")
		return err
	}

	tw := newFabricTable(w)
	fmt.Fprintln(tw, "TRANSFER ID\tSTATUS\tMOVED\tTOTAL\tRATE\tETA\tCOMPRESSION")
	for _, transfer := range transfers {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			orDash(transfer.TransferID),
			orDash(transfer.Status),
			formatBytes(transfer.BytesMoved),
			formatBytes(transfer.BytesTotal),
			formatRate(transfer.MeasuredBps),
			formatETA(transfer.ETASeconds),
			orDash(transfer.Compression),
		)
	}

	return tw.Flush()
}

// printFabricTransfer writes one transfer's progress and the inputs its
// scheduling decision was made from.
func printFabricTransfer(w io.Writer, transfer *api.FabricTransfer) error {
	fabricField(w, "Transfer ID", orDash(transfer.TransferID))
	fabricField(w, "Status", orDash(transfer.Status))
	fabricField(w, "Moved", fmt.Sprintf("%s / %s", formatBytes(transfer.BytesMoved), formatBytes(transfer.BytesTotal)))
	fabricField(w, "Rate", formatRate(transfer.MeasuredBps))
	fabricField(w, "Compression", orDash(transfer.Compression))
	fabricField(w, "ETA", formatETA(transfer.ETASeconds))

	if len(transfer.DecisionInputs) > 0 {
		fmt.Fprintln(w, "Decision inputs:")
		for _, key := range sortedKeys(transfer.DecisionInputs) {
			fmt.Fprintf(w, "  %s: %s\n", key, formatDecisionInput(transfer.DecisionInputs[key]))
		}
	}

	return nil
}

// newFabricTable creates a column writer for fabric tables. Fields are padded so
// the columns line up without borders, matching the other list commands.
func newFabricTable(w io.Writer) *tabwriter.Writer {
	return tabwriter.NewWriter(w, 0, 8, 2, ' ', 0)
}

// fabricField writes a labeled field with the labels aligned in one column.
func fabricField(w io.Writer, label, value string) {
	fmt.Fprintf(w, "%-13s%s\n", label+":", value)
}

// writeFabricLogTail writes one captured stream. An empty stream is marked so
// "no output yet" cannot be mistaken for a missing section.
func writeFabricLogTail(w io.Writer, name, content string) {
	fmt.Fprintf(w, "\n--- %s (tail) ---\n", name)
	if strings.TrimSpace(content) == "" {
		fmt.Fprintln(w, "(empty)")
		return
	}

	fmt.Fprint(w, content)
	if !strings.HasSuffix(content, "\n") {
		fmt.Fprintln(w)
	}
}

// formatNodeLink renders a node's measured link to the local node. The local
// node and never-probed peers have no profile; a stale profile is flagged
// because placement must re-probe before trusting it.
func formatNodeLink(link *api.FabricLinkProfile) string {
	if link == nil {
		return "-"
	}
	if link.Stale {
		return fmt.Sprintf("%.2fms (stale)", link.RTTMS)
	}
	return fmt.Sprintf("%.2fms", link.RTTMS)
}

// formatBytes renders a byte count in binary units, matching the MiB/GiB the
// fabric reports capacity in.
func formatBytes(bytes int64) string {
	if bytes < 0 {
		return "-" + formatBytes(-bytes)
	}
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	}

	units := []string{"KiB", "MiB", "GiB", "TiB", "PiB", "EiB"}
	value := float64(bytes) / 1024
	unit := 0
	for value >= 1024 && unit < len(units)-1 {
		value /= 1024
		unit++
	}

	return fmt.Sprintf("%.1f %s", value, units[unit])
}

// formatRate renders a measured throughput. An unmeasured rate has no value to
// report.
func formatRate(bytesPerSecond int64) string {
	if bytesPerSecond <= 0 {
		return "-"
	}
	return formatBytes(bytesPerSecond) + "/s"
}

// formatETA renders an optional estimated time to completion in seconds.
func formatETA(seconds *float64) string {
	if seconds == nil || *seconds < 0 {
		return "-"
	}
	return time.Duration(*seconds * float64(time.Second)).Round(time.Millisecond).String()
}

// formatDecisionInput renders a decision_inputs value. JSON numbers decode to
// float64, which would otherwise print in exponent notation.
func formatDecisionInput(value interface{}) string {
	if number, ok := value.(float64); ok {
		return strconv.FormatFloat(number, 'f', -1, 64)
	}
	return fmt.Sprintf("%v", value)
}

// memFreeBytes is the memory a node can still reserve for VMs.
func memFreeBytes(node api.FabricNode) int64 {
	return (node.MemTotalMB - node.MemAllocatedMB) * 1024 * 1024
}

// gibibytes converts the fabric's GiB capacity fields to bytes.
func gibibytes(gb int64) int64 {
	return gb * 1024 * 1024 * 1024
}

func yesNo(value bool) string {
	if value {
		return "yes"
	}
	return "no"
}

func orDash(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func sortedKeys(values map[string]interface{}) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
