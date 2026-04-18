package handlers

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const maxScannedEntries = 256

func normalizeScanTypes(scanTypes []ScanType) []ScanType {
	if len(scanTypes) == 0 {
		return []ScanType{ScanTypeSecrets, ScanTypeFilesystem}
	}

	seen := make(map[ScanType]struct{}, len(scanTypes))
	normalized := make([]ScanType, 0, len(scanTypes))
	for _, scanType := range scanTypes {
		candidate := ScanType(strings.ToLower(strings.TrimSpace(string(scanType))))
		if candidate == "" {
			continue
		}
		if _, ok := seen[candidate]; ok {
			continue
		}
		seen[candidate] = struct{}{}
		normalized = append(normalized, candidate)
	}
	if len(normalized) == 0 {
		return []ScanType{ScanTypeSecrets, ScanTypeFilesystem}
	}
	return normalized
}

func containsScanType(scanTypes []ScanType, candidate ScanType) bool {
	for _, scanType := range scanTypes {
		if scanType == candidate {
			return true
		}
	}
	return false
}

func runLocalScan(ctx context.Context, scanID string, targets []string, requestedScanTypes []ScanType, startedAt time.Time) (*ScanResults, error) {
	scanTypes := normalizeScanTypes(requestedScanTypes)
	findings := make([]SecurityFinding, 0)

	for _, target := range targets {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		targetFindings, err := scanTarget(target, scanTypes)
		if err != nil {
			findings = append(findings, SecurityFinding{
				ID:           fmt.Sprintf("%s-%d", scanID, len(findings)+1),
				Category:     "scan",
				Severity:     SeverityHigh,
				Target:       target,
				Title:        "Scan target unavailable",
				Description:  err.Error(),
				DiscoveredAt: time.Now().UTC(),
			})
			continue
		}
		findings = append(findings, targetFindings...)
	}

	completedAt := time.Now().UTC()
	return &ScanResults{
		ScanID:      scanID,
		Status:      "completed",
		Targets:     append([]string(nil), targets...),
		ScanTypes:   append([]ScanType(nil), scanTypes...),
		Findings:    findings,
		Summary:     summarizeFindings(findings),
		StartedAt:   startedAt,
		CompletedAt: completedAt,
	}, nil
}

func scanTarget(target string, scanTypes []ScanType) ([]SecurityFinding, error) {
	info, err := os.Stat(target)
	if err != nil {
		return nil, fmt.Errorf("failed to access target %q: %w", target, err)
	}

	findings := make([]SecurityFinding, 0)
	if info.Mode().Perm()&0o002 != 0 {
		findings = append(findings, newFinding("filesystem", SeverityHigh, target, "World-writable path detected", "The target path is world-writable, which increases tampering risk.", "Tighten filesystem permissions to remove world write access."))
	}

	if !info.IsDir() {
		return append(findings, scanFile(target, info, scanTypes)...), nil
	}

	entries := 0
	walkErr := filepath.WalkDir(target, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			findings = append(findings, newFinding("filesystem", SeverityMedium, path, "Unreadable path", walkErr.Error(), "Ensure the scanner can read the target path."))
			return nil
		}
		if path == target {
			return nil
		}
		entries++
		if entries > maxScannedEntries {
			return fs.SkipAll
		}

		info, err := entry.Info()
		if err != nil {
			return nil
		}
		if info.Mode().Perm()&0o002 != 0 {
			findings = append(findings, newFinding("filesystem", SeverityHigh, path, "World-writable path detected", "A descendant path is world-writable, which increases tampering risk.", "Tighten filesystem permissions to remove world write access."))
		}
		if entry.IsDir() {
			return nil
		}
		findings = append(findings, scanFile(path, info, scanTypes)...)
		return nil
	})
	if walkErr != nil && walkErr != fs.SkipAll {
		return findings, walkErr
	}

	if entries > maxScannedEntries {
		findings = append(findings, newFinding("scan", SeverityInfo, target, "Scan scope capped", fmt.Sprintf("The scan inspected only the first %d descendant entries to keep request latency bounded.", maxScannedEntries), "Run targeted scans on specific directories when you need deeper coverage."))
	}

	return findings, nil
}

func scanFile(path string, info os.FileInfo, scanTypes []ScanType) []SecurityFinding {
	findings := make([]SecurityFinding, 0)
	base := strings.ToLower(filepath.Base(path))

	if containsScanType(scanTypes, ScanTypeSecrets) {
		switch {
		case strings.Contains(base, ".env"), strings.Contains(base, "secret"), strings.Contains(base, "token"):
			findings = append(findings, newFinding("secrets", SeverityHigh, path, "Sensitive configuration file detected", "The scan found a file name that commonly stores credentials or secrets.", "Move secrets into a managed secret store or ensure the file is excluded from shared environments."))
		case strings.HasSuffix(base, ".pem"), strings.HasSuffix(base, ".key"), strings.Contains(base, "id_rsa"):
			findings = append(findings, newFinding("secrets", SeverityCritical, path, "Private key material detected", "The scan found a file name consistent with private key material.", "Restrict access to the key and rotate it if exposure is possible."))
		}
	}

	if containsScanType(scanTypes, ScanTypeFilesystem) && info.Size() == 0 {
		findings = append(findings, newFinding("filesystem", SeverityInfo, path, "Empty file detected", "The scanner found an empty file at the target path.", "Remove stale files or populate them with the intended configuration."))
	}

	if containsScanType(scanTypes, ScanTypeDependencies) {
		switch base {
		case "package-lock.json", "pnpm-lock.yaml", "go.mod", "go.sum", "requirements.txt":
			findings = append(findings, newFinding("dependencies", SeverityInfo, path, "Dependency manifest detected", "A dependency manifest was found and can be reviewed by an external vulnerability scanner.", "Run a dependency scanner against this manifest for package CVE coverage."))
		}
	}

	return findings
}

func summarizeFindings(findings []SecurityFinding) ScanSummary {
	summary := ScanSummary{Total: len(findings)}
	for _, finding := range findings {
		switch finding.Severity {
		case SeverityCritical:
			summary.Critical++
		case SeverityHigh:
			summary.High++
		case SeverityMedium:
			summary.Medium++
		case SeverityLow:
			summary.Low++
		default:
			summary.Info++
		}
	}
	return summary
}

func newFinding(category string, severity FindingSeverity, target string, title string, description string, recommendation string) SecurityFinding {
	return SecurityFinding{
		ID:             fmt.Sprintf("finding-%d", time.Now().UTC().UnixNano()),
		Category:       category,
		Severity:       severity,
		Target:         target,
		Title:          title,
		Description:    description,
		Recommendation: recommendation,
		DiscoveredAt:   time.Now().UTC(),
	}
}
