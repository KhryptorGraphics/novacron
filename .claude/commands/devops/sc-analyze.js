#!/usr/bin/env node

/**
 * /sc:analyze - Code Analysis and Quality Assessment
 *
 * Usage:
 *   /sc:analyze [target] [--focus quality|security|performance|architecture] [--depth quick|deep] [--format text|json|report]
 *
 * Behavior:
 * - Discover: Categorize source files by language/extensions
 * - Scan: Heuristic, multi-domain static analysis (no builds or runtime)
 * - Evaluate: Severity + impact
 * - Recommend: Actionable guidance
 * - Report: Structured output (text/json/markdown report)
 */

const fs = require('fs');
const path = require('path');

module.exports = {
  name: 'sc:analyze',
  description: 'Multi-domain static code analysis (quality, security, performance, architecture)',
  usage: '/sc:analyze [target] [--focus quality|security|performance|architecture] [--depth quick|deep] [--format text|json|report]',

  async execute(args) {
    const parsed = parseArgs(args);
    const targetDir = path.resolve(process.cwd(), parsed.target || '.');

    if (!fs.existsSync(targetDir) || !fs.statSync(targetDir).isDirectory()) {
      return { error: `Target directory not found: ${parsed.target}` };
    }

    const focusSet = new Set(parsed.focus ? [parsed.focus] : ['quality','security','performance','architecture']);
    const depth = parsed.depth || 'quick';
    const format = parsed.format || 'text';

    // 1) Discover
    const discovered = discoverFiles(targetDir);

    // 2) Scan per domain
    const findings = [];
    if (focusSet.has('quality')) findings.push(...scanQuality(discovered));
    if (focusSet.has('security')) findings.push(...scanSecurity(discovered));
    if (focusSet.has('performance')) findings.push(...scanPerformance(discovered));
    if (focusSet.has('architecture')) findings.push(...scanArchitecture(targetDir, discovered));

    // 3) Evaluate: prioritize by severity order
    const severityOrder = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
    findings.sort((a, b) => (severityOrder[b.severity] - severityOrder[a.severity]) || a.domain.localeCompare(b.domain));

    // 4) Aggregate metrics
    const metrics = {
      files_scanned: discovered.all.length,
      by_language: discovered.byLangCounts,
      by_domain_counts: ['quality','security','performance','architecture'].reduce((acc, d) => {
        acc[d] = findings.filter(f => f.domain === d).length; return acc;
      }, {}),
      severity_counts: ['CRITICAL','HIGH','MEDIUM','LOW'].reduce((acc, s) => {
        acc[s] = findings.filter(f => f.severity === s).length; return acc;
      }, {}),
      depth,
      focus: Array.from(focusSet),
    };

    // 5) Report
    const now = new Date();
    const stamp = now.toISOString().replace(/[:.]/g,'-');
    const reportsDir = path.join(process.cwd(), '.reports');
    try { if (!fs.existsSync(reportsDir)) fs.mkdirSync(reportsDir); } catch {}

    let reportPath = null;
    if (format === 'json') {
      reportPath = path.join(reportsDir, `sc-analysis-${stamp}.json`);
      fs.writeFileSync(reportPath, JSON.stringify({ metrics, findings }, null, 2));
      console.log(`Saved JSON report → ${path.relative(process.cwd(), reportPath)}`);
    } else if (format === 'report') {
      reportPath = path.join(reportsDir, `sc-analysis-${stamp}.md`);
      fs.writeFileSync(reportPath, renderMarkdownReport(metrics, findings));
      console.log(`Saved Markdown report → ${path.relative(process.cwd(), reportPath)}`);
    } else {
      // text
      console.log(renderTextSummary(metrics, findings));
    }

    return {
      success: true,
      depth,
      focus: Array.from(focusSet),
      format,
      files_scanned: metrics.files_scanned,
      severity_counts: metrics.severity_counts,
      domain_counts: metrics.by_domain_counts,
      report: reportPath ? path.relative(process.cwd(), reportPath) : null,
    };
  }
};

// -----------------
// Argument Parsing
// -----------------
function parseArgs(args) {
  const out = { target: null, focus: null, depth: null, format: null };
  const it = [...args];
  if (it[0] && !String(it[0]).startsWith('--')) { out.target = it.shift(); }
  while (it.length) {
    const k = it.shift();
    if (k === '--focus') out.focus = (it.shift() || '').toLowerCase();
    else if (k === '--depth') out.depth = (it.shift() || '').toLowerCase();
    else if (k === '--format') out.format = (it.shift() || '').toLowerCase();
    else if (k.startsWith('--focus=')) out.focus = k.split('=')[1].toLowerCase();
    else if (k.startsWith('--depth=')) out.depth = k.split('=')[1].toLowerCase();
    else if (k.startsWith('--format=')) out.format = k.split('=')[1].toLowerCase();
  }
  return out;
}

// -----------------
// Discovery
// -----------------
function discoverFiles(root) {
  const ignoreDirs = new Set(['.git','node_modules','vendor','dist','build','.next','.cache','.venv','coverage','.reports']);
  const all = [];
  (function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      if (e.name.startsWith('.DS_')) continue;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) {
        if (ignoreDirs.has(e.name)) continue;
        walk(p);
      } else if (e.isFile()) {
        all.push(p);
      }
    }
  })(root);

  const byLang = new Map();
  const langOf = (f) => {
    const ext = path.extname(f).toLowerCase();
    if (ext === '.go') return 'go';
    if (ext === '.ts' || ext === '.tsx') return 'ts';
    if (ext === '.js' || ext === '.jsx' || ext === '.mjs') return 'js';
    if (ext === '.sh') return 'sh';
    if (ext === '.yml' || ext === '.yaml') return 'yaml';
    if (ext === '.dockerfile' || path.basename(f).toLowerCase() === 'dockerfile') return 'docker';
    if (ext === '.md') return 'md';
    if (ext === '.json') return 'json';
    return 'other';
  };
  for (const f of all) {
    const lang = langOf(f);
    if (!byLang.has(lang)) byLang.set(lang, []);
    byLang.get(lang).push(f);
  }
  const byLangCounts = Object.fromEntries([...byLang.entries()].map(([k,v]) => [k, v.length]));
  return { all, byLang, byLangCounts };
}

// -----------------
// Scanners
// -----------------
function scanQuality(discovered) {
  const findings = [];
  // Large files and TODO/FIXME
  for (const f of discovered.all) {
    if (!isTextFile(f)) continue;
    const content = safeRead(f);
    const lines = content.split(/\r?\n/);
    if (lines.length > 1000 && !isGeneratedFile(f, content)) {
      findings.push(makeFinding('quality', 'MEDIUM', f, 1,
        `Large file (${lines.length} LOC)`,
        'Consider refactoring into smaller modules for maintainability'));
    }
    const todoCount = (content.match(/\b(TODO|FIXME|XXX)\b/g) || []).length;
    if (todoCount >= 10) {
      findings.push(makeFinding('quality', 'LOW', f, 1,
        `High number of TODO/FIXME markers (${todoCount})`,
        'Address outstanding TODOs or track them as issues'));
    }
  }

  // Go-specific: panic, fmt prints outside CLI, missing tests in package dirs
  const goFiles = discovered.byLang.get('go') || [];
  for (const f of goFiles) {
    const content = safeRead(f);
    const isTest = f.endsWith('_test.go');
    if (!isTest && /\bpanic\s*\(/.test(content)) {
      findings.push(makeFinding('quality', 'MEDIUM', f, 1,
        'Use of panic() in non-test code',
        'Prefer explicit error returns; reserve panic for truly unrecoverable states'));
    }
    if (!isTest && /\bfmt\.(Print|Printf|Println)\b/.test(content) && !f.includes(path.join('cli', 'cmd'))) {
      findings.push(makeFinding('quality', 'LOW', f, 1,
        'Direct fmt.Print* logging in library code',
        'Use structured logger (e.g., logrus/zap) to enable levels and fields'));
    }
    if (!isTest && /\bcontext\.Background\(\)/.test(content)) {
      findings.push(makeFinding('quality', 'LOW', f, 1,
        'context.Background() used in library code',
        'Thread context via parameters to enable cancellation and deadlines'));
    }
  }
  // Go package dirs with code but no *_test.go
  const byDir = new Map();
  for (const f of goFiles) {
    const dir = path.dirname(f);
    if (!byDir.has(dir)) byDir.set(dir, []);
    byDir.get(dir).push(f);
  }
  for (const [dir, files] of byDir.entries()) {
    const hasTests = files.some(f => f.endsWith('_test.go'));
    const hasCode = files.some(f => !f.endsWith('_test.go'));
    if (hasCode && !hasTests) {
      findings.push(makeFinding('quality', 'LOW', dir, 1,
        'Go package without tests',
        'Add unit tests (filename *_test.go) for critical paths'));
    }
  }
  return findings;
}

function scanSecurity(discovered) {
  const findings = [];
  for (const f of discovered.all) {
    if (!isTextFile(f)) continue;
    const content = safeRead(f);
    // Generic patterns
    if (/http:\/\//.test(content) && !/localhost|127\.0\.0\.1/.test(content)) {
      findings.push(makeFinding('security', 'LOW', f, 1,
        'Plain HTTP URL detected',
        'Use HTTPS for external endpoints'));
    }
    if (/insecureSkipVerify\s*:\s*true/.test(content)) {
      findings.push(makeFinding('security', 'HIGH', f, 1,
        'TLS verification disabled',
        'Avoid InsecureSkipVerify=true except in isolated tests'));
    }
    if (/exec\.Command\s*\(/.test(content) || /os\/exec"/.test(content)) {
      findings.push(makeFinding('security', 'MEDIUM', f, 1,
        'Use of os/exec (command execution)',
        'Validate and sanitize inputs; minimize shell usage; prefer direct APIs'));
    }
    if (/\b(md5|sha1)\.New\s*\(/.test(content)) {
      findings.push(makeFinding('security', 'HIGH', f, 1,
        'Weak hash function (MD5/SHA1)',
        'Use SHA-256 or better; for passwords use bcrypt/argon2'));
    }
    if (/AWS[_-]?SECRET|PASSWORD=|SECRET=|TOKEN=/.test(content) && !/example|sample/i.test(content)) {
      findings.push(makeFinding('security', 'MEDIUM', f, 1,
        'Potential hardcoded secret',
        'Move secrets to environment variables or secret manager (Vault)'));
    }
    if (/SELECT\s+[^;]+\+/.test(content)) {
      findings.push(makeFinding('security', 'MEDIUM', f, 1,
        'String-concatenated SQL query',
        'Use parameterized queries / prepared statements'));
    }
  }
  // Dockerfile root user
  const dockerFiles = (discovered.byLang.get('docker') || []).concat(discovered.all.filter(f => path.basename(f).toLowerCase() === 'dockerfile'));
  for (const f of dockerFiles) {
    const c = safeRead(f);
    if (/^USER\s+root/m.test(c) || !/^USER\s+/m.test(c)) {
      findings.push(makeFinding('security', 'LOW', f, 1,
        'Dockerfile runs as root or no USER specified',
        'Specify a non-root USER for better container security'));
    }
  }
  return findings;
}

function scanPerformance(discovered) {
  const findings = [];
  const goFiles = discovered.byLang.get('go') || [];
  for (const f of goFiles) {
    const c = safeRead(f);
    if (/ioutil\.ReadAll\s*\(/.test(c) || /io\.ReadAll\s*\(/.test(c)) {
      findings.push(makeFinding('performance', 'LOW', f, 1,
        'ReadAll() may load entire stream into memory',
        'Stream or bound reads; ensure inputs are size-limited'));
    }
    if (/\btime\.Sleep\s*\(/.test(c) && !f.endsWith('_test.go')) {
      findings.push(makeFinding('performance', 'LOW', f, 1,
        'time.Sleep in production code',
        'Avoid sleeps for coordination; use timeouts, contexts, or channels'));
    }
    if (/\bgo\s+func\s*\(/.test(c) && !/context\.Context/.test(c)) {
      findings.push(makeFinding('performance', 'LOW', f, 1,
        'Goroutine launched without context',
        'Propagate context to enable cancellation and avoid leaks'));
    }
    if (/fmt\.Sprintf\([^)]*\)\s*\+/.test(c)) {
      findings.push(makeFinding('performance', 'LOW', f, 1,
        'String concat around fmt.Sprintf()',
        'Prefer a single formatting call or strings.Builder'));
    }
  }
  return findings;
}

function scanArchitecture(root, discovered) {
  const findings = [];
  // Multiple main packages (may be OK for CLI + server but flag for review)
  const goFiles = discovered.byLang.get('go') || [];
  const mainFiles = goFiles.filter(f => /package\s+main\b/.test(safeRead(f)));
  if (mainFiles.length > 2) {
    findings.push(makeFinding('architecture', 'LOW', root, 1,
      `Multiple entry points detected (${mainFiles.length} main packages)`,
      'Validate intentional: separate binaries vs accidental duplication'));
  }
  // Check for missing go.mod (monorepo with multiple modules is fine). If any Go file and no go.mod at or above root
  const hasGo = goFiles.length > 0;
  const hasGoMod = fs.existsSync(path.join(root, 'go.mod')) || fs.existsSync(path.join(root, 'backend', 'go.mod')) || fs.existsSync(path.join(root, 'backend', 'core', 'go.mod'));
  if (hasGo && !hasGoMod) {
    findings.push(makeFinding('architecture', 'MEDIUM', root, 1,
      'Go files present but no go.mod found',
      'Initialize module(s) with go mod init and proper module boundaries'));
  }
  // Large directories (possible god-packages)
  const byDirCounts = dirCounts(goFiles);
  for (const [dir, count] of byDirCounts) {
    if (count >= 40) {
      findings.push(makeFinding('architecture', 'LOW', dir, 1,
        `Large Go package (${count} files)`,
        'Consider splitting into cohesive subpackages'));
    }
  }
  return findings;
}

// -----------------
// Helpers
// -----------------
function makeFinding(domain, severity, file, line, description, recommendation) {
  return { id: `${domain}-${hash(`${file}:${line}:${description}`)}`,
           domain, severity, file: normalizePath(file), line, description, recommendation };
}

function isTextFile(f) {
  const ext = path.extname(f).toLowerCase();
  const base = path.basename(f).toLowerCase();
  const textExts = new Set(['.go','.ts','.tsx','.js','.jsx','.mjs','.json','.yml','.yaml','.md','.txt','.sh','.dockerfile']);
  if (base === 'dockerfile') return true;
  return textExts.has(ext);
}

function isGeneratedFile(f, content) {
  if (/Code generated by|DO NOT EDIT/i.test(content)) return true;
  const base = path.basename(f).toLowerCase();
  if (base.endsWith('.pb.go') || base.endsWith('_mock.go')) return true;
  return false;
}

function safeRead(f) {
  try { return fs.readFileSync(f, 'utf8'); } catch { return ''; }
}

function dirCounts(files) {
  const m = new Map();
  for (const f of files) {
    const d = path.dirname(f);
    m.set(d, (m.get(d) || 0) + 1);
  }
  return m;
}

function normalizePath(p) {
  try { return path.relative(process.cwd(), p); } catch { return p; }
}

function hash(s) {
  let h = 0; for (let i=0;i<s.length;i++) { h = ((h<<5)-h) + s.charCodeAt(i); h |= 0; }
  return (h>>>0).toString(36);
}

function renderTextSummary(metrics, findings) {
  let out = '';
  out += `Scanned files: ${metrics.files_scanned}\n`;
  out += `Depth: ${metrics.depth} | Focus: ${metrics.focus.join(', ')}\n`;
  out += `By language: ${JSON.stringify(metrics.by_language)}\n`;
  out += `Severity counts: ${JSON.stringify(metrics.severity_counts)}\n`;
  out += `Domain counts: ${JSON.stringify(metrics.by_domain_counts)}\n`;
  out += `\nTop findings:\n`;
  const top = findings.slice(0, 20);
  for (const f of top) {
    out += `- [${f.severity}] (${f.domain}) ${f.file}:${f.line} — ${f.description}\n  ↳ ${f.recommendation}\n`;
  }
  if (findings.length > top.length) out += `...and ${findings.length - top.length} more\n`;
  return out;
}

function renderMarkdownReport(metrics, findings) {
  const lines = [];
  lines.push(`# NovaCron Static Code Analysis Report`);
  lines.push('');
  lines.push(`- Generated: ${new Date().toISOString()}`);
  lines.push(`- Depth: ${metrics.depth}`);
  lines.push(`- Focus: ${metrics.focus.join(', ')}`);
  lines.push('');
  lines.push(`## Overview`);
  lines.push('');
  lines.push(`- Files scanned: ${metrics.files_scanned}`);
  lines.push(`- By language: ${code(JSON.stringify(metrics.by_language))}`);
  lines.push(`- Severity counts: ${code(JSON.stringify(metrics.severity_counts))}`);
  lines.push(`- Domain counts: ${code(JSON.stringify(metrics.by_domain_counts))}`);
  lines.push('');
  const domains = ['security','quality','performance','architecture'];
  for (const d of domains) {
    const list = findings.filter(f => f.domain === d);
    lines.push(`## ${capitalize(d)} Findings (${list.length})`);
    lines.push('');
    for (const f of list) {
      lines.push(`- [${f.severity}] ${f.file}:${f.line} — ${escapeMd(f.description)}`);
      lines.push(`  - Recommendation: ${escapeMd(f.recommendation)}`);
    }
    lines.push('');
  }
  lines.push('---');
  lines.push('_Generated by /sc:analyze_');
  return lines.join('\n');
}

function code(s) { return '`' + s.replace(/`/g,'\`') + '`'; }
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function escapeMd(s) { return s.replace(/[<>_*`]/g, m => ({'<':'&lt;','>':'&gt;','_':'\\_','*':'\\*','`':'\\`'}[m])); }

