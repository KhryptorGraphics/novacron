import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const { summarizePlaywrightReport } = require('./summary-accounting');

/**
 * Global Teardown for Playwright Tests
 *
 * Runs once after all tests to:
 * - Clean up test data
 * - Remove temporary files
 * - Stop external services
 * - Generate final reports
 * - Archive test artifacts
 */
async function globalTeardown(config: FullConfig): Promise<void> {
  console.log('🧹 Starting Playwright Global Teardown...');

  // Clean up temporary files
  await cleanupTemporaryFiles();

  // Clean up test data
  await cleanupTestData();

  // Archive test artifacts
  await archiveArtifacts();

  // Generate summary report
  await generateSummaryReport(config);

  // Clean up old artifacts based on retention policy
  await cleanupOldArtifacts();

  console.log('✅ Global Teardown Complete');
}

/**
 * Clean up temporary files created during tests
 */
async function cleanupTemporaryFiles(): Promise<void> {
  console.log('🗑️  Cleaning up temporary files...');

  const tempFiles = [
    'tests/e2e/fixtures/auth-state.json',
  ];

  for (const file of tempFiles) {
    const filePath = path.resolve(__dirname, `../../${file}`);
    try {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`✅ Removed: ${file}`);
      }
    } catch (error) {
      console.warn(`⚠️  Failed to remove ${file}:`, error);
    }
  }
}

/**
 * Clean up test data from database or external services
 */
async function cleanupTestData(): Promise<void> {
  console.log('🗑️  Cleaning up test data...');

  // Add cleanup logic for test data
  // Example: Delete test users, VMs, etc.
  try {
    // This is a placeholder - implement actual cleanup based on your needs
    console.log('✅ Test data cleanup complete');
  } catch (error) {
    console.warn('⚠️  Test data cleanup failed:', error);
  }
}

/**
 * Archive test artifacts for long-term storage
 */
async function archiveArtifacts(): Promise<void> {
  console.log('📦 Archiving test artifacts...');

  // Only archive on CI or if explicitly requested
  if (!process.env.CI && process.env.ARCHIVE_ARTIFACTS !== 'true') {
    console.log('⏭️  Skipping artifact archiving (not on CI)');
    return;
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const archiveDir = path.resolve(__dirname, `../../tests/e2e/archives/${timestamp}`);

  try {
    // Create archive directory
    if (!fs.existsSync(archiveDir)) {
      fs.mkdirSync(archiveDir, { recursive: true });
    }

    // Copy important artifacts
    const artifactDirs = [
      'tests/e2e/reports/html',
      'tests/e2e/reports/json',
      'tests/e2e/reports/junit',
    ];

    for (const dir of artifactDirs) {
      const sourcePath = path.resolve(__dirname, `../../${dir}`);
      const targetPath = path.join(archiveDir, path.basename(dir));

      if (fs.existsSync(sourcePath)) {
        copyRecursive(sourcePath, targetPath);
        console.log(`✅ Archived: ${dir} -> ${targetPath}`);
      }
    }

    console.log(`✅ Artifacts archived to: ${archiveDir}`);
  } catch (error) {
    console.warn('⚠️  Artifact archiving failed:', error);
  }
}

/**
 * Generate summary report of test execution
 */
async function generateSummaryReport(config: FullConfig): Promise<void> {
  console.log('📊 Generating summary report...');

  const reportPath = path.resolve(__dirname, 'reports/json/results.json');

  if (!fs.existsSync(reportPath)) {
    console.log('⏭️  No JSON report found, skipping summary');
    return;
  }

  try {
    const reportData = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    const aggregate = summarizePlaywrightReport(reportData);

    const summary = {
      timestamp: new Date().toISOString(),
      environment: process.env.TEST_ENV || 'local',
      totalTests: aggregate.totalTests,
      passed: aggregate.passed,
      failed: aggregate.failed,
      skipped: aggregate.skipped,
      flaky: aggregate.flaky,
      duration: aggregate.duration,
      projects: config.projects.map(p => p.name),
    };

    const summaryPath = path.resolve(__dirname, 'reports/summary.json');
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

    // Log summary to console
    console.log('\n📈 Test Execution Summary:');
    console.log(`   Total Tests: ${summary.totalTests}`);
    console.log(`   ✅ Passed: ${summary.passed}`);
    console.log(`   ❌ Failed: ${summary.failed}`);
    console.log(`   ⏭️  Skipped: ${summary.skipped}`);
    console.log(`   🔄 Flaky: ${summary.flaky}`);
    console.log(`   ⏱️  Duration: ${(summary.duration / 1000).toFixed(2)}s\n`);

    console.log(`✅ Summary saved to: ${summaryPath}`);
  } catch (error) {
    console.warn('⚠️  Failed to generate summary report:', error);
  }
}

/**
 * Clean up old artifacts based on retention policy
 */
async function cleanupOldArtifacts(): Promise<void> {
  console.log('🗑️  Cleaning up old artifacts...');

  const retentionDays = parseInt(process.env.ARTIFACT_RETENTION_DAYS || '7', 10);
  const archivesDir = path.resolve(__dirname, '../../tests/e2e/archives');

  if (!fs.existsSync(archivesDir)) {
    return;
  }

  try {
    const now = Date.now();
    const maxAge = retentionDays * 24 * 60 * 60 * 1000;

    const archives = fs.readdirSync(archivesDir);

    for (const archive of archives) {
      const archivePath = path.join(archivesDir, archive);
      const stats = fs.statSync(archivePath);

      if (now - stats.mtime.getTime() > maxAge) {
        fs.rmSync(archivePath, { recursive: true, force: true });
        console.log(`✅ Removed old archive: ${archive}`);
      }
    }
  } catch (error) {
    console.warn('⚠️  Failed to clean up old artifacts:', error);
  }
}

/**
 * Helper function to copy directory recursively
 */
function copyRecursive(source: string, target: string): void {
  if (!fs.existsSync(target)) {
    fs.mkdirSync(target, { recursive: true });
  }

  const files = fs.readdirSync(source);

  for (const file of files) {
    const sourcePath = path.join(source, file);
    const targetPath = path.join(target, file);
    const stats = fs.statSync(sourcePath);

    if (stats.isDirectory()) {
      copyRecursive(sourcePath, targetPath);
    } else {
      fs.copyFileSync(sourcePath, targetPath);
    }
  }
}

export default globalTeardown;
