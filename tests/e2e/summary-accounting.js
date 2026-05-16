function collectPlaywrightTests(suites = []) {
  const tests = [];

  for (const suite of suites) {
    if (Array.isArray(suite.specs)) {
      for (const spec of suite.specs) {
        if (Array.isArray(spec.tests)) {
          tests.push(...spec.tests);
        }
      }
    }

    if (Array.isArray(suite.suites)) {
      tests.push(...collectPlaywrightTests(suite.suites));
    }
  }

  return tests;
}

function classifyPlaywrightTest(test) {
  if (test.status === 'expected') {
    return 'passed';
  }
  if (test.status === 'unexpected') {
    return 'failed';
  }
  if (test.status === 'flaky') {
    return 'flaky';
  }
  if (test.status === 'skipped') {
    return 'skipped';
  }

  const statuses = (test.results || []).map((result) => result.status);
  if (statuses.length === 0) {
    return 'passed';
  }
  if (statuses.every((status) => status === 'skipped')) {
    return 'skipped';
  }
  if (statuses.some((status) => status === 'passed') && statuses.some((status) => status !== 'passed' && status !== 'skipped')) {
    return 'flaky';
  }
  if (statuses.some((status) => ['failed', 'timedOut', 'interrupted'].includes(status))) {
    return 'failed';
  }
  if (statuses.some((status) => status === 'passed')) {
    return 'passed';
  }

  return 'failed';
}

function summarizePlaywrightReport(reportData = {}) {
  const tests = collectPlaywrightTests(reportData.suites || []);

  if (tests.length === 0) {
    const stats = reportData.stats || {};
    return {
      totalTests: Number(stats.expected || 0) + Number(stats.unexpected || 0) + Number(stats.flaky || 0) + Number(stats.skipped || 0),
      passed: Number(stats.expected || 0),
      failed: Number(stats.unexpected || 0),
      skipped: Number(stats.skipped || 0),
      flaky: Number(stats.flaky || 0),
      duration: Number(stats.duration || 0),
    };
  }

  const summary = {
    totalTests: tests.length,
    passed: 0,
    failed: 0,
    skipped: 0,
    flaky: 0,
    duration: Number(reportData.stats?.duration || 0),
  };

  for (const test of tests) {
    summary[classifyPlaywrightTest(test)] += 1;
  }

  return summary;
}

module.exports = {
  classifyPlaywrightTest,
  collectPlaywrightTests,
  summarizePlaywrightReport,
};
