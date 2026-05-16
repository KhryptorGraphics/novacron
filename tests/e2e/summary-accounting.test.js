const { summarizePlaywrightReport } = require('./summary-accounting');

describe('Playwright summary accounting', () => {
  it('counts actual Playwright test outcomes instead of deriving passed from stats', () => {
    const report = {
      stats: {
        expected: 4,
        unexpected: 1,
        flaky: 1,
        skipped: 1,
        duration: 12345,
      },
      suites: [
        {
          specs: [
            { tests: [{ status: 'expected', results: [{ status: 'passed' }] }] },
            { tests: [{ status: 'unexpected', results: [{ status: 'failed' }] }] },
            { tests: [{ status: 'flaky', results: [{ status: 'failed' }, { status: 'passed' }] }] },
            { tests: [{ status: 'skipped', results: [{ status: 'skipped' }] }] },
          ],
        },
      ],
    };

    expect(summarizePlaywrightReport(report)).toEqual({
      totalTests: 4,
      passed: 1,
      failed: 1,
      skipped: 1,
      flaky: 1,
      duration: 12345,
    });
  });

  it('falls back to Playwright stats when no suite tree is present', () => {
    expect(
      summarizePlaywrightReport({
        stats: {
          expected: 2,
          unexpected: 1,
          flaky: 1,
          skipped: 3,
          duration: 500,
        },
      })
    ).toEqual({
      totalTests: 7,
      passed: 2,
      failed: 1,
      skipped: 3,
      flaky: 1,
      duration: 500,
    });
  });
});
