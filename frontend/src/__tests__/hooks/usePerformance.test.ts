import { renderHook } from '@testing-library/react';
import { usePerformance } from '@/hooks/usePerformance';

// jsdom does not implement PerformanceObserver; provide a no-op mock so the
// hook's effects run without throwing.
beforeAll(() => {
  class MockPerformanceObserver {
    constructor(_cb: PerformanceObserverCallback) {}
    observe() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  (global as unknown as { PerformanceObserver: unknown }).PerformanceObserver =
    MockPerformanceObserver;

  // jsdom's performance object lacks getEntriesByType; stub it to an empty list.
  if (typeof performance.getEntriesByType !== 'function') {
    (performance as unknown as { getEntriesByType: unknown }).getEntriesByType = () => [];
  }
});

describe('usePerformance (Core Web Vitals)', () => {
  it('exposes the metrics/isSupported/reportMetrics/grade surface', () => {
    const { result } = renderHook(() => usePerformance());

    expect(result.current).toEqual(
      expect.objectContaining({
        metrics: expect.any(Object),
        isSupported: expect.any(Boolean),
        reportMetrics: expect.any(Function),
        grade: expect.any(String),
      })
    );
  });

  it('reports the five Core Web Vitals metric keys', () => {
    const { result } = renderHook(() => usePerformance());

    (['lcp', 'fid', 'cls', 'fcp', 'ttfb'] as const).forEach((key) =>
      expect(result.current.metrics).toHaveProperty(key)
    );
  });

  it('grades as N/A until metrics arrive', () => {
    const { result } = renderHook(() => usePerformance());

    expect(result.current.grade).toBe('N/A');
  });

  it('reportMetrics is a safe no-op without an analytics endpoint', () => {
    const { result } = renderHook(() => usePerformance());

    expect(() => result.current.reportMetrics()).not.toThrow();
  });
});
