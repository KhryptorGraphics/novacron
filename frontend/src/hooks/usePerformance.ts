import { useEffect, useState, useCallback } from 'react';

// Performance monitoring hook for Core Web Vitals
export function usePerformance() {
  const [metrics, setMetrics] = useState({
    lcp: null as number | null,
    fid: null as number | null,
    cls: null as number | null,
    fcp: null as number | null,
    ttfb: null as number | null,
  });

  const [isSupported, setIsSupported] = useState(false);

  useEffect(() => {
    // Check if performance APIs are supported
    if (typeof window !== 'undefined' && 'performance' in window) {
      setIsSupported(true);

      // Measure First Contentful Paint (FCP)
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'paint' && entry.name === 'first-contentful-paint') {
            setMetrics(prev => ({ ...prev, fcp: entry.startTime }));
          }
        });
      });
      observer.observe({ entryTypes: ['paint'] });

      // Measure Time to First Byte (TTFB)
      const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigationEntry) {
        const ttfb = navigationEntry.responseStart - navigationEntry.requestStart;
        setMetrics(prev => ({ ...prev, ttfb }));
      }

      return () => observer.disconnect();
    }
  }, []);

  useEffect(() => {
    if (!isSupported) return;

    // Web Vitals measurement using the web-vitals library pattern
    const measureWebVitals = async () => {
      try {
        // Measure Largest Contentful Paint (LCP)
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          setMetrics(prev => ({ ...prev, lcp: lastEntry.startTime }));
        });
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

        // Measure First Input Delay (FID)
        const fidObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            if (entry.entryType === 'first-input') {
              const fid = entry.processingStart - entry.startTime;
              setMetrics(prev => ({ ...prev, fid }));
            }
          });
        });
        fidObserver.observe({ entryTypes: ['first-input'] });

        // Measure Cumulative Layout Shift (CLS)
        let clsValue = 0;
        const clsObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              clsValue += (entry as any).value;
              setMetrics(prev => ({ ...prev, cls: clsValue }));
            }
          });
        });
        clsObserver.observe({ entryTypes: ['layout-shift'] });

        // Cleanup function
        return () => {
          lcpObserver.disconnect();
          fidObserver.disconnect();
          clsObserver.disconnect();
        };
      } catch (error) {
        console.warn('Performance measurement failed:', error);
      }
    };

    const cleanup = measureWebVitals();
    return () => {
      if (cleanup instanceof Promise) {
        cleanup.then(cleanupFn => cleanupFn && cleanupFn());
      } else if (typeof cleanup === 'function') {
        cleanup();
      }
    };
  }, [isSupported]);

  // Send metrics to analytics
  const reportMetrics = useCallback((analyticsEndpoint?: string) => {
    if (!analyticsEndpoint || !isSupported) return;

    const validMetrics = Object.entries(metrics)
      .filter(([_, value]) => value !== null)
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

    if (Object.keys(validMetrics).length > 0) {
      // Send to analytics service
      fetch(analyticsEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: window.location.href,
          timestamp: Date.now(),
          metrics: validMetrics,
        }),
      }).catch((error) => {
        console.warn('Failed to report performance metrics:', error);
      });
    }
  }, [metrics, isSupported]);

  // Get performance grade
  const getGrade = useCallback(() => {
    if (!isSupported) return 'N/A';

    const scores = {
      lcp: metrics.lcp !== null ? (metrics.lcp <= 2500 ? 'good' : metrics.lcp <= 4000 ? 'needs-improvement' : 'poor') : null,
      fid: metrics.fid !== null ? (metrics.fid <= 100 ? 'good' : metrics.fid <= 300 ? 'needs-improvement' : 'poor') : null,
      cls: metrics.cls !== null ? (metrics.cls <= 0.1 ? 'good' : metrics.cls <= 0.25 ? 'needs-improvement' : 'poor') : null,
      fcp: metrics.fcp !== null ? (metrics.fcp <= 1800 ? 'good' : metrics.fcp <= 3000 ? 'needs-improvement' : 'poor') : null,
      ttfb: metrics.ttfb !== null ? (metrics.ttfb <= 800 ? 'good' : metrics.ttfb <= 1800 ? 'needs-improvement' : 'poor') : null,
    };

    const validScores = Object.values(scores).filter(score => score !== null);
    const goodScores = validScores.filter(score => score === 'good').length;
    const totalScores = validScores.length;

    if (totalScores === 0) return 'N/A';
    if (goodScores / totalScores >= 0.8) return 'good';
    if (goodScores / totalScores >= 0.5) return 'needs-improvement';
    return 'poor';
  }, [metrics, isSupported]);

  return {
    metrics,
    isSupported,
    reportMetrics,
    grade: getGrade(),
  };
}

// Hook for measuring component render performance
export function useRenderPerformance(componentName: string) {
  const [renderTime, setRenderTime] = useState<number | null>(null);
  const [renderCount, setRenderCount] = useState(0);

  useEffect(() => {
    const startTime = performance.now();
    setRenderCount(prev => prev + 1);

    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      setRenderTime(duration);
      
      // Log slow renders in development
      if (process.env.NODE_ENV === 'development' && duration > 16) {
        console.warn(`Slow render detected in ${componentName}: ${duration.toFixed(2)}ms`);
      }
    };
  });

  return { renderTime, renderCount };
}

// Hook for memory usage monitoring
export function useMemoryMonitoring() {
  const [memoryInfo, setMemoryInfo] = useState<{
    usedJSHeapSize?: number;
    totalJSHeapSize?: number;
    jsHeapSizeLimit?: number;
  }>({});

  useEffect(() => {
    if (typeof window === 'undefined' || !(performance as any).memory) {
      return;
    }

    const updateMemoryInfo = () => {
      const memory = (performance as any).memory;
      setMemoryInfo({
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
      });
    };

    updateMemoryInfo();

    // Update every 10 seconds
    const interval = setInterval(updateMemoryInfo, 10000);
    return () => clearInterval(interval);
  }, []);

  const getMemoryUsagePercentage = useCallback(() => {
    if (!memoryInfo.usedJSHeapSize || !memoryInfo.jsHeapSizeLimit) return null;
    return (memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit) * 100;
  }, [memoryInfo]);

  const isMemoryPressureHigh = useCallback(() => {
    const percentage = getMemoryUsagePercentage();
    return percentage !== null && percentage > 90;
  }, [getMemoryUsagePercentage]);

  return {
    memoryInfo,
    getMemoryUsagePercentage,
    isMemoryPressureHigh,
  };
}