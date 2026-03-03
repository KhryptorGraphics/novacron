import { renderHook, act } from '@testing-library/react';
import { usePerformance } from '@/hooks/usePerformance';

// Mock performance API
const mockPerformanceMark = jest.fn();
const mockPerformanceMeasure = jest.fn();
const mockPerformanceGetEntriesByType = jest.fn();

Object.defineProperty(window, 'performance', {
  value: {
    mark: mockPerformanceMark,
    measure: mockPerformanceMeasure,
    getEntriesByType: mockPerformanceGetEntriesByType,
    now: () => Date.now(),
  },
  writable: true,
});

describe('usePerformance', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('tracks component mount time', () => {
    const { result } = renderHook(() => usePerformance('TestComponent'));

    expect(mockPerformanceMark).toHaveBeenCalledWith('TestComponent-start');
  });

  it('measures performance on unmount', () => {
    const { unmount } = renderHook(() => usePerformance('TestComponent'));

    unmount();

    expect(mockPerformanceMark).toHaveBeenCalledWith('TestComponent-end');
    expect(mockPerformanceMeasure).toHaveBeenCalledWith(
      'TestComponent-duration',
      'TestComponent-start',
      'TestComponent-end'
    );
  });

  it('provides manual measurement function', () => {
    const { result } = renderHook(() => usePerformance('TestComponent'));

    act(() => {
      result.current.measurePerformance('custom-operation');
    });

    expect(mockPerformanceMark).toHaveBeenCalledWith('custom-operation-end');
    expect(mockPerformanceMeasure).toHaveBeenCalledWith(
      'custom-operation-duration',
      'TestComponent-start',
      'custom-operation-end'
    );
  });

  it('returns performance metrics', () => {
    mockPerformanceGetEntriesByType.mockReturnValue([
      { name: 'TestComponent-duration', duration: 150 },
    ]);

    const { result } = renderHook(() => usePerformance('TestComponent'));

    act(() => {
      const metrics = result.current.getMetrics();
      expect(metrics).toEqual([
        { name: 'TestComponent-duration', duration: 150 },
      ]);
    });

    expect(mockPerformanceGetEntriesByType).toHaveBeenCalledWith('measure');
  });
});
