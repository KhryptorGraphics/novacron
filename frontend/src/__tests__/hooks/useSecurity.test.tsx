import { renderHook, waitFor } from '@testing-library/react';
import { useCompliance, useSecurityEvents } from '../../hooks/useSecurity';
import { securityAPI } from '../../lib/api/security';

jest.mock('../../lib/api/security', () => ({
  securityAPI: {
    getSecurityEvents: jest.fn(),
    acknowledgeSecurityEvent: jest.fn(),
    getComplianceRequirements: jest.fn(),
    getComplianceByCategory: jest.fn(),
    triggerComplianceCheck: jest.fn(),
  },
}));

jest.mock('../../components/ui/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}));

describe('useSecurity hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns an honest empty state when security events fail to load', async () => {
    (securityAPI.getSecurityEvents as jest.Mock).mockRejectedValueOnce(
      new Error('events unavailable'),
    );

    const { result } = renderHook(() => useSecurityEvents(false));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('events unavailable');
    expect(result.current.events).toEqual([]);
    expect(result.current.total).toBe(0);
  });

  it('returns an honest empty state when compliance data fails to load', async () => {
    (securityAPI.getComplianceRequirements as jest.Mock).mockRejectedValueOnce(
      new Error('compliance unavailable'),
    );
    (securityAPI.getComplianceByCategory as jest.Mock).mockRejectedValueOnce(
      new Error('compliance unavailable'),
    );

    const { result } = renderHook(() => useCompliance());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('compliance unavailable');
    expect(result.current.requirements).toEqual([]);
    expect(result.current.categoryBreakdown).toEqual([]);
  });
});
