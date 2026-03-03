import React from 'react';
import { render, screen } from '@/src/__tests__/utils/test-utils';

// Mock loading components
const Spinner = ({ size = 'default' }: { size?: 'sm' | 'default' | 'lg' }) => (
  <div 
    className={`spinner spinner-${size}`} 
    role="status" 
    aria-label="Loading"
    data-testid="spinner"
  >
    <span className="sr-only">Loading...</span>
  </div>
);

const Skeleton = ({ 
  width = '100%', 
  height = '20px',
  className = '' 
}: { 
  width?: string; 
  height?: string; 
  className?: string;
}) => (
  <div 
    className={`skeleton ${className}`}
    style={{ width, height }}
    data-testid="skeleton"
  />
);

const LoadingCard = () => (
  <div className="loading-card" data-testid="loading-card">
    <Skeleton width="100%" height="24px" className="mb-2" />
    <Skeleton width="80%" height="16px" className="mb-1" />
    <Skeleton width="60%" height="16px" />
  </div>
);

describe('Loading Components', () => {
  describe('Spinner', () => {
    it('renders spinner with accessibility attributes', () => {
      render(<Spinner />);
      
      const spinner = screen.getByRole('status');
      expect(spinner).toBeInTheDocument();
      expect(spinner).toHaveAttribute('aria-label', 'Loading');
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('applies size classes correctly', () => {
      render(<Spinner size="lg" />);
      
      const spinner = screen.getByTestId('spinner');
      expect(spinner).toHaveClass('spinner-lg');
    });
  });

  describe('Skeleton', () => {
    it('renders skeleton with default dimensions', () => {
      render(<Skeleton />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveStyle({ width: '100%', height: '20px' });
    });

    it('applies custom dimensions', () => {
      render(<Skeleton width="200px" height="40px" />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveStyle({ width: '200px', height: '40px' });
    });

    it('applies custom classes', () => {
      render(<Skeleton className="custom-skeleton" />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveClass('skeleton', 'custom-skeleton');
    });
  });

  describe('LoadingCard', () => {
    it('renders multiple skeleton elements', () => {
      render(<LoadingCard />);
      
      const skeletons = screen.getAllByTestId('skeleton');
      expect(skeletons).toHaveLength(3);
    });

    it('has proper loading card structure', () => {
      render(<LoadingCard />);
      
      const loadingCard = screen.getByTestId('loading-card');
      expect(loadingCard).toBeInTheDocument();
      expect(loadingCard).toHaveClass('loading-card');
    });
  });
});
