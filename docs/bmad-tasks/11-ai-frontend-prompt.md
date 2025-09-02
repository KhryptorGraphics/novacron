# AI Frontend Development Prompt - NovaCron Dashboard

## Objective
Generate a modern, responsive, and highly interactive dashboard for NovaCron VM management platform using Next.js 13.5.6, React 18.2.0, and Tailwind CSS 3.3.0.

## Design Requirements

### Visual Design System
- **Theme**: Dark mode default with light mode toggle
- **Color Palette**: 
  - Primary: Blue (#3B82F6)
  - Success: Green (#10B981)
  - Warning: Yellow (#F59E0B)
  - Error: Red (#EF4444)
  - Background: Dark (#0F172A) / Light (#FFFFFF)
- **Typography**: Inter font family, responsive sizing
- **Spacing**: 8px grid system
- **Shadows**: Subtle elevation for cards
- **Animations**: Smooth transitions, skeleton loading

### Layout Structure
```
┌─────────────────────────────────────────┐
│ Header (Logo | Search | User | Settings)│
├────────┬────────────────────────────────┤
│Sidebar │                                │
│  Nav   │         Main Content Area      │
│  Menu  │                                │
│        │                                │
└────────┴────────────────────────────────┘
```

## Component Specifications

### 1. Dashboard Overview Page
```typescript
// components/dashboard/Overview.tsx
interface OverviewProps {
  stats: {
    totalVMs: number;
    runningVMs: number;
    cpuUsage: number;
    memoryUsage: number;
    storageUsage: number;
    monthlyCost: number;
  };
  alerts: Alert[];
  recentActivity: Activity[];
}

// Features:
- Real-time stats cards with trend indicators
- Interactive donut charts for resource usage
- Alert timeline with severity indicators
- Activity feed with live updates
- Quick action buttons (Create VM, etc.)
```

### 2. VM Management Grid
```typescript
// components/vms/VMGrid.tsx
interface VMGridProps {
  vms: VM[];
  view: 'grid' | 'list' | 'cards';
  onSelect: (vm: VM) => void;
  onAction: (action: string, vm: VM) => void;
}

// Features:
- Virtualized scrolling for 1000+ VMs
- Real-time status updates via WebSocket
- Inline actions (start/stop/restart)
- Bulk selection with shift+click
- Advanced filtering and sorting
- Drag-and-drop for migration
```

### 3. Resource Monitoring Charts
```typescript
// components/monitoring/ResourceCharts.tsx
interface ResourceChartsProps {
  timeRange: '1h' | '24h' | '7d' | '30d';
  metrics: MetricData[];
  vmId?: string;
}

// Features:
- Line charts with zoom/pan using Recharts
- Real-time data streaming
- Threshold indicators
- Predictive trend lines
- Export to PNG/CSV
- Responsive breakpoints
```

### 4. AI Optimization Panel
```typescript
// components/ai/OptimizationPanel.tsx
interface OptimizationPanelProps {
  recommendations: Recommendation[];
  savings: MonthlySavings;
  onApply: (rec: Recommendation) => void;
  onDismiss: (rec: Recommendation) => void;
}

// Features:
- Animated savings counter
- Confidence score visualization
- Before/after comparison
- One-click apply with confirmation
- Undo capability
- Learning feedback system
```

### 5. Multi-Cloud Topology View
```typescript
// components/topology/CloudTopology.tsx
interface CloudTopologyProps {
  providers: CloudProvider[];
  connections: Connection[];
  onNodeClick: (node: Node) => void;
}

// Features:
- Interactive network diagram using D3.js
- Zoom/pan navigation
- Real-time connection status
- Traffic flow animation
- Node grouping by region
- Mini-map for navigation
```

## Interactive Features

### Real-time Updates
```javascript
// hooks/useRealTimeData.ts
const useRealTimeData = (endpoint: string) => {
  useEffect(() => {
    const ws = new WebSocket(`wss://api.novacron.io/ws/${endpoint}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updateState(data);
    };
    return () => ws.close();
  }, [endpoint]);
};
```

### Keyboard Shortcuts
```javascript
// hooks/useKeyboardShortcuts.ts
const shortcuts = {
  'cmd+k': () => openCommandPalette(),
  'cmd+n': () => createNewVM(),
  'cmd+/': () => toggleSearch(),
  'esc': () => closeModals(),
  'g h': () => navigate('/'),
  'g v': () => navigate('/vms'),
};
```

### Drag and Drop
```javascript
// components/dnd/DraggableVM.tsx
<DndProvider backend={HTML5Backend}>
  <DraggableVM vm={vm} onDrop={handleMigration} />
  <DropZone host={host} accepts={['vm']} />
</DndProvider>
```

## State Management

### Redux Toolkit Setup
```typescript
// store/slices/vmSlice.ts
const vmSlice = createSlice({
  name: 'vms',
  initialState: {
    items: [],
    loading: false,
    error: null,
    filters: {},
    selectedIds: [],
  },
  reducers: {
    setVMs: (state, action) => {
      state.items = action.payload;
    },
    updateVM: (state, action) => {
      const index = state.items.findIndex(vm => vm.id === action.payload.id);
      if (index !== -1) {
        state.items[index] = action.payload;
      }
    },
  },
});
```

## Performance Optimizations

### Code Splitting
```javascript
// Lazy load heavy components
const MonitoringDashboard = lazy(() => import('./MonitoringDashboard'));
const TopologyView = lazy(() => import('./TopologyView'));

// Use Suspense with fallback
<Suspense fallback={<LoadingSpinner />}>
  <MonitoringDashboard />
</Suspense>
```

### Virtualization
```javascript
// For large lists
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={600}
  itemCount={vms.length}
  itemSize={80}
  width="100%"
>
  {VMRow}
</FixedSizeList>
```

### Memoization
```javascript
// Prevent unnecessary re-renders
const MemoizedVMCard = memo(VMCard, (prevProps, nextProps) => {
  return prevProps.vm.id === nextProps.vm.id &&
         prevProps.vm.status === nextProps.vm.status;
});
```

## Responsive Design

### Breakpoints
```css
/* tailwind.config.js */
module.exports = {
  theme: {
    screens: {
      'sm': '640px',   // Mobile landscape
      'md': '768px',   // Tablet
      'lg': '1024px',  // Desktop
      'xl': '1280px',  // Large desktop
      '2xl': '1536px', // Ultra-wide
    }
  }
}
```

### Mobile Adaptations
```jsx
// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
  {vms.map(vm => <VMCard key={vm.id} vm={vm} />)}
</div>

// Collapsible sidebar on mobile
<aside className="w-64 lg:w-72 hidden md:block">
  <Navigation />
</aside>
```

## Accessibility (WCAG 2.1 AA)

### ARIA Labels
```jsx
<button 
  aria-label="Start virtual machine"
  aria-pressed={vm.status === 'running'}
  role="button"
  tabIndex={0}
>
  <PlayIcon aria-hidden="true" />
</button>
```

### Keyboard Navigation
```jsx
// Focus management
const handleKeyDown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    handleClick();
  }
};
```

### Screen Reader Support
```jsx
<div role="region" aria-label="VM Statistics">
  <h2 className="sr-only">Virtual Machine Statistics</h2>
  <div role="status" aria-live="polite">
    {loading ? 'Loading VMs...' : `${vms.length} VMs loaded`}
  </div>
</div>
```

## Error Handling

### Error Boundaries
```jsx
class ErrorBoundary extends Component {
  componentDidCatch(error, errorInfo) {
    logErrorToService(error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback onReset={this.reset} />;
    }
    return this.props.children;
  }
}
```

### User-Friendly Errors
```jsx
const ErrorMessage = ({ error }) => (
  <Alert variant="error">
    <AlertTitle>Something went wrong</AlertTitle>
    <AlertDescription>
      {error.message || 'An unexpected error occurred'}
    </AlertDescription>
    <Button onClick={retry}>Try Again</Button>
  </Alert>
);
```

## Testing Requirements

### Component Tests
```javascript
// __tests__/VMCard.test.tsx
describe('VMCard', () => {
  it('displays VM name and status', () => {
    render(<VMCard vm={mockVM} />);
    expect(screen.getByText('production-web-01')).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument();
  });
  
  it('handles start action correctly', async () => {
    const onAction = jest.fn();
    render(<VMCard vm={mockVM} onAction={onAction} />);
    fireEvent.click(screen.getByLabelText('Start VM'));
    expect(onAction).toHaveBeenCalledWith('start', mockVM);
  });
});
```

## File Structure
```
src/
├── app/
│   ├── dashboard/
│   │   └── page.tsx
│   ├── vms/
│   │   └── page.tsx
│   └── layout.tsx
├── components/
│   ├── ui/           # Reusable UI components
│   ├── dashboard/    # Dashboard-specific
│   ├── vms/         # VM management
│   └── monitoring/  # Monitoring charts
├── hooks/           # Custom React hooks
├── lib/            # Utilities and helpers
├── store/          # Redux store
└── styles/         # Global styles
```

## Delivery Requirements

1. **Complete component library** with Storybook documentation
2. **Responsive layouts** tested on all major breakpoints
3. **Dark/light theme** with system preference detection
4. **Loading states** for all async operations
5. **Error handling** with user-friendly messages
6. **Accessibility** WCAG 2.1 AA compliant
7. **Performance** Lighthouse score >90
8. **Testing** >80% coverage with Jest/React Testing Library
9. **Documentation** JSDoc comments and README
10. **Live demo** Deployed to Vercel/Netlify

---
*AI Frontend Prompt generated using BMad Generate AI Frontend Prompt Task*
*Date: 2025-01-30*
*Framework: Next.js 13.5.6 + React 18.2.0 + Tailwind CSS 3.3.0*