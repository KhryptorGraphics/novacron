// Export all NovaCron UI flows and components
export { VMLifecycleFlow } from './VMLifecycleFlow';
export { MigrationWorkflow } from './MigrationWorkflow';
export { BackupRecoveryFlow } from './BackupRecoveryFlow';
export { PerformanceOptimizationFlow } from './PerformanceOptimizationFlow';
export { UserSelfServicePortal } from './UserSelfServicePortal';
export { MobileAppControls } from './MobileAppControls';

// Re-export shared UI components
export { 
  WizardComponents,
  Wizard, 
  WizardHeader, 
  WizardContent, 
  WizardFooter, 
  WizardStep, 
  WizardValidation, 
  WizardSummary, 
  useWizard 
} from '../ui/wizard';

export { 
  AdvancedFormComponents,
  AdvancedForm, 
  FormField, 
  FormSection, 
  FormActions, 
  FormValidationSummary, 
  useAdvancedForm 
} from '../ui/advanced-form';

export { 
  KeyboardShortcutComponents,
  KeyboardShortcutProvider, 
  Shortcut, 
  KeyDisplay, 
  QuickShortcut, 
  useKeyboardShortcuts, 
  useCommonShortcuts 
} from '../ui/keyboard-shortcuts';

export { 
  AccessibilityComponents,
  AccessibilityProvider, 
  AccessibilityPanel, 
  ScreenReaderOnly, 
  Announcement, 
  FocusTrap, 
  useAccessibility 
} from '../ui/accessibility';

// Flow types and interfaces
export interface FlowComponentProps {
  className?: string;
  onComplete?: () => void;
  onCancel?: () => void;
}

export interface FlowStep {
  id: string;
  title: string;
  component: React.ComponentType<any>;
  validation?: () => boolean;
  optional?: boolean;
}

export interface FlowData {
  [key: string]: any;
}

// Flow navigation helper
export const FlowNavigation = {
  VM_LIFECYCLE: '/flows/vm-lifecycle',
  MIGRATION: '/flows/migration',
  BACKUP_RECOVERY: '/flows/backup-recovery',
  PERFORMANCE: '/flows/performance',
  SELF_SERVICE: '/flows/self-service',
  MOBILE: '/flows/mobile'
} as const;

// Flow categories
export const FlowCategories = {
  INFRASTRUCTURE: 'infrastructure',
  OPERATIONS: 'operations',
  SELF_SERVICE: 'self-service',
  MOBILE: 'mobile'
} as const;

// Flow metadata
export const FlowMetadata = {
  'vm-lifecycle': {
    title: 'VM Lifecycle Management',
    description: 'Create and manage virtual machine lifecycles with step-by-step guidance',
    category: FlowCategories.INFRASTRUCTURE,
    complexity: 'intermediate',
    estimatedTime: '10-15 minutes',
    prerequisites: ['Resource planning', 'Network configuration'],
    features: [
      'Step-by-step VM creation wizard',
      'Resource configuration with recommendations',
      'Template selection with filtering',
      'Network and storage configuration',
      'Cost estimation',
      'Auto-save and draft recovery'
    ]
  },
  'migration': {
    title: 'VM Migration Workflow',
    description: 'Migrate virtual machines between locations with zero-downtime',
    category: FlowCategories.OPERATIONS,
    complexity: 'advanced',
    estimatedTime: '30-60 minutes',
    prerequisites: ['Source and target environments', 'Network connectivity'],
    features: [
      'Pre-migration validation',
      'Real-time progress monitoring',
      'Bandwidth and ETA tracking',
      'Rollback controls',
      'Post-migration verification'
    ]
  },
  'backup-recovery': {
    title: 'Backup & Recovery Flow',
    description: 'Comprehensive backup scheduling and point-in-time recovery',
    category: FlowCategories.OPERATIONS,
    complexity: 'intermediate',
    estimatedTime: '5-30 minutes',
    prerequisites: ['Storage configuration', 'Backup policies'],
    features: [
      'Policy configuration interface',
      'Selective restore options',
      'Progress tracking with logs',
      'Verification and testing',
      'Automated scheduling'
    ]
  },
  'performance': {
    title: 'Performance Optimization',
    description: 'AI-powered performance analysis and optimization recommendations',
    category: FlowCategories.OPERATIONS,
    complexity: 'advanced',
    estimatedTime: '15-45 minutes',
    prerequisites: ['Performance metrics', 'System monitoring'],
    features: [
      'Bottleneck identification wizard',
      'AI optimization recommendations',
      'One-click optimization actions',
      'Before/after comparison',
      'Continuous monitoring setup'
    ]
  },
  'self-service': {
    title: 'User Self-Service Portal',
    description: 'Empower users with self-service IT resource requests and management',
    category: FlowCategories.SELF_SERVICE,
    complexity: 'beginner',
    estimatedTime: '5-10 minutes',
    prerequisites: ['User account', 'Cost center assignment'],
    features: [
      'Resource request workflow',
      'Approval status tracking',
      'Usage analytics and reports',
      'Support ticket integration',
      'Knowledge base access'
    ]
  },
  'mobile': {
    title: 'Mobile App Controls',
    description: 'Native mobile experience for VM management on-the-go',
    category: FlowCategories.MOBILE,
    complexity: 'beginner',
    estimatedTime: 'Real-time',
    prerequisites: ['Mobile device', 'Biometric authentication'],
    features: [
      'Quick VM actions (start/stop/restart)',
      'Push notifications for alerts',
      'Biometric authentication',
      'Offline mode with sync',
      'Gesture controls'
    ]
  }
} as const;