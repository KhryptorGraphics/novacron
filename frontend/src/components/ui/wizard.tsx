'use client';

import React, { createContext, useContext, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  ChevronLeft, 
  ChevronRight, 
  CheckCircle, 
  Circle,
  AlertTriangle,
  Info
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface WizardStep {
  id: string;
  title: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  optional?: boolean;
  validation?: () => boolean | Promise<boolean>;
}

interface WizardContextType {
  steps: WizardStep[];
  currentStepIndex: number;
  currentStep: WizardStep;
  isFirstStep: boolean;
  isLastStep: boolean;
  completedSteps: Set<string>;
  nextStep: () => Promise<void>;
  prevStep: () => void;
  goToStep: (stepId: string) => void;
  markStepComplete: (stepId: string) => void;
  validateCurrentStep: () => Promise<boolean>;
}

const WizardContext = createContext<WizardContextType | undefined>(undefined);

export const useWizard = () => {
  const context = useContext(WizardContext);
  if (!context) {
    throw new Error('useWizard must be used within a WizardProvider');
  }
  return context;
};

interface WizardProps {
  children: React.ReactNode;
  steps: WizardStep[];
  onComplete?: () => void;
  onStepChange?: (stepIndex: number, step: WizardStep) => void;
  className?: string;
}

export function Wizard({ children, steps, onComplete, onStepChange, className }: WizardProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());

  const currentStep = steps[currentStepIndex];
  const isFirstStep = currentStepIndex === 0;
  const isLastStep = currentStepIndex === steps.length - 1;

  const validateCurrentStep = async (): Promise<boolean> => {
    if (!currentStep.validation) return true;
    
    try {
      return await currentStep.validation();
    } catch (error) {
      console.error('Step validation failed:', error);
      return false;
    }
  };

  const nextStep = async () => {
    const isValid = await validateCurrentStep();
    if (!isValid) return;

    markStepComplete(currentStep.id);
    
    if (isLastStep) {
      onComplete?.();
    } else {
      const newIndex = currentStepIndex + 1;
      setCurrentStepIndex(newIndex);
      onStepChange?.(newIndex, steps[newIndex]);
    }
  };

  const prevStep = () => {
    if (!isFirstStep) {
      const newIndex = currentStepIndex - 1;
      setCurrentStepIndex(newIndex);
      onStepChange?.(newIndex, steps[newIndex]);
    }
  };

  const goToStep = (stepId: string) => {
    const stepIndex = steps.findIndex(step => step.id === stepId);
    if (stepIndex !== -1 && stepIndex <= currentStepIndex + 1) {
      setCurrentStepIndex(stepIndex);
      onStepChange?.(stepIndex, steps[stepIndex]);
    }
  };

  const markStepComplete = (stepId: string) => {
    setCompletedSteps(prev => new Set(prev).add(stepId));
  };

  const contextValue: WizardContextType = {
    steps,
    currentStepIndex,
    currentStep,
    isFirstStep,
    isLastStep,
    completedSteps,
    nextStep,
    prevStep,
    goToStep,
    markStepComplete,
    validateCurrentStep
  };

  return (
    <WizardContext.Provider value={contextValue}>
      <div className={cn('w-full', className)}>
        {children}
      </div>
    </WizardContext.Provider>
  );
}

interface WizardHeaderProps {
  className?: string;
  showProgress?: boolean;
}

export function WizardHeader({ className, showProgress = true }: WizardHeaderProps) {
  const { steps, currentStepIndex, completedSteps } = useWizard();
  
  const progressPercentage = ((currentStepIndex + 1) / steps.length) * 100;

  return (
    <Card className={cn('mb-6', className)}>
      <CardContent className="pt-6">
        {showProgress && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Progress</span>
              <span className="text-sm text-gray-600">
                Step {currentStepIndex + 1} of {steps.length}
              </span>
            </div>
            <Progress value={progressPercentage} className="w-full" />
          </div>
        )}
        
        <div className="flex items-center space-x-2 overflow-x-auto pb-2">
          {steps.map((step, index) => {
            const Icon = step.icon || Circle;
            const isCompleted = completedSteps.has(step.id);
            const isCurrent = index === currentStepIndex;
            const isAccessible = index <= currentStepIndex || isCompleted;
            
            return (
              <React.Fragment key={step.id}>
                <div
                  className={cn(
                    'flex items-center space-x-2 px-3 py-2 rounded-lg transition-all cursor-pointer min-w-fit',
                    isCurrent && 'bg-blue-100 text-blue-800 ring-2 ring-blue-500',
                    isCompleted && !isCurrent && 'bg-green-100 text-green-800',
                    !isCurrent && !isCompleted && isAccessible && 'bg-gray-100 text-gray-600 hover:bg-gray-200',
                    !isAccessible && 'bg-gray-50 text-gray-400 cursor-not-allowed opacity-60'
                  )}
                  onClick={() => isAccessible && step.id}
                >
                  {isCompleted ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                  <span className="text-sm font-medium whitespace-nowrap">{step.title}</span>
                  {step.optional && (
                    <Badge variant="secondary" className="text-xs ml-1">
                      Optional
                    </Badge>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <ChevronRight className="h-4 w-4 text-gray-400 flex-shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

interface WizardContentProps {
  children: React.ReactNode;
  className?: string;
}

export function WizardContent({ children, className }: WizardContentProps) {
  const { currentStep } = useWizard();

  return (
    <Card className={cn('mb-6', className)}>
      <CardHeader>
        <CardTitle>{currentStep.title}</CardTitle>
        {currentStep.description && (
          <CardDescription>{currentStep.description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  );
}

interface WizardFooterProps {
  className?: string;
  nextLabel?: string;
  prevLabel?: string;
  finishLabel?: string;
  showCancel?: boolean;
  onCancel?: () => void;
  customActions?: React.ReactNode;
}

export function WizardFooter({ 
  className, 
  nextLabel = 'Next',
  prevLabel = 'Previous', 
  finishLabel = 'Finish',
  showCancel = false,
  onCancel,
  customActions 
}: WizardFooterProps) {
  const { isFirstStep, isLastStep, nextStep, prevStep } = useWizard();

  return (
    <div className={cn('flex items-center justify-between', className)}>
      <div className="flex space-x-2">
        {showCancel && (
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        )}
        {customActions}
      </div>
      
      <div className="flex space-x-2">
        <Button
          variant="outline"
          onClick={prevStep}
          disabled={isFirstStep}
        >
          <ChevronLeft className="mr-2 h-4 w-4" />
          {prevLabel}
        </Button>

        <Button onClick={nextStep}>
          {isLastStep ? finishLabel : nextLabel}
          {!isLastStep && <ChevronRight className="ml-2 h-4 w-4" />}
        </Button>
      </div>
    </div>
  );
}

// Wizard Step Component for individual step content
interface WizardStepProps {
  stepId: string;
  children: React.ReactNode;
  className?: string;
}

export function WizardStep({ stepId, children, className }: WizardStepProps) {
  const { currentStep } = useWizard();
  
  if (currentStep.id !== stepId) {
    return null;
  }

  return (
    <div className={cn('w-full', className)}>
      {children}
    </div>
  );
}

// Wizard Validation Component
interface WizardValidationProps {
  errors?: string[];
  warnings?: string[];
  className?: string;
}

export function WizardValidation({ errors = [], warnings = [], className }: WizardValidationProps) {
  if (errors.length === 0 && warnings.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-2 mb-4', className)}>
      {errors.map((error, index) => (
        <div key={`error-${index}`} className="flex items-center space-x-2 p-3 rounded-lg bg-red-50 border border-red-200">
          <AlertTriangle className="h-4 w-4 text-red-600 flex-shrink-0" />
          <span className="text-sm text-red-800">{error}</span>
        </div>
      ))}
      
      {warnings.map((warning, index) => (
        <div key={`warning-${index}`} className="flex items-center space-x-2 p-3 rounded-lg bg-yellow-50 border border-yellow-200">
          <Info className="h-4 w-4 text-yellow-600 flex-shrink-0" />
          <span className="text-sm text-yellow-800">{warning}</span>
        </div>
      ))}
    </div>
  );
}

// Wizard Summary Component
interface WizardSummaryProps {
  data: Record<string, any>;
  className?: string;
  renderField?: (key: string, value: any) => React.ReactNode;
}

export function WizardSummary({ data, className, renderField }: WizardSummaryProps) {
  const defaultRenderField = (key: string, value: any) => (
    <div key={key} className="flex justify-between py-2 border-b border-gray-100">
      <span className="font-medium capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
      <span className="text-gray-600">
        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
      </span>
    </div>
  );

  return (
    <div className={cn('space-y-4', className)}>
      <h3 className="text-lg font-semibold">Summary</h3>
      <div className="space-y-1">
        {Object.entries(data).map(([key, value]) => 
          renderField ? renderField(key, value) : defaultRenderField(key, value)
        )}
      </div>
    </div>
  );
}

// Export all components as a combined object for easier imports
export const WizardComponents = {
  Wizard,
  WizardHeader,
  WizardContent,
  WizardFooter,
  WizardStep,
  WizardValidation,
  WizardSummary,
  useWizard
};