'use client';

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Save,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Info,
  X,
  Eye,
  EyeOff,
  Calendar,
  Clock,
  Upload,
  FileText,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Form validation types
interface ValidationRule {
  type: 'required' | 'minLength' | 'maxLength' | 'pattern' | 'custom' | 'email' | 'url' | 'number';
  value?: any;
  message: string;
}

interface FieldConfig {
  name: string;
  label: string;
  type: 'text' | 'email' | 'password' | 'number' | 'textarea' | 'select' | 'checkbox' | 'radio' | 'file' | 'date' | 'datetime-local';
  placeholder?: string;
  description?: string;
  defaultValue?: any;
  options?: { label: string; value: any }[];
  validation?: ValidationRule[];
  dependencies?: { field: string; condition: (value: any) => boolean }[];
  formatters?: {
    display?: (value: any) => string;
    storage?: (value: any) => any;
  };
  autocomplete?: string;
  multiple?: boolean;
  accept?: string; // for file inputs
  disabled?: boolean;
  readonly?: boolean;
}

interface FormError {
  field: string;
  message: string;
  type: 'error' | 'warning';
}

interface FormContextType {
  data: Record<string, any>;
  errors: FormError[];
  isDirty: boolean;
  isSubmitting: boolean;
  autoSaveEnabled: boolean;
  lastSaved?: Date;
  updateField: (name: string, value: any) => void;
  validateField: (name: string) => boolean;
  validateForm: () => boolean;
  resetForm: () => void;
  submitForm: () => Promise<void>;
  addError: (field: string, message: string, type?: 'error' | 'warning') => void;
  clearErrors: (field?: string) => void;
  getFieldValue: (name: string) => any;
  isFieldVisible: (field: FieldConfig) => boolean;
}

const FormContext = createContext<FormContextType | undefined>(undefined);

export const useAdvancedForm = () => {
  const context = useContext(FormContext);
  if (!context) {
    throw new Error('useAdvancedForm must be used within an AdvancedFormProvider');
  }
  return context;
};

interface AdvancedFormProps {
  children: React.ReactNode;
  fields: FieldConfig[];
  initialData?: Record<string, any>;
  onSubmit?: (data: Record<string, any>) => Promise<void>;
  onChange?: (data: Record<string, any>) => void;
  autoSave?: boolean;
  autoSaveInterval?: number; // seconds
  className?: string;
  title?: string;
  description?: string;
}

export function AdvancedForm({ 
  children, 
  fields, 
  initialData = {}, 
  onSubmit,
  onChange,
  autoSave = false,
  autoSaveInterval = 30,
  className,
  title,
  description
}: AdvancedFormProps) {
  const [data, setData] = useState<Record<string, any>>(initialData);
  const [errors, setErrors] = useState<FormError[]>([]);
  const [isDirty, setIsDirty] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date>();
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout>();

  // Initialize default values
  useEffect(() => {
    const defaultData = { ...initialData };
    fields.forEach(field => {
      if (field.defaultValue !== undefined && !(field.name in defaultData)) {
        defaultData[field.name] = field.defaultValue;
      }
    });
    setData(defaultData);
  }, [fields, initialData]);

  // Auto-save functionality
  useEffect(() => {
    if (autoSave && isDirty) {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
      
      autoSaveTimeoutRef.current = setTimeout(async () => {
        try {
          const savedData = { ...data };
          // Simulate auto-save
          await new Promise(resolve => setTimeout(resolve, 500));
          localStorage.setItem('form-autosave', JSON.stringify(savedData));
          setLastSaved(new Date());
          setIsDirty(false);
        } catch (error) {
          console.error('Auto-save failed:', error);
        }
      }, autoSaveInterval * 1000);
    }

    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, [data, isDirty, autoSave, autoSaveInterval]);

  const updateField = (name: string, value: any) => {
    setData(prev => ({ ...prev, [name]: value }));
    setIsDirty(true);
    clearErrors(name);
    
    // Apply formatters
    const field = fields.find(f => f.name === name);
    if (field?.formatters?.storage) {
      value = field.formatters.storage(value);
    }
    
    onChange?.(data);
  };

  const validateField = (name: string): boolean => {
    const field = fields.find(f => f.name === name);
    if (!field || !field.validation) return true;

    const value = data[name];
    const fieldErrors: FormError[] = [];

    for (const rule of field.validation) {
      switch (rule.type) {
        case 'required':
          if (!value || (typeof value === 'string' && value.trim() === '')) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'minLength':
          if (typeof value === 'string' && value.length < rule.value) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'maxLength':
          if (typeof value === 'string' && value.length > rule.value) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'pattern':
          if (typeof value === 'string' && !rule.value.test(value)) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'email':
          const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (typeof value === 'string' && value && !emailPattern.test(value)) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'url':
          try {
            if (value && typeof value === 'string') {
              new URL(value);
            }
          } catch {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'number':
          if (value && isNaN(Number(value))) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
        
        case 'custom':
          if (typeof rule.value === 'function' && !rule.value(value, data)) {
            fieldErrors.push({ field: name, message: rule.message, type: 'error' });
          }
          break;
      }
    }

    // Clear existing errors for this field and add new ones
    setErrors(prev => [
      ...prev.filter(error => error.field !== name),
      ...fieldErrors
    ]);

    return fieldErrors.length === 0;
  };

  const validateForm = (): boolean => {
    let isValid = true;
    
    fields.forEach(field => {
      if (!isFieldVisible(field)) return;
      if (!validateField(field.name)) {
        isValid = false;
      }
    });

    return isValid;
  };

  const resetForm = () => {
    setData(initialData);
    setErrors([]);
    setIsDirty(false);
    setLastSaved(undefined);
  };

  const submitForm = async () => {
    if (!validateForm()) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit?.(data);
      setIsDirty(false);
      localStorage.removeItem('form-autosave');
    } catch (error) {
      console.error('Form submission failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const addError = (field: string, message: string, type: 'error' | 'warning' = 'error') => {
    setErrors(prev => [...prev.filter(e => e.field !== field || e.message !== message), { field, message, type }]);
  };

  const clearErrors = (field?: string) => {
    if (field) {
      setErrors(prev => prev.filter(error => error.field !== field));
    } else {
      setErrors([]);
    }
  };

  const getFieldValue = (name: string) => {
    return data[name];
  };

  const isFieldVisible = (field: FieldConfig): boolean => {
    if (!field.dependencies) return true;
    
    return field.dependencies.every(dep => {
      const dependencyValue = data[dep.field];
      return dep.condition(dependencyValue);
    });
  };

  const contextValue: FormContextType = {
    data,
    errors,
    isDirty,
    isSubmitting,
    autoSaveEnabled: autoSave,
    lastSaved,
    updateField,
    validateField,
    validateForm,
    resetForm,
    submitForm,
    addError,
    clearErrors,
    getFieldValue,
    isFieldVisible
  };

  return (
    <FormContext.Provider value={contextValue}>
      <div className={cn('w-full', className)}>
        {(title || description) && (
          <div className="mb-6">
            {title && <h2 className="text-2xl font-bold">{title}</h2>}
            {description && <p className="text-gray-600 mt-1">{description}</p>}
          </div>
        )}
        {children}
      </div>
    </FormContext.Provider>
  );
}

// Form field component
interface FormFieldProps {
  field: FieldConfig;
  className?: string;
}

export function FormField({ field, className }: FormFieldProps) {
  const { updateField, getFieldValue, errors, isFieldVisible } = useAdvancedForm();
  const [showPassword, setShowPassword] = useState(false);
  
  if (!isFieldVisible(field)) {
    return null;
  }

  const value = getFieldValue(field.name);
  const fieldErrors = errors.filter(error => error.field === field.name);
  const hasErrors = fieldErrors.length > 0;

  const handleChange = (newValue: any) => {
    updateField(field.name, newValue);
  };

  const renderField = () => {
    switch (field.type) {
      case 'textarea':
        return (
          <Textarea
            id={field.name}
            placeholder={field.placeholder}
            value={value || ''}
            onChange={(e) => handleChange(e.target.value)}
            disabled={field.disabled}
            readOnly={field.readonly}
            className={hasErrors ? 'border-red-500' : ''}
            rows={4}
          />
        );
      
      case 'select':
        return (
          <Select
            value={value || ''}
            onValueChange={handleChange}
            disabled={field.disabled}
          >
            <SelectTrigger className={hasErrors ? 'border-red-500' : ''}>
              <SelectValue placeholder={field.placeholder} />
            </SelectTrigger>
            <SelectContent>
              {field.options?.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      
      case 'checkbox':
        return (
          <div className="flex items-center space-x-2">
            <Checkbox
              id={field.name}
              checked={value || false}
              onCheckedChange={handleChange}
              disabled={field.disabled}
            />
            <Label htmlFor={field.name} className="text-sm">
              {field.label}
            </Label>
          </div>
        );
      
      case 'file':
        return (
          <div className="space-y-2">
            <Input
              id={field.name}
              type="file"
              onChange={(e) => {
                const file = e.target.files?.[0];
                handleChange(file);
              }}
              disabled={field.disabled}
              accept={field.accept}
              multiple={field.multiple}
              className={hasErrors ? 'border-red-500' : ''}
            />
            {value && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <FileText className="h-4 w-4" />
                <span>{value.name}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleChange(null)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        );
      
      case 'password':
        return (
          <div className="relative">
            <Input
              id={field.name}
              type={showPassword ? 'text' : 'password'}
              placeholder={field.placeholder}
              value={value || ''}
              onChange={(e) => handleChange(e.target.value)}
              disabled={field.disabled}
              readOnly={field.readonly}
              autoComplete={field.autocomplete}
              className={hasErrors ? 'border-red-500' : ''}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3"
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
        );
      
      case 'date':
      case 'datetime-local':
        return (
          <div className="relative">
            <Input
              id={field.name}
              type={field.type}
              value={value || ''}
              onChange={(e) => handleChange(e.target.value)}
              disabled={field.disabled}
              readOnly={field.readonly}
              className={hasErrors ? 'border-red-500' : ''}
            />
            <Calendar className="absolute right-3 top-3 h-4 w-4 text-gray-400 pointer-events-none" />
          </div>
        );
      
      default:
        return (
          <Input
            id={field.name}
            type={field.type}
            placeholder={field.placeholder}
            value={field.formatters?.display ? field.formatters.display(value) : (value || '')}
            onChange={(e) => handleChange(e.target.value)}
            disabled={field.disabled}
            readOnly={field.readonly}
            autoComplete={field.autocomplete}
            className={hasErrors ? 'border-red-500' : ''}
          />
        );
    }
  };

  return (
    <div className={cn('space-y-2', className)}>
      {field.type !== 'checkbox' && (
        <Label htmlFor={field.name} className="flex items-center space-x-2">
          <span>{field.label}</span>
          {field.validation?.some(rule => rule.type === 'required') && (
            <span className="text-red-500 text-xs">*</span>
          )}
        </Label>
      )}
      
      {renderField()}
      
      {field.description && (
        <p className="text-sm text-gray-600">{field.description}</p>
      )}
      
      {fieldErrors.map((error, index) => (
        <div
          key={index}
          className={cn(
            'flex items-center space-x-2 text-sm p-2 rounded',
            error.type === 'error' ? 'text-red-700 bg-red-50' : 'text-yellow-700 bg-yellow-50'
          )}
        >
          {error.type === 'error' ? (
            <AlertTriangle className="h-4 w-4" />
          ) : (
            <Info className="h-4 w-4" />
          )}
          <span>{error.message}</span>
        </div>
      ))}
    </div>
  );
}

// Form section component for grouping fields
interface FormSectionProps {
  title?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function FormSection({ 
  title, 
  description, 
  children, 
  className,
  collapsible = false,
  defaultCollapsed = false 
}: FormSectionProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  return (
    <Card className={cn('mb-6', className)}>
      {(title || description) && (
        <CardHeader 
          className={collapsible ? 'cursor-pointer' : ''}
          onClick={() => collapsible && setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center justify-between">
            <div>
              {title && <CardTitle>{title}</CardTitle>}
              {description && <CardDescription>{description}</CardDescription>}
            </div>
            {collapsible && (
              <Button variant="ghost" size="sm">
                {isCollapsed ? 'Expand' : 'Collapse'}
              </Button>
            )}
          </div>
        </CardHeader>
      )}
      
      {(!collapsible || !isCollapsed) && (
        <CardContent className="space-y-4">
          {children}
        </CardContent>
      )}
    </Card>
  );
}

// Form actions component
interface FormActionsProps {
  className?: string;
  showReset?: boolean;
  showAutoSave?: boolean;
  submitLabel?: string;
  resetLabel?: string;
  customActions?: React.ReactNode;
}

export function FormActions({ 
  className,
  showReset = true,
  showAutoSave = true,
  submitLabel = 'Submit',
  resetLabel = 'Reset',
  customActions 
}: FormActionsProps) {
  const { 
    submitForm, 
    resetForm, 
    isSubmitting, 
    isDirty, 
    autoSaveEnabled, 
    lastSaved, 
    errors 
  } = useAdvancedForm();

  const hasErrors = errors.filter(e => e.type === 'error').length > 0;

  return (
    <div className={cn('flex items-center justify-between pt-6 border-t', className)}>
      <div className="flex items-center space-x-4">
        {showAutoSave && autoSaveEnabled && (
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Zap className="h-4 w-4" />
            <span>Auto-save enabled</span>
            {lastSaved && (
              <span>• Last saved: {lastSaved.toLocaleTimeString()}</span>
            )}
          </div>
        )}
        
        {isDirty && !autoSaveEnabled && (
          <Badge variant="outline" className="text-orange-600 border-orange-600">
            Unsaved changes
          </Badge>
        )}
        
        {customActions}
      </div>
      
      <div className="flex space-x-2">
        {showReset && (
          <Button
            type="button"
            variant="outline"
            onClick={resetForm}
            disabled={isSubmitting || !isDirty}
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            {resetLabel}
          </Button>
        )}
        
        <Button
          type="submit"
          onClick={submitForm}
          disabled={isSubmitting || hasErrors}
        >
          {isSubmitting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Submitting...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              {submitLabel}
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

// Form validation summary
export function FormValidationSummary({ className }: { className?: string }) {
  const { errors } = useAdvancedForm();
  
  const errorMessages = errors.filter(e => e.type === 'error');
  const warningMessages = errors.filter(e => e.type === 'warning');

  if (errorMessages.length === 0 && warningMessages.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-2 mb-6', className)}>
      {errorMessages.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Please fix the following errors:
            <ul className="list-disc list-inside mt-2">
              {errorMessages.map((error, index) => (
                <li key={index}>{error.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      
      {warningMessages.length > 0 && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            Please review the following warnings:
            <ul className="list-disc list-inside mt-2">
              {warningMessages.map((warning, index) => (
                <li key={index}>{warning.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

// Export all form components
export const AdvancedFormComponents = {
  AdvancedForm,
  FormField,
  FormSection,
  FormActions,
  FormValidationSummary,
  useAdvancedForm
};