'use client';

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Eye,
  EyeOff,
  Volume2,
  VolumeX,
  Monitor,
  Sun,
  Moon,
  Type,
  Contrast,
  MousePointer,
  Keyboard,
  AlertTriangle,
  CheckCircle,
  Settings,
  Accessibility,
  ZoomIn,
  ZoomOut,
  SkipForward
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccessibilitySettings {
  highContrast: boolean;
  largeText: boolean;
  reducedMotion: boolean;
  screenReader: boolean;
  keyboardNavigation: boolean;
  focusIndicator: boolean;
  alternativeText: boolean;
  colorBlindSupport: boolean;
  fontSize: number; // percentage
  zoomLevel: number; // percentage
  darkMode: boolean;
  soundEnabled: boolean;
  autoAnnounce: boolean;
}

interface AccessibilityContextType {
  settings: AccessibilitySettings;
  updateSetting: (key: keyof AccessibilitySettings, value: any) => void;
  announceToScreenReader: (message: string) => void;
  checkCompliance: () => AccessibilityCheck[];
  focusElement: (selector: string) => void;
  skipToContent: () => void;
}

interface AccessibilityCheck {
  id: string;
  category: 'wcag-a' | 'wcag-aa' | 'wcag-aaa' | 'best-practice';
  level: 'error' | 'warning' | 'info';
  message: string;
  element?: string;
  fix?: string;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
  initialSettings?: Partial<AccessibilitySettings>;
}

export function AccessibilityProvider({ children, initialSettings = {} }: AccessibilityProviderProps) {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    highContrast: false,
    largeText: false,
    reducedMotion: false,
    screenReader: false,
    keyboardNavigation: true,
    focusIndicator: true,
    alternativeText: true,
    colorBlindSupport: false,
    fontSize: 100,
    zoomLevel: 100,
    darkMode: false,
    soundEnabled: true,
    autoAnnounce: false,
    ...initialSettings
  });

  const screenReaderRef = useRef<HTMLDivElement>(null);

  // Apply accessibility settings to document
  useEffect(() => {
    const root = document.documentElement;
    
    // High contrast mode
    if (settings.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    
    // Large text
    if (settings.largeText) {
      root.classList.add('large-text');
    } else {
      root.classList.remove('large-text');
    }
    
    // Reduced motion
    if (settings.reducedMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }
    
    // Font size
    root.style.fontSize = `${settings.fontSize}%`;
    
    // Zoom level
    root.style.zoom = `${settings.zoomLevel}%`;
    
    // Dark mode
    if (settings.darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // Focus indicator enhancement
    if (settings.focusIndicator) {
      root.classList.add('enhanced-focus');
    } else {
      root.classList.remove('enhanced-focus');
    }
  }, [settings]);

  // Add CSS for accessibility features
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .high-contrast {
        filter: contrast(150%) brightness(110%);
      }
      
      .high-contrast * {
        text-shadow: none !important;
        box-shadow: none !important;
      }
      
      .large-text {
        font-size: 120% !important;
      }
      
      .large-text * {
        font-size: inherit !important;
      }
      
      .reduce-motion * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        transform: none !important;
      }
      
      .enhanced-focus *:focus {
        outline: 3px solid #0066cc !important;
        outline-offset: 2px !important;
        z-index: 999 !important;
        position: relative !important;
      }
      
      .color-blind-support {
        --red: #cc0000;
        --green: #00cc00;
        --blue: #0066cc;
        --yellow: #cccc00;
        --orange: #cc6600;
        --purple: #6600cc;
      }
      
      .screen-reader-only {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
      }
      
      .skip-link {
        position: absolute;
        top: -40px;
        left: 6px;
        background: #000;
        color: #fff;
        padding: 8px;
        text-decoration: none;
        z-index: 1000;
        border-radius: 4px;
      }
      
      .skip-link:focus {
        top: 6px;
      }
    `;
    
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const updateSetting = (key: keyof AccessibilitySettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    
    // Save to localStorage
    localStorage.setItem('accessibility-settings', JSON.stringify({ ...settings, [key]: value }));
  };

  const announceToScreenReader = (message: string) => {
    if (!settings.screenReader && !settings.autoAnnounce) return;
    
    if (screenReaderRef.current) {
      screenReaderRef.current.textContent = message;
      
      // Clear after a delay to allow screen readers to announce
      setTimeout(() => {
        if (screenReaderRef.current) {
          screenReaderRef.current.textContent = '';
        }
      }, 1000);
    }
  };

  const checkCompliance = (): AccessibilityCheck[] => {
    const checks: AccessibilityCheck[] = [];
    
    // Check for images without alt text
    const imagesWithoutAlt = document.querySelectorAll('img:not([alt])');
    if (imagesWithoutAlt.length > 0) {
      checks.push({
        id: 'missing-alt-text',
        category: 'wcag-a',
        level: 'error',
        message: `${imagesWithoutAlt.length} images missing alt text`,
        element: 'img',
        fix: 'Add alt attribute to all images'
      });
    }
    
    // Check for buttons without accessible names
    const buttonsWithoutNames = document.querySelectorAll('button:not([aria-label]):not([aria-labelledby])');
    const buttonsWithEmptyContent = Array.from(buttonsWithoutNames).filter(btn => !btn.textContent?.trim());
    if (buttonsWithEmptyContent.length > 0) {
      checks.push({
        id: 'buttons-no-accessible-name',
        category: 'wcag-a',
        level: 'error',
        message: `${buttonsWithEmptyContent.length} buttons without accessible names`,
        element: 'button',
        fix: 'Add aria-label or ensure buttons have text content'
      });
    }
    
    // Check for form inputs without labels
    const inputsWithoutLabels = document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])');
    const unlabeledInputs = Array.from(inputsWithoutLabels).filter(input => {
      const id = input.getAttribute('id');
      return !id || !document.querySelector(`label[for="${id}"]`);
    });
    if (unlabeledInputs.length > 0) {
      checks.push({
        id: 'inputs-without-labels',
        category: 'wcag-a',
        level: 'error',
        message: `${unlabeledInputs.length} form inputs without labels`,
        element: 'input',
        fix: 'Associate labels with form inputs using for/id attributes'
      });
    }
    
    // Check for headings hierarchy
    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
    const headingLevels = headings.map(h => parseInt(h.tagName[1]));
    let hasSkippedLevel = false;
    for (let i = 1; i < headingLevels.length; i++) {
      if (headingLevels[i] - headingLevels[i-1] > 1) {
        hasSkippedLevel = true;
        break;
      }
    }
    if (hasSkippedLevel) {
      checks.push({
        id: 'heading-hierarchy',
        category: 'wcag-aa',
        level: 'warning',
        message: 'Heading hierarchy has skipped levels',
        element: 'h1-h6',
        fix: 'Ensure heading levels follow sequential order'
      });
    }
    
    // Check for contrast ratios (simplified check)
    if (!settings.highContrast) {
      checks.push({
        id: 'color-contrast',
        category: 'wcag-aa',
        level: 'info',
        message: 'Consider enabling high contrast mode for better visibility',
        fix: 'Enable high contrast in accessibility settings'
      });
    }
    
    return checks;
  };

  const focusElement = (selector: string) => {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      element.focus();
      announceToScreenReader(`Focused on ${element.textContent || selector}`);
    }
  };

  const skipToContent = () => {
    const mainContent = document.querySelector('main, [role="main"], #main-content') as HTMLElement;
    if (mainContent) {
      mainContent.focus();
      announceToScreenReader('Skipped to main content');
    }
  };

  const contextValue: AccessibilityContextType = {
    settings,
    updateSetting,
    announceToScreenReader,
    checkCompliance,
    focusElement,
    skipToContent
  };

  return (
    <AccessibilityContext.Provider value={contextValue}>
      {/* Skip to content link */}
      <a href="#main-content" className="skip-link" onClick={(e) => {
        e.preventDefault();
        skipToContent();
      }}>
        Skip to main content
      </a>
      
      {/* Screen reader announcements */}
      <div
        ref={screenReaderRef}
        className="screen-reader-only"
        aria-live="polite"
        aria-atomic="true"
      />
      
      {children}
    </AccessibilityContext.Provider>
  );
}

// Accessibility settings panel
interface AccessibilityPanelProps {
  className?: string;
}

export function AccessibilityPanel({ className }: AccessibilityPanelProps) {
  const { settings, updateSetting, checkCompliance } = useAccessibility();
  const [complianceChecks, setComplianceChecks] = useState<AccessibilityCheck[]>([]);
  const [showChecks, setShowChecks] = useState(false);

  const runComplianceCheck = () => {
    const checks = checkCompliance();
    setComplianceChecks(checks);
    setShowChecks(true);
  };

  return (
    <Card className={cn('w-full max-w-2xl', className)}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Accessibility className="h-5 w-5" />
          <span>Accessibility Settings</span>
        </CardTitle>
        <CardDescription>
          Customize accessibility features for better usability
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Visual Settings */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Visual</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Contrast className="h-4 w-4" />
                <span>High Contrast</span>
              </div>
              <Switch
                checked={settings.highContrast}
                onCheckedChange={(checked) => updateSetting('highContrast', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Type className="h-4 w-4" />
                <span>Large Text</span>
              </div>
              <Switch
                checked={settings.largeText}
                onCheckedChange={(checked) => updateSetting('largeText', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {settings.darkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                <span>Dark Mode</span>
              </div>
              <Switch
                checked={settings.darkMode}
                onCheckedChange={(checked) => updateSetting('darkMode', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Eye className="h-4 w-4" />
                <span>Color Blind Support</span>
              </div>
              <Switch
                checked={settings.colorBlindSupport}
                onCheckedChange={(checked) => updateSetting('colorBlindSupport', checked)}
              />
            </div>
          </div>
          
          {/* Font Size Control */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Font Size: {settings.fontSize}%</label>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => updateSetting('fontSize', Math.max(50, settings.fontSize - 10))}
              >
                <Type className="h-4 w-4" />
                <span className="ml-1">-</span>
              </Button>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${(settings.fontSize - 50) / 150 * 100}%` }}
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => updateSetting('fontSize', Math.min(200, settings.fontSize + 10))}
              >
                <Type className="h-4 w-4" />
                <span className="ml-1">+</span>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Motion Settings */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Motion</h3>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Monitor className="h-4 w-4" />
              <span>Reduced Motion</span>
            </div>
            <Switch
              checked={settings.reducedMotion}
              onCheckedChange={(checked) => updateSetting('reducedMotion', checked)}
            />
          </div>
        </div>
        
        {/* Navigation Settings */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Navigation</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Keyboard className="h-4 w-4" />
                <span>Keyboard Navigation</span>
              </div>
              <Switch
                checked={settings.keyboardNavigation}
                onCheckedChange={(checked) => updateSetting('keyboardNavigation', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <MousePointer className="h-4 w-4" />
                <span>Enhanced Focus</span>
              </div>
              <Switch
                checked={settings.focusIndicator}
                onCheckedChange={(checked) => updateSetting('focusIndicator', checked)}
              />
            </div>
          </div>
        </div>
        
        {/* Audio Settings */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Audio</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {settings.soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                <span>Sound Effects</span>
              </div>
              <Switch
                checked={settings.soundEnabled}
                onCheckedChange={(checked) => updateSetting('soundEnabled', checked)}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Volume2 className="h-4 w-4" />
                <span>Auto Announce</span>
              </div>
              <Switch
                checked={settings.autoAnnounce}
                onCheckedChange={(checked) => updateSetting('autoAnnounce', checked)}
              />
            </div>
          </div>
        </div>
        
        {/* Compliance Check */}
        <div className="space-y-4 pt-6 border-t">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Compliance Check</h3>
            <Button onClick={runComplianceCheck} variant="outline">
              <CheckCircle className="mr-2 h-4 w-4" />
              Run Check
            </Button>
          </div>
          
          {showChecks && (
            <div className="space-y-2">
              {complianceChecks.length === 0 ? (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    Great! No accessibility issues detected.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-2">
                  {complianceChecks.map(check => (
                    <Alert key={check.id} variant={check.level === 'error' ? 'destructive' : 'default'}>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="flex items-center justify-between">
                          <div>
                            <strong>{check.message}</strong>
                            <p className="text-sm mt-1">{check.fix}</p>
                          </div>
                          <Badge variant="outline">
                            {check.category.toUpperCase()}
                          </Badge>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Screen reader only text component
interface ScreenReaderOnlyProps {
  children: React.ReactNode;
}

export function ScreenReaderOnly({ children }: ScreenReaderOnlyProps) {
  return (
    <span className="screen-reader-only">
      {children}
    </span>
  );
}

// Announcement component for screen readers
interface AnnouncementProps {
  message: string;
  priority?: 'polite' | 'assertive';
}

export function Announcement({ message, priority = 'polite' }: AnnouncementProps) {
  const { announceToScreenReader } = useAccessibility();
  
  useEffect(() => {
    announceToScreenReader(message);
  }, [message, announceToScreenReader]);

  return (
    <div
      aria-live={priority}
      aria-atomic="true"
      className="screen-reader-only"
    >
      {message}
    </div>
  );
}

// Focus trap component
interface FocusTrapProps {
  children: React.ReactNode;
  active?: boolean;
}

export function FocusTrap({ children, active = true }: FocusTrapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const firstFocusableRef = useRef<HTMLElement>();
  const lastFocusableRef = useRef<HTMLElement>();

  useEffect(() => {
    if (!active || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>;

    if (focusableElements.length === 0) return;

    firstFocusableRef.current = focusableElements[0];
    lastFocusableRef.current = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusableRef.current) {
          e.preventDefault();
          lastFocusableRef.current?.focus();
        }
      } else {
        if (document.activeElement === lastFocusableRef.current) {
          e.preventDefault();
          firstFocusableRef.current?.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    firstFocusableRef.current?.focus();

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }, [active]);

  return (
    <div ref={containerRef}>
      {children}
    </div>
  );
}

// Export all accessibility components
export const AccessibilityComponents = {
  AccessibilityProvider,
  AccessibilityPanel,
  ScreenReaderOnly,
  Announcement,
  FocusTrap,
  useAccessibility
};