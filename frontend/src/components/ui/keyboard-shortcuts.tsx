'use client';

import React, { createContext, useContext, useEffect, useRef, useState, KeyboardEvent } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { 
  Keyboard,
  Command,
  Search,
  Play,
  Save,
  Copy,
  Undo,
  Redo,
  HelpCircle,
  X
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface KeyboardShortcut {
  id: string;
  keys: string[]; // e.g., ['ctrl', 'k'] or ['cmd', 'k']
  description: string;
  category: string;
  action: () => void;
  global?: boolean; // whether it works globally or only in specific contexts
  disabled?: boolean;
  context?: string; // specific context where this shortcut applies
}

interface KeyboardShortcutContextType {
  shortcuts: KeyboardShortcut[];
  registerShortcut: (shortcut: KeyboardShortcut) => void;
  unregisterShortcut: (id: string) => void;
  enableShortcut: (id: string) => void;
  disableShortcut: (id: string) => void;
  showHelp: () => void;
  hideHelp: () => void;
  isHelpVisible: boolean;
}

const KeyboardShortcutContext = createContext<KeyboardShortcutContextType | undefined>(undefined);

export const useKeyboardShortcuts = () => {
  const context = useContext(KeyboardShortcutContext);
  if (!context) {
    throw new Error('useKeyboardShortcuts must be used within a KeyboardShortcutProvider');
  }
  return context;
};

interface KeyboardShortcutProviderProps {
  children: React.ReactNode;
  shortcuts?: KeyboardShortcut[];
}

export function KeyboardShortcutProvider({ children, shortcuts: initialShortcuts = [] }: KeyboardShortcutProviderProps) {
  const [shortcuts, setShortcuts] = useState<KeyboardShortcut[]>(initialShortcuts);
  const [isHelpVisible, setIsHelpVisible] = useState(false);
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  const pressedKeysRef = useRef<Set<string>>(new Set());

  // Update ref whenever state changes
  useEffect(() => {
    pressedKeysRef.current = pressedKeys;
  }, [pressedKeys]);

  // Normalize key names for cross-platform compatibility
  const normalizeKey = (key: string): string => {
    const keyMap: Record<string, string> = {
      'Control': 'ctrl',
      'Meta': 'cmd',
      'Alt': 'alt',
      'Shift': 'shift',
      'ArrowUp': 'up',
      'ArrowDown': 'down',
      'ArrowLeft': 'left',
      'ArrowRight': 'right',
      'Escape': 'esc',
      'Enter': 'enter',
      ' ': 'space'
    };
    
    return keyMap[key] || key.toLowerCase();
  };

  // Check if pressed keys match a shortcut
  const matchesShortcut = (shortcut: KeyboardShortcut, pressed: Set<string>): boolean => {
    if (shortcut.keys.length !== pressed.size) return false;
    
    return shortcut.keys.every(key => {
      const normalizedKey = normalizeKey(key);
      return pressed.has(normalizedKey);
    });
  };

  // Handle keydown events
  const handleKeyDown = (event: KeyboardEvent<Document>) => {
    const normalizedKey = normalizeKey(event.key);
    
    // Add key to pressed set
    setPressedKeys(prev => new Set(prev).add(normalizedKey));
    
    // Check for matching shortcuts
    const currentPressed = new Set(pressedKeysRef.current);
    currentPressed.add(normalizedKey);
    
    for (const shortcut of shortcuts) {
      if (shortcut.disabled) continue;
      
      if (matchesShortcut(shortcut, currentPressed)) {
        event.preventDefault();
        event.stopPropagation();
        shortcut.action();
        break;
      }
    }
  };

  // Handle keyup events
  const handleKeyUp = (event: KeyboardEvent<Document>) => {
    const normalizedKey = normalizeKey(event.key);
    setPressedKeys(prev => {
      const newSet = new Set(prev);
      newSet.delete(normalizedKey);
      return newSet;
    });
  };

  // Set up global event listeners
  useEffect(() => {
    const handleGlobalKeyDown = (event: globalThis.KeyboardEvent) => {
      handleKeyDown(event as any);
    };
    
    const handleGlobalKeyUp = (event: globalThis.KeyboardEvent) => {
      handleKeyUp(event as any);
    };
    
    const handleBlur = () => {
      setPressedKeys(new Set());
    };

    document.addEventListener('keydown', handleGlobalKeyDown);
    document.addEventListener('keyup', handleGlobalKeyUp);
    window.addEventListener('blur', handleBlur);

    return () => {
      document.removeEventListener('keydown', handleGlobalKeyDown);
      document.removeEventListener('keyup', handleGlobalKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, [shortcuts]);

  const registerShortcut = (shortcut: KeyboardShortcut) => {
    setShortcuts(prev => [...prev.filter(s => s.id !== shortcut.id), shortcut]);
  };

  const unregisterShortcut = (id: string) => {
    setShortcuts(prev => prev.filter(s => s.id !== id));
  };

  const enableShortcut = (id: string) => {
    setShortcuts(prev => prev.map(s => s.id === id ? { ...s, disabled: false } : s));
  };

  const disableShortcut = (id: string) => {
    setShortcuts(prev => prev.map(s => s.id === id ? { ...s, disabled: true } : s));
  };

  const showHelp = () => setIsHelpVisible(true);
  const hideHelp = () => setIsHelpVisible(false);

  // Default shortcuts
  useEffect(() => {
    const defaultShortcuts: KeyboardShortcut[] = [
      {
        id: 'help',
        keys: ['?'],
        description: 'Show keyboard shortcuts',
        category: 'General',
        action: showHelp,
        global: true
      },
      {
        id: 'close-help',
        keys: ['esc'],
        description: 'Close help dialog',
        category: 'General',
        action: hideHelp,
        global: true
      },
      {
        id: 'command-palette',
        keys: ['ctrl', 'k'],
        description: 'Open command palette',
        category: 'Navigation',
        action: () => {
          // Command palette implementation would go here
          console.log('Command palette opened');
        },
        global: true
      }
    ];

    defaultShortcuts.forEach(registerShortcut);
  }, []);

  const contextValue: KeyboardShortcutContextType = {
    shortcuts,
    registerShortcut,
    unregisterShortcut,
    enableShortcut,
    disableShortcut,
    showHelp,
    hideHelp,
    isHelpVisible
  };

  return (
    <KeyboardShortcutContext.Provider value={contextValue}>
      {children}
      <KeyboardShortcutHelp />
    </KeyboardShortcutContext.Provider>
  );
}

// Individual shortcut component for specific contexts
interface ShortcutProps {
  keys: string[];
  description: string;
  action: () => void;
  disabled?: boolean;
  context?: string;
  category?: string;
}

export function Shortcut({ 
  keys, 
  description, 
  action, 
  disabled = false, 
  context, 
  category = 'Custom' 
}: ShortcutProps) {
  const { registerShortcut, unregisterShortcut } = useKeyboardShortcuts();
  const shortcutId = useRef(`shortcut-${Date.now()}-${Math.random()}`);

  useEffect(() => {
    const shortcut: KeyboardShortcut = {
      id: shortcutId.current,
      keys,
      description,
      category,
      action,
      disabled,
      context
    };
    
    registerShortcut(shortcut);
    
    return () => {
      unregisterShortcut(shortcutId.current);
    };
  }, [keys, description, action, disabled, context, category, registerShortcut, unregisterShortcut]);

  return null; // This component doesn't render anything
}

// Key display component
interface KeyDisplayProps {
  keys: string[];
  className?: string;
}

export function KeyDisplay({ keys, className }: KeyDisplayProps) {
  const formatKey = (key: string): string => {
    const keyMap: Record<string, string> = {
      'ctrl': '⌃',
      'cmd': '⌘',
      'alt': '⌥',
      'shift': '⇧',
      'up': '↑',
      'down': '↓',
      'left': '←',
      'right': '→',
      'esc': 'Esc',
      'enter': '↵',
      'space': 'Space',
      'tab': '⇥'
    };
    
    return keyMap[key.toLowerCase()] || key.toUpperCase();
  };

  return (
    <div className={cn('flex items-center space-x-1', className)}>
      {keys.map((key, index) => (
        <React.Fragment key={key}>
          <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded">
            {formatKey(key)}
          </kbd>
          {index < keys.length - 1 && (
            <span className="text-gray-400 text-xs">+</span>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// Help dialog component
function KeyboardShortcutHelp() {
  const { shortcuts, isHelpVisible, hideHelp } = useKeyboardShortcuts();
  const [searchTerm, setSearchTerm] = useState('');

  // Group shortcuts by category
  const groupedShortcuts = shortcuts.reduce((groups, shortcut) => {
    if (!groups[shortcut.category]) {
      groups[shortcut.category] = [];
    }
    groups[shortcut.category].push(shortcut);
    return groups;
  }, {} as Record<string, KeyboardShortcut[]>);

  // Filter shortcuts based on search term
  const filteredShortcuts = Object.entries(groupedShortcuts).reduce((result, [category, categoryShortcuts]) => {
    const filtered = categoryShortcuts.filter(
      shortcut => 
        shortcut.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        shortcut.keys.some(key => key.toLowerCase().includes(searchTerm.toLowerCase()))
    );
    
    if (filtered.length > 0) {
      result[category] = filtered;
    }
    
    return result;
  }, {} as Record<string, KeyboardShortcut[]>);

  return (
    <Dialog open={isHelpVisible} onOpenChange={(open) => !open && hideHelp()}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Keyboard className="h-5 w-5" />
            <span>Keyboard Shortcuts</span>
          </DialogTitle>
          <DialogDescription>
            Boost your productivity with these keyboard shortcuts
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search shortcuts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          
          {/* Shortcuts by category */}
          {Object.entries(filteredShortcuts).map(([category, categoryShortcuts]) => (
            <div key={category}>
              <h3 className="text-lg font-semibold mb-3 text-gray-900">{category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {categoryShortcuts.map(shortcut => (
                  <div
                    key={shortcut.id}
                    className={cn(
                      'flex items-center justify-between p-3 rounded-lg border',
                      shortcut.disabled ? 'opacity-50 bg-gray-50' : 'bg-white hover:bg-gray-50'
                    )}
                  >
                    <span className="text-sm font-medium">{shortcut.description}</span>
                    <KeyDisplay keys={shortcut.keys} />
                  </div>
                ))}
              </div>
            </div>
          ))}
          
          {Object.keys(filteredShortcuts).length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <HelpCircle className="mx-auto h-12 w-12 mb-4" />
              <h3 className="text-lg font-medium mb-2">No shortcuts found</h3>
              <p>Try adjusting your search term</p>
            </div>
          )}
        </div>
        
        <div className="flex justify-end pt-4 border-t">
          <Button onClick={hideHelp}>
            <X className="mr-2 h-4 w-4" />
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Quick shortcut display component for UI elements
interface QuickShortcutProps {
  keys: string[];
  className?: string;
  size?: 'sm' | 'md';
}

export function QuickShortcut({ keys, className, size = 'sm' }: QuickShortcutProps) {
  return (
    <div className={cn(
      'flex items-center space-x-0.5 opacity-60',
      size === 'sm' ? 'text-xs' : 'text-sm',
      className
    )}>
      {keys.map((key, index) => (
        <React.Fragment key={key}>
          <kbd className={cn(
            'font-mono font-semibold bg-gray-100 border border-gray-200 rounded px-1',
            size === 'sm' ? 'text-xs py-0.5' : 'text-sm py-1'
          )}>
            {key.toUpperCase()}
          </kbd>
          {index < keys.length - 1 && (
            <span className="text-gray-400">+</span>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// Common shortcuts hook
export const useCommonShortcuts = () => {
  const { registerShortcut } = useKeyboardShortcuts();

  const registerSaveShortcut = (onSave: () => void) => {
    registerShortcut({
      id: 'save',
      keys: ['ctrl', 's'],
      description: 'Save current document',
      category: 'File',
      action: onSave
    });
  };

  const registerUndoRedoShortcuts = (onUndo: () => void, onRedo: () => void) => {
    registerShortcut({
      id: 'undo',
      keys: ['ctrl', 'z'],
      description: 'Undo last action',
      category: 'Edit',
      action: onUndo
    });
    
    registerShortcut({
      id: 'redo',
      keys: ['ctrl', 'y'],
      description: 'Redo last action',
      category: 'Edit',
      action: onRedo
    });
  };

  const registerCopyShortcut = (onCopy: () => void) => {
    registerShortcut({
      id: 'copy',
      keys: ['ctrl', 'c'],
      description: 'Copy selection',
      category: 'Edit',
      action: onCopy
    });
  };

  return {
    registerSaveShortcut,
    registerUndoRedoShortcuts,
    registerCopyShortcut
  };
};

// Export components
export const KeyboardShortcutComponents = {
  KeyboardShortcutProvider,
  Shortcut,
  KeyDisplay,
  QuickShortcut,
  useKeyboardShortcuts,
  useCommonShortcuts
};