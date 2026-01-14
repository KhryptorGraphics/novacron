"""
Core Enhancement Framework
=========================

Base classes and registry for MLE-Star enhancements.
Provides plugin architecture for extensible ML capabilities.
"""

import abc
import logging
import inspect
from typing import Dict, Any, Optional, List, Type, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for enhancements"""
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    priority: int = 100  # Lower numbers = higher priority
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseEnhancement(abc.ABC):
    """Base class for all MLE-Star enhancements"""
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or self._default_config()
        self._initialized = False
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abc.abstractmethod
    def _default_config(self) -> EnhancementConfig:
        """Return default configuration for this enhancement"""
        pass
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the enhancement. Return True if successful."""
        pass
    
    @abc.abstractmethod
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhancement to MLE-Star workflow"""
        pass
    
    def validate_dependencies(self, available_enhancements: List[str]) -> bool:
        """Validate that required dependencies are available"""
        missing = set(self.config.dependencies) - set(available_enhancements)
        if missing:
            self._logger.warning(f"Missing dependencies for {self.config.name}: {missing}")
            return False
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return enhancement metadata"""
        return {
            'name': self.config.name,
            'version': self.config.version,
            'enabled': self.config.enabled,
            'priority': self.config.priority,
            'dependencies': self.config.dependencies,
            'class': self.__class__.__name__,
            'module': self.__class__.__module__
        }


class EnhancementRegistry:
    """Registry for managing MLE-Star enhancements"""
    
    def __init__(self):
        self._enhancements: Dict[str, BaseEnhancement] = {}
        self._config_path: Optional[Path] = None
        self._logger = logging.getLogger(__name__)
    
    def register(self, name: str, enhancement: BaseEnhancement) -> None:
        """Register an enhancement"""
        if not isinstance(enhancement, BaseEnhancement):
            raise ValueError(f"Enhancement must inherit from BaseEnhancement")
        
        self._enhancements[name] = enhancement
        self._logger.info(f"Registered enhancement: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister an enhancement"""
        if name in self._enhancements:
            del self._enhancements[name]
            self._logger.info(f"Unregistered enhancement: {name}")
    
    def get_enhancement(self, name: str) -> Optional[BaseEnhancement]:
        """Get enhancement by name"""
        return self._enhancements.get(name)
    
    def list_enhancements(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """List all registered enhancements"""
        enhancements = []
        for name, enhancement in self._enhancements.items():
            if not enabled_only or enhancement.config.enabled:
                metadata = enhancement.get_metadata()
                metadata['registered_name'] = name
                enhancements.append(metadata)
        
        # Sort by priority
        return sorted(enhancements, key=lambda x: x['priority'])
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all enabled enhancements"""
        results = {}
        
        # Sort by priority and dependencies
        sorted_enhancements = self._resolve_dependency_order()
        
        for name in sorted_enhancements:
            enhancement = self._enhancements[name]
            if not enhancement.config.enabled:
                continue
            
            try:
                # Check dependencies
                available = [n for n, e in self._enhancements.items() 
                           if e.config.enabled and e._initialized]
                
                if not enhancement.validate_dependencies(available):
                    self._logger.error(f"Failed to initialize {name}: missing dependencies")
                    results[name] = False
                    continue
                
                # Initialize
                success = enhancement.initialize()
                enhancement._initialized = success
                results[name] = success
                
                if success:
                    self._logger.info(f"Successfully initialized enhancement: {name}")
                else:
                    self._logger.error(f"Failed to initialize enhancement: {name}")
                    
            except Exception as e:
                self._logger.error(f"Error initializing {name}: {e}")
                results[name] = False
        
        return results
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all enabled enhancements to workflow"""
        enhanced_workflow = workflow.copy()
        
        # Apply enhancements in priority order
        sorted_enhancements = self._resolve_dependency_order()
        
        for name in sorted_enhancements:
            enhancement = self._enhancements[name]
            if enhancement.config.enabled and enhancement._initialized:
                try:
                    enhanced_workflow = enhancement.enhance_workflow(enhanced_workflow)
                    self._logger.debug(f"Applied enhancement: {name}")
                except Exception as e:
                    self._logger.error(f"Error applying enhancement {name}: {e}")
        
        return enhanced_workflow
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load enhancement configuration from file"""
        config_path = Path(config_path)
        self._config_path = config_path
        
        if not config_path.exists():
            self._logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update enhancement configurations
            enhancements_config = config_data.get('enhancements', {})
            for name, config in enhancements_config.items():
                if name in self._enhancements:
                    # Update configuration
                    enhancement = self._enhancements[name]
                    for key, value in config.items():
                        if hasattr(enhancement.config, key):
                            setattr(enhancement.config, key, value)
                    
                    self._logger.info(f"Updated config for enhancement: {name}")
                    
        except Exception as e:
            self._logger.error(f"Error loading config: {e}")
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save current enhancement configuration to file"""
        if config_path is None and self._config_path is None:
            raise ValueError("No config path specified")
        
        config_path = Path(config_path or self._config_path)
        
        # Build configuration
        config_data = {
            'enhancements': {}
        }
        
        for name, enhancement in self._enhancements.items():
            config_data['enhancements'][name] = {
                'enabled': enhancement.config.enabled,
                'priority': enhancement.config.priority,
                'parameters': enhancement.config.parameters
            }
        
        # Save to file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        self._logger.info(f"Saved enhancement config to: {config_path}")
    
    def _resolve_dependency_order(self) -> List[str]:
        """Resolve enhancement initialization order based on dependencies"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            # Visit dependencies first
            enhancement = self._enhancements.get(name)
            if enhancement:
                for dep in enhancement.config.dependencies:
                    if dep in self._enhancements:
                        visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        # Visit all enhancements
        for name in self._enhancements:
            visit(name)
        
        return result


class EnhancementLoader:
    """Utility for automatically loading enhancements from modules"""
    
    @staticmethod
    def load_from_module(module_name: str, registry: EnhancementRegistry) -> int:
        """Load all enhancements from a module"""
        try:
            module = __import__(module_name, fromlist=[''])
            loaded = 0
            
            # Find all BaseEnhancement subclasses
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseEnhancement) and 
                    obj != BaseEnhancement):
                    
                    # Instantiate and register
                    enhancement = obj()
                    registry.register(name, enhancement)
                    loaded += 1
            
            return loaded
            
        except ImportError as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            return 0
    
    @staticmethod
    def discover_enhancements(package_name: str, registry: EnhancementRegistry) -> int:
        """Discover and load all enhancements from a package"""
        import pkgutil
        
        try:
            package = __import__(package_name, fromlist=[''])
            loaded = 0
            
            # Walk through all submodules
            for _, module_name, _ in pkgutil.walk_packages(
                package.__path__, 
                package.__name__ + "."
            ):
                count = EnhancementLoader.load_from_module(module_name, registry)
                loaded += count
            
            return loaded
            
        except ImportError as e:
            logger.error(f"Failed to discover enhancements in {package_name}: {e}")
            return 0