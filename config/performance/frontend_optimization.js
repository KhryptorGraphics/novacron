// Frontend Performance Optimization for NovaCron v10
// Implements lazy loading, code splitting, and advanced frontend optimizations

import { lazy, Suspense, memo, useMemo, useCallback, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';

/**
 * Advanced Code Splitting and Lazy Loading Manager
 * Implements intelligent component loading with performance monitoring
 */
class FrontendOptimizationManager {
  constructor(config = {}) {
    this.config = {
      chunkSize: 50000, // 50KB chunks
      preloadThreshold: 0.7, // Preload when 70% likely to be needed
      maxConcurrentLoads: 3,
      cacheSize: 100,
      performanceThreshold: 100, // 100ms
      ...config
    };

    this.loadingCache = new Map();
    this.componentCache = new Map();
    this.performanceMetrics = new Map();
    this.preloadQueue = [];
    this.loadingQueue = [];
    this.currentLoads = 0;

    this.initializeOptimizations();
  }

  initializeOptimizations() {
    // Initialize performance observer
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        this.analyzePerformanceEntries(list.getEntries());
      });
      
      this.performanceObserver.observe({ 
        entryTypes: ['navigation', 'resource', 'measure', 'paint'] 
      });
    }

    // Initialize intersection observer for viewport-based loading
    this.intersectionObserver = new IntersectionObserver(
      this.handleIntersection.bind(this),
      { 
        rootMargin: '50px',
        threshold: [0, 0.25, 0.5, 0.75, 1.0]
      }
    );

    // Initialize prefetch strategies
    this.initializePrefetchStrategies();
  }

  /**
   * Creates an optimized lazy-loaded component with intelligent preloading
   */
  createLazyComponent(importFn, options = {}) {
    const {
      fallback = <LoadingSpinner />,
      preload = false,
      chunkName = 'dynamic-chunk',
      priority = 'normal'
    } = options;

    // Create cached loader with error boundary
    const loader = this.createCachedLoader(importFn, chunkName);
    
    // Create lazy component with enhanced error handling
    const LazyComponent = lazy(() => 
      loader().catch(error => {
        console.error(`Failed to load component ${chunkName}:`, error);
        return { default: () => <ErrorFallback error={error} /> };
      })
    );

    // Preload if requested
    if (preload) {
      this.preloadComponent(loader, priority);
    }

    // Return wrapped component with performance monitoring
    return memo((props) => (
      <Suspense fallback={this.createIntelligentFallback(fallback, chunkName)}>
        <PerformanceWrapper componentName={chunkName}>
          <LazyComponent {...props} />
        </PerformanceWrapper>
      </Suspense>
    ));
  }

  /**
   * Intelligent component preloading based on user behavior
   */
  preloadComponent(loader, priority = 'normal') {
    if (this.loadingCache.has(loader)) {
      return this.loadingCache.get(loader);
    }

    const preloadPromise = new Promise((resolve, reject) => {
      const task = {
        loader,
        resolve,
        reject,
        priority,
        timestamp: performance.now()
      };

      if (priority === 'high' || this.currentLoads < this.config.maxConcurrentLoads) {
        this.executeLoad(task);
      } else {
        this.preloadQueue.push(task);
        this.sortPreloadQueue();
      }
    });

    this.loadingCache.set(loader, preloadPromise);
    return preloadPromise;
  }

  executeLoad(task) {
    this.currentLoads++;
    
    const startTime = performance.now();
    performance.mark(`load-start-${task.loader.name}`);

    task.loader()
      .then(module => {
        const loadTime = performance.now() - startTime;
        performance.mark(`load-end-${task.loader.name}`);
        performance.measure(
          `load-${task.loader.name}`,
          `load-start-${task.loader.name}`,
          `load-end-${task.loader.name}`
        );

        this.recordLoadMetrics(task.loader.name, loadTime, true);
        task.resolve(module);
      })
      .catch(error => {
        this.recordLoadMetrics(task.loader.name, performance.now() - startTime, false);
        task.reject(error);
      })
      .finally(() => {
        this.currentLoads--;
        this.processNextInQueue();
      });
  }

  processNextInQueue() {
    if (this.preloadQueue.length > 0 && this.currentLoads < this.config.maxConcurrentLoads) {
      const nextTask = this.preloadQueue.shift();
      this.executeLoad(nextTask);
    }
  }

  sortPreloadQueue() {
    this.preloadQueue.sort((a, b) => {
      // Sort by priority and age
      const priorityWeight = { high: 3, normal: 2, low: 1 };
      const aPriority = priorityWeight[a.priority] || 1;
      const bPriority = priorityWeight[b.priority] || 1;
      
      if (aPriority !== bPriority) {
        return bPriority - aPriority;
      }
      
      return b.timestamp - a.timestamp; // Newer first
    });
  }

  createCachedLoader(importFn, chunkName) {
    return () => {
      if (this.componentCache.has(chunkName)) {
        return Promise.resolve(this.componentCache.get(chunkName));
      }

      return importFn().then(module => {
        this.componentCache.set(chunkName, module);
        return module;
      });
    };
  }

  createIntelligentFallback(fallback, chunkName) {
    const metrics = this.performanceMetrics.get(chunkName);
    const estimatedLoadTime = metrics?.averageLoadTime || 1000;

    return (
      <div className="loading-container">
        {fallback}
        <ProgressBar estimatedTime={estimatedLoadTime} />
      </div>
    );
  }

  recordLoadMetrics(chunkName, loadTime, success) {
    const existing = this.performanceMetrics.get(chunkName) || {
      loadCount: 0,
      totalLoadTime: 0,
      averageLoadTime: 0,
      successRate: 0,
      failures: 0
    };

    existing.loadCount++;
    existing.totalLoadTime += loadTime;
    existing.averageLoadTime = existing.totalLoadTime / existing.loadCount;

    if (success) {
      existing.successRate = ((existing.loadCount - existing.failures) / existing.loadCount) * 100;
    } else {
      existing.failures++;
      existing.successRate = ((existing.loadCount - existing.failures) / existing.loadCount) * 100;
    }

    this.performanceMetrics.set(chunkName, existing);
  }

  initializePrefetchStrategies() {
    // Mouse hover prefetching
    this.setupHoverPrefetch();
    
    // Viewport-based prefetching
    this.setupViewportPrefetch();
    
    // User journey prediction
    this.setupJourneyPrediction();
    
    // Network-aware prefetching
    this.setupNetworkAwarePrefetch();
  }

  setupHoverPrefetch() {
    let hoverTimeout;
    
    document.addEventListener('mouseover', (event) => {
      const prefetchElement = event.target.closest('[data-prefetch]');
      if (!prefetchElement) return;

      clearTimeout(hoverTimeout);
      hoverTimeout = setTimeout(() => {
        const componentPath = prefetchElement.dataset.prefetch;
        this.prefetchByPath(componentPath, 'high');
      }, 150); // Delay to avoid unnecessary prefetching
    });

    document.addEventListener('mouseout', () => {
      clearTimeout(hoverTimeout);
    });
  }

  setupViewportPrefetch() {
    const prefetchElements = document.querySelectorAll('[data-viewport-prefetch]');
    prefetchElements.forEach(element => {
      this.intersectionObserver.observe(element);
    });
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
        const componentPath = entry.target.dataset.viewportPrefetch;
        this.prefetchByPath(componentPath, 'normal');
      }
    });
  }

  setupJourneyPrediction() {
    // Analyze user navigation patterns
    this.navigationHistory = JSON.parse(localStorage.getItem('nav-history') || '[]');
    
    window.addEventListener('beforeunload', () => {
      localStorage.setItem('nav-history', JSON.stringify(this.navigationHistory));
    });

    // Predict next likely components based on current route
    this.predictNextComponents();
  }

  predictNextComponents() {
    const currentPath = window.location.pathname;
    const predictions = this.analyzeNavigationPatterns(currentPath);
    
    predictions.forEach(({ path, probability }) => {
      if (probability > this.config.preloadThreshold) {
        this.prefetchByPath(path, 'low');
      }
    });
  }

  analyzeNavigationPatterns(currentPath) {
    const patterns = this.navigationHistory
      .filter((entry, index, arr) => 
        index < arr.length - 1 && arr[index].path === currentPath
      )
      .map((entry, index, arr) => arr[index + 1].path);

    const patternCounts = patterns.reduce((acc, path) => {
      acc[path] = (acc[path] || 0) + 1;
      return acc;
    }, {});

    const total = patterns.length;
    return Object.entries(patternCounts).map(([path, count]) => ({
      path,
      probability: count / total
    }));
  }

  setupNetworkAwarePrefetch() {
    if ('connection' in navigator) {
      const connection = navigator.connection;
      
      // Adjust prefetching based on network conditions
      const adjustPrefetchStrategy = () => {
        const effectiveType = connection.effectiveType;
        
        switch (effectiveType) {
          case 'slow-2g':
          case '2g':
            this.config.maxConcurrentLoads = 1;
            this.config.preloadThreshold = 0.9;
            break;
          case '3g':
            this.config.maxConcurrentLoads = 2;
            this.config.preloadThreshold = 0.8;
            break;
          case '4g':
          default:
            this.config.maxConcurrentLoads = 3;
            this.config.preloadThreshold = 0.7;
            break;
        }
      };

      connection.addEventListener('change', adjustPrefetchStrategy);
      adjustPrefetchStrategy();
    }
  }

  prefetchByPath(path, priority) {
    // Map paths to import functions - this would be configured based on your routing
    const routeImportMap = {
      '/dashboard': () => import('../pages/Dashboard'),
      '/users': () => import('../pages/Users'),
      '/settings': () => import('../pages/Settings'),
      // Add more routes as needed
    };

    const importFn = routeImportMap[path];
    if (importFn) {
      this.preloadComponent(() => importFn(), priority);
    }
  }

  analyzePerformanceEntries(entries) {
    entries.forEach(entry => {
      if (entry.entryType === 'measure' && entry.name.startsWith('load-')) {
        const chunkName = entry.name.replace('load-', '');
        
        // Record performance metrics
        if (entry.duration > this.config.performanceThreshold) {
          console.warn(`Slow component load detected: ${chunkName} took ${entry.duration}ms`);
        }
      }
    });
  }

  getPerformanceReport() {
    const report = {
      totalComponents: this.performanceMetrics.size,
      averageLoadTime: 0,
      cacheHitRate: 0,
      preloadEffectiveness: 0,
      components: {}
    };

    let totalLoadTime = 0;
    let totalLoads = 0;

    this.performanceMetrics.forEach((metrics, chunkName) => {
      totalLoadTime += metrics.totalLoadTime;
      totalLoads += metrics.loadCount;
      
      report.components[chunkName] = {
        ...metrics,
        score: this.calculatePerformanceScore(metrics)
      };
    });

    report.averageLoadTime = totalLoads > 0 ? totalLoadTime / totalLoads : 0;
    report.cacheHitRate = this.calculateCacheHitRate();
    
    return report;
  }

  calculatePerformanceScore(metrics) {
    const loadTimeScore = Math.max(0, 100 - (metrics.averageLoadTime / 10));
    const reliabilityScore = metrics.successRate;
    
    return (loadTimeScore * 0.6 + reliabilityScore * 0.4);
  }

  calculateCacheHitRate() {
    let hits = 0;
    let total = 0;

    this.performanceMetrics.forEach(metrics => {
      total += metrics.loadCount;
      // Assume cache hits for loads faster than threshold
      hits += metrics.loadCount * (metrics.averageLoadTime < 50 ? 1 : 0);
    });

    return total > 0 ? (hits / total) * 100 : 0;
  }
}

/**
 * Advanced Bundle Optimization Utilities
 */
class BundleOptimizer {
  constructor() {
    this.chunkAnalyzer = new ChunkAnalyzer();
    this.dependencyOptimizer = new DependencyOptimizer();
    this.treeShaker = new TreeShaker();
  }

  /**
   * Analyze bundle composition and suggest optimizations
   */
  analyzeBundles() {
    const analysis = {
      totalSize: 0,
      chunkSizes: new Map(),
      duplicateDependencies: [],
      unusedCode: [],
      optimizationOpportunities: []
    };

    // This would integrate with webpack-bundle-analyzer or similar
    // For demonstration, showing the structure
    
    if (window.__webpack_require__) {
      const chunks = window.__webpack_require__.cache;
      
      Object.keys(chunks).forEach(chunkId => {
        const chunk = chunks[chunkId];
        if (chunk && chunk.exports) {
          const size = this.estimateChunkSize(chunk);
          analysis.chunkSizes.set(chunkId, size);
          analysis.totalSize += size;
        }
      });
    }

    return analysis;
  }

  estimateChunkSize(chunk) {
    // Rough estimation based on serialized size
    try {
      return JSON.stringify(chunk.exports).length;
    } catch (e) {
      return 1000; // Default estimate
    }
  }

  /**
   * Dynamic import optimization
   */
  optimizeImports() {
    // Implement dynamic import analysis and optimization
    return {
      optimizedImports: [],
      savings: 0,
      recommendations: []
    };
  }
}

/**
 * Image Optimization and Lazy Loading
 */
class ImageOptimizer {
  constructor() {
    this.observer = new IntersectionObserver(this.handleImageIntersection.bind(this), {
      rootMargin: '50px 0px',
      threshold: 0.01
    });

    this.loadedImages = new Set();
    this.initializeImageOptimizations();
  }

  initializeImageOptimizations() {
    // Setup responsive image loading
    this.setupResponsiveImages();
    
    // Setup progressive image loading
    this.setupProgressiveLoading();
    
    // Setup WebP/AVIF support detection
    this.detectModernImageFormats();
  }

  lazyLoadImages(container = document) {
    const images = container.querySelectorAll('img[data-src], img[data-srcset]');
    images.forEach(img => this.observer.observe(img));
  }

  handleImageIntersection(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        this.loadImage(entry.target);
        this.observer.unobserve(entry.target);
      }
    });
  }

  loadImage(img) {
    return new Promise((resolve, reject) => {
      const imageLoader = new Image();
      
      imageLoader.onload = () => {
        // Apply loaded image
        if (img.dataset.srcset) {
          img.srcset = img.dataset.srcset;
        }
        if (img.dataset.src) {
          img.src = img.dataset.src;
        }
        
        img.classList.add('loaded');
        this.loadedImages.add(img);
        resolve(img);
      };

      imageLoader.onerror = reject;

      // Start loading
      if (img.dataset.srcset) {
        imageLoader.srcset = img.dataset.srcset;
      } else if (img.dataset.src) {
        imageLoader.src = img.dataset.src;
      }
    });
  }

  setupResponsiveImages() {
    // Configure responsive images based on device pixel ratio and viewport
    const pixelRatio = window.devicePixelRatio || 1;
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight
    };

    // Add responsive image sizing logic here
  }

  setupProgressiveLoading() {
    // Implement progressive JPEG and other progressive loading techniques
  }

  detectModernImageFormats() {
    this.supportedFormats = {
      webp: this.checkWebPSupport(),
      avif: this.checkAVIFSupport()
    };
  }

  checkWebPSupport() {
    return new Promise(resolve => {
      const webP = new Image();
      webP.onload = webP.onerror = () => {
        resolve(webP.height === 2);
      };
      webP.src = 'data:image/webp;base64,UklGRjoAAABXRUJQVlA4IC4AAACyAgCdASoCAAIALmk0mk0iIiIiIgBoSygABc6WWgAA/veff/0PP8bA//LwYAAA';
    });
  }

  checkAVIFSupport() {
    return new Promise(resolve => {
      const avif = new Image();
      avif.onload = avif.onerror = () => {
        resolve(avif.height === 2);
      };
      avif.src = 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgABogQEAwgMg8f8D///8WfhwB8+ErK42A=';
    });
  }
}

/**
 * React Component Performance Wrappers
 */

// Performance monitoring wrapper
const PerformanceWrapper = ({ children, componentName }) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      if (renderTime > 16.67) { // > 1 frame at 60fps
        console.warn(`Slow render detected in ${componentName}: ${renderTime.toFixed(2)}ms`);
      }
      
      // Record performance metric
      performance.measure(`render-${componentName}`, {
        start: startTime,
        end: endTime
      });
    };
  });

  return children;
};

// Memoized component factory
const createMemoizedComponent = (Component, propsAreEqual) => {
  return memo(Component, propsAreEqual);
};

// Optimized event handlers
const useOptimizedCallback = (callback, deps) => {
  return useCallback(callback, deps);
};

const useOptimizedMemo = (factory, deps) => {
  return useMemo(factory, deps);
};

// Virtual scrolling component
const VirtualScrollList = ({ items, renderItem, itemHeight = 50, containerHeight = 400 }) => {
  const [scrollTop, setScrollTop] = useState(0);
  
  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(startIndex + Math.ceil(containerHeight / itemHeight) + 1, items.length);
  
  const visibleItems = items.slice(startIndex, endIndex);
  const offsetY = startIndex * itemHeight;
  
  const handleScroll = useCallback((e) => {
    setScrollTop(e.target.scrollTop);
  }, []);

  return (
    <div 
      style={{ height: containerHeight, overflow: 'auto' }}
      onScroll={handleScroll}
    >
      <div style={{ height: items.length * itemHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map((item, index) => (
            <div key={startIndex + index} style={{ height: itemHeight }}>
              {renderItem(item, startIndex + index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Service Worker Integration for Advanced Caching
 */
class ServiceWorkerManager {
  constructor() {
    this.isSupported = 'serviceWorker' in navigator;
    this.registration = null;
    this.initializeServiceWorker();
  }

  async initializeServiceWorker() {
    if (!this.isSupported) {
      console.warn('Service Workers not supported');
      return;
    }

    try {
      this.registration = await navigator.serviceWorker.register('/sw.js');
      console.log('Service Worker registered successfully');
      
      // Listen for updates
      this.registration.addEventListener('updatefound', () => {
        const newWorker = this.registration.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed') {
            this.handleServiceWorkerUpdate();
          }
        });
      });
      
    } catch (error) {
      console.error('Service Worker registration failed:', error);
    }
  }

  handleServiceWorkerUpdate() {
    if (confirm('New version available. Reload to update?')) {
      window.location.reload();
    }
  }

  // Preload critical resources
  preloadCriticalResources(resources) {
    if (this.registration && this.registration.active) {
      this.registration.active.postMessage({
        type: 'PRELOAD_RESOURCES',
        resources: resources
      });
    }
  }
}

/**
 * Critical CSS and Resource Loading
 */
class CriticalResourceLoader {
  constructor() {
    this.loadedResources = new Set();
    this.criticalCSS = null;
    this.initializeCriticalLoading();
  }

  initializeCriticalLoading() {
    // Extract and inline critical CSS
    this.extractCriticalCSS();
    
    // Setup non-critical resource loading
    this.loadNonCriticalResources();
  }

  extractCriticalCSS() {
    // This would typically be done at build time
    // Here we simulate the process
    const criticalStyles = Array.from(document.styleSheets)
      .filter(sheet => sheet.href && sheet.href.includes('critical'))
      .map(sheet => this.extractCSSRules(sheet))
      .join('');

    if (criticalStyles) {
      this.inlineCSS(criticalStyles);
    }
  }

  extractCSSRules(stylesheet) {
    try {
      return Array.from(stylesheet.cssRules)
        .map(rule => rule.cssText)
        .join('');
    } catch (e) {
      // Cross-origin stylesheet
      return '';
    }
  }

  inlineCSS(css) {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.insertBefore(style, document.head.firstChild);
  }

  loadNonCriticalResources() {
    // Load non-critical CSS asynchronously
    this.loadNonCriticalCSS();
    
    // Load non-critical JavaScript
    this.loadNonCriticalJS();
    
    // Preload key resources
    this.preloadKeyResources();
  }

  loadNonCriticalCSS() {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/assets/non-critical.css';
    link.media = 'print';
    link.onload = () => {
      link.media = 'all';
    };
    document.head.appendChild(link);
  }

  loadNonCriticalJS() {
    const script = document.createElement('script');
    script.src = '/assets/non-critical.js';
    script.async = true;
    document.head.appendChild(script);
  }

  preloadKeyResources() {
    const resources = [
      { href: '/api/user-data', as: 'fetch', crossorigin: 'anonymous' },
      { href: '/assets/hero-image.webp', as: 'image' },
      { href: '/assets/icons.woff2', as: 'font', type: 'font/woff2', crossorigin: 'anonymous' }
    ];

    resources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      Object.assign(link, resource);
      document.head.appendChild(link);
    });
  }
}

/**
 * Loading and Error Components
 */
const LoadingSpinner = memo(() => (
  <div className="loading-spinner" role="status" aria-label="Loading">
    <div className="spinner"></div>
    <span className="sr-only">Loading...</span>
  </div>
));

const ProgressBar = memo(({ estimatedTime }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) return 100;
        // Exponential progress curve
        return prev + (100 - prev) * 0.1;
      });
    }, estimatedTime / 10);

    return () => clearInterval(interval);
  }, [estimatedTime]);

  return (
    <div className="progress-bar">
      <div 
        className="progress-fill" 
        style={{ width: `${progress}%` }}
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
});

const ErrorFallback = memo(({ error, resetError }) => (
  <div className="error-fallback" role="alert">
    <h2>Something went wrong</h2>
    <details>
      <summary>Error details</summary>
      <pre>{error.message}</pre>
    </details>
    <button onClick={resetError}>Try again</button>
  </div>
));

// Chunk analysis utilities
class ChunkAnalyzer {
  analyze() {
    // Implementation for chunk analysis
    return {
      chunks: [],
      optimization: [],
      dependencies: []
    };
  }
}

class DependencyOptimizer {
  optimize() {
    // Implementation for dependency optimization
    return {
      removedDependencies: [],
      optimizedImports: [],
      savings: 0
    };
  }
}

class TreeShaker {
  shake() {
    // Implementation for tree shaking analysis
    return {
      removedCode: [],
      savings: 0
    };
  }
}

// Export the main optimization manager and utilities
export {
  FrontendOptimizationManager,
  BundleOptimizer,
  ImageOptimizer,
  ServiceWorkerManager,
  CriticalResourceLoader,
  VirtualScrollList,
  PerformanceWrapper,
  createMemoizedComponent,
  useOptimizedCallback,
  useOptimizedMemo,
  LoadingSpinner,
  ProgressBar,
  ErrorFallback
};

// Default configuration
export const defaultFrontendConfig = {
  optimization: {
    enabled: true,
    chunkSize: 50000,
    preloadThreshold: 0.7,
    maxConcurrentLoads: 3,
    cacheSize: 100,
    performanceThreshold: 100
  },
  lazyLoading: {
    enabled: true,
    intersectionMargin: '50px',
    threshold: 0.01
  },
  bundleOptimization: {
    enabled: true,
    treeShaking: true,
    codesplitting: true,
    dynamicImports: true
  },
  serviceWorker: {
    enabled: true,
    cacheStrategy: 'stale-while-revalidate',
    preloadCritical: true
  }
};

// Initialize optimization manager
const optimizationManager = new FrontendOptimizationManager(defaultFrontendConfig.optimization);

// Global initialization
if (typeof window !== 'undefined') {
  // Initialize image optimization
  const imageOptimizer = new ImageOptimizer();
  imageOptimizer.lazyLoadImages();
  
  // Initialize service worker
  const serviceWorkerManager = new ServiceWorkerManager();
  
  // Initialize critical resource loader
  const criticalResourceLoader = new CriticalResourceLoader();
  
  // Expose to window for debugging
  window.NovaCronOptimization = {
    optimizationManager,
    imageOptimizer,
    serviceWorkerManager,
    criticalResourceLoader
  };
}