# Media Serving & CDN Optimization for Spark Dating App

## Overview

This document outlines the comprehensive media optimization strategy for the Spark dating app, focusing on profile image delivery, video content, and CDN architecture optimized for dating app usage patterns.

## Media Performance Requirements

### Target Metrics
- **Image Load Time**: <500ms for profile thumbnails, <2s for full-size images
- **Video Streaming**: <3s initial buffering, <1s seek time
- **CDN Cache Hit Rate**: >95% for images, >90% for video content
- **Bandwidth Efficiency**: 60% reduction through optimization
- **Mobile Optimization**: Adaptive quality based on connection speed

## CDN Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Global CDN Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CloudFlare  │  │    AWS      │  │    Regional         │  │
│  │   Edge      │  │ CloudFront  │  │    Caches           │  │
│  │             │  │             │  │                     │  │
│  │ • Image     │  │ • Video     │  │ • User proximity    │  │
│  │   optimization│ │   streaming │  │ • Language-based   │  │
│  │ • WebP/AVIF │  │ • Adaptive  │  │ • Device-specific   │  │
│  │ • Smart     │  │   bitrate   │  │   optimization      │  │
│  │   compression│ │ • HLS/DASH  │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
        ┌─────────────▼─────────────┐    ┌▼──────────────────┐
        │     Media Processing      │    │   Origin Servers   │
        │        Pipeline           │    │                   │
        │                          │    │ • Master images   │
        │ • Real-time resizing     │    │ • Original videos │
        │ • Format conversion      │    │ • Metadata store  │
        │ • Quality optimization   │    │ • Backup storage  │
        │ • Watermark insertion    │    │                   │
        └──────────────────────────┘    └───────────────────┘
```

## Image Optimization Strategy

### Multi-Format Strategy
```go
type ImageOptimizer struct {
    processor    *ImageProcessor
    cdnManager   *CDNManager
    cacheManager *CacheManager
    metrics      *OptimizationMetrics
}

type ImageFormat struct {
    Format      string  // "webp", "avif", "jpeg", "png"
    Quality     int     // 1-100
    MaxWidth    int
    MaxHeight   int
    Compression string  // "lossless", "lossy"
}

// Spark-specific image configurations
var SparkImageConfigs = map[string][]ImageFormat{
    "profile_thumbnail": {
        {Format: "avif", Quality: 80, MaxWidth: 150, MaxHeight: 150},
        {Format: "webp", Quality: 85, MaxWidth: 150, MaxHeight: 150},
        {Format: "jpeg", Quality: 90, MaxWidth: 150, MaxHeight: 150},
    },
    "profile_card": {
        {Format: "avif", Quality: 85, MaxWidth: 400, MaxHeight: 600},
        {Format: "webp", Quality: 90, MaxWidth: 400, MaxHeight: 600},
        {Format: "jpeg", Quality: 92, MaxWidth: 400, MaxHeight: 600},
    },
    "profile_fullsize": {
        {Format: "avif", Quality: 90, MaxWidth: 1080, MaxHeight: 1920},
        {Format: "webp", Quality: 95, MaxWidth: 1080, MaxHeight: 1920},
        {Format: "jpeg", Quality: 95, MaxWidth: 1080, MaxHeight: 1920},
    },
}

func (io *ImageOptimizer) OptimizeProfileImage(userID, imageID string, originalImage []byte) error {
    // Generate all required formats and sizes
    for useCase, formats := range SparkImageConfigs {
        for _, format := range formats {
            optimized, err := io.processor.Process(originalImage, ProcessingOptions{
                Format:     format.Format,
                Quality:    format.Quality,
                MaxWidth:   format.MaxWidth,
                MaxHeight:  format.MaxHeight,
                Progressive: true, // Progressive JPEG for better perceived performance
                Strip:      true,  // Remove EXIF data for privacy and size
            })
            
            if err != nil {
                return fmt.Errorf("image processing failed: %w", err)
            }
            
            // Upload to CDN with smart cache headers
            cdnKey := fmt.Sprintf("profiles/%s/%s/%s.%s", userID, useCase, imageID, format.Format)
            err = io.cdnManager.Upload(cdnKey, optimized, CDNUploadOptions{
                ContentType: fmt.Sprintf("image/%s", format.Format),
                CacheControl: "public, max-age=31536000", // 1 year for immutable content
                Metadata: map[string]string{
                    "user-id":   userID,
                    "use-case":  useCase,
                    "format":    format.Format,
                    "optimized": "true",
                },
            })
            
            if err != nil {
                return fmt.Errorf("CDN upload failed: %w", err)
            }
        }
    }
    
    return nil
}
```

### Smart Image Delivery
```go
// Client-aware image serving
func (api *SparkMediaAPI) ServeProfileImage(ctx *gin.Context) {
    userID := ctx.Param("user_id")
    imageID := ctx.Param("image_id")
    useCase := ctx.DefaultQuery("size", "profile_card")
    
    // Detect client capabilities
    acceptHeader := ctx.GetHeader("Accept")
    userAgent := ctx.GetHeader("User-Agent")
    
    // Determine optimal format
    var format string
    if strings.Contains(acceptHeader, "image/avif") {
        format = "avif"
    } else if strings.Contains(acceptHeader, "image/webp") {
        format = "webp"
    } else {
        format = "jpeg"
    }
    
    // Adjust quality based on connection
    quality := api.detectConnectionSpeed(ctx)
    adjustedUseCase := api.adjustForConnection(useCase, quality)
    
    // Generate optimized URL
    cdnURL := fmt.Sprintf("https://cdn.sparkdating.com/profiles/%s/%s/%s.%s", 
        userID, adjustedUseCase, imageID, format)
    
    // Set smart cache headers
    api.setImageCacheHeaders(ctx, userID, imageID)
    
    // Redirect to CDN or serve directly based on load
    if api.shouldServeDirect() {
        ctx.Data(200, fmt.Sprintf("image/%s", format), imageData)
    } else {
        ctx.Redirect(302, cdnURL)
    }
}

// Connection speed detection
func (api *SparkMediaAPI) detectConnectionSpeed(ctx *gin.Context) ConnectionSpeed {
    // Check for connection speed hints
    if effectiveType := ctx.GetHeader("ECT"); effectiveType != "" {
        switch effectiveType {
        case "slow-2g", "2g":
            return Slow2G
        case "3g":
            return Regular3G
        case "4g":
            return LTE
        default:
            return WiFi
        }
    }
    
    // Check for data saver mode
    if saveData := ctx.GetHeader("Save-Data"); saveData == "on" {
        return DataSaver
    }
    
    // Default to reasonable assumption
    return Regular3G
}
```

### Progressive Image Loading
```javascript
// Frontend progressive image loading
class SparkImageLoader {
    constructor() {
        this.imageCache = new Map();
        this.loadingQueue = [];
        this.maxConcurrentLoads = 6;
        this.currentLoads = 0;
    }
    
    // Load profile image with progressive enhancement
    loadProfileImage(userID, imageID, container, options = {}) {
        const sizes = ['thumbnail', 'card', 'fullsize'];
        const formats = this.getSupportedFormats();
        
        // Start with thumbnail for immediate display
        this.loadImageProgressive(userID, imageID, sizes, formats, container, options);
    }
    
    loadImageProgressive(userID, imageID, sizes, formats, container, options) {
        const img = container.querySelector('img') || document.createElement('img');
        
        // Load thumbnail first (usually cached)
        const thumbnailURL = this.buildImageURL(userID, imageID, 'thumbnail', formats[0]);
        
        img.src = thumbnailURL;
        img.style.filter = 'blur(2px)'; // Blur placeholder
        img.style.transition = 'filter 0.3s ease';
        
        if (!container.contains(img)) {
            container.appendChild(img);
        }
        
        // Load higher quality version
        const targetSize = options.size || 'card';
        const highQualityURL = this.buildImageURL(userID, imageID, targetSize, formats[0]);
        
        // Preload high quality image
        const highQualityImg = new Image();
        highQualityImg.onload = () => {
            img.src = highQualityURL;
            img.style.filter = 'none';
            
            // Cache the loaded image
            this.imageCache.set(highQualityURL, highQualityImg);
        };
        
        // Add to loading queue to manage concurrent loads
        this.queueImageLoad(highQualityImg, highQualityURL);
    }
    
    // Intelligent image format selection
    getSupportedFormats() {
        const formats = [];
        
        // Check AVIF support
        if (this.supportsFormat('avif')) {
            formats.push('avif');
        }
        
        // Check WebP support
        if (this.supportsFormat('webp')) {
            formats.push('webp');
        }
        
        // Always include JPEG as fallback
        formats.push('jpeg');
        
        return formats;
    }
    
    // Network-aware loading
    adjustLoadingStrategy() {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        
        if (connection) {
            const effectiveType = connection.effectiveType;
            const saveData = connection.saveData;
            
            if (saveData || effectiveType === 'slow-2g' || effectiveType === '2g') {
                // Use data saver mode
                this.maxConcurrentLoads = 2;
                return 'data-saver';
            } else if (effectiveType === '3g') {
                this.maxConcurrentLoads = 4;
                return 'medium-quality';
            } else {
                this.maxConcurrentLoads = 6;
                return 'high-quality';
            }
        }
        
        return 'medium-quality';
    }
}
```

## Video Optimization Strategy

### Adaptive Streaming Setup
```go
type VideoOptimizer struct {
    encoder     *FFmpegEncoder
    cdnManager  *CDNManager
    storage     *VideoStorage
    analytics   *VideoAnalytics
}

// Video encoding profiles for dating app
var SparkVideoProfiles = []EncodingProfile{
    {
        Name:       "mobile_low",
        Resolution: "480x640",
        Bitrate:    500,  // kbps
        FPS:        24,
        Format:     "mp4",
        Codec:      "h264",
        AudioCodec: "aac",
        AudioBitrate: 64,
    },
    {
        Name:       "mobile_high",
        Resolution: "720x960",
        Bitrate:    1200,
        FPS:        30,
        Format:     "mp4",
        Codec:      "h264",
        AudioCodec: "aac",
        AudioBitrate: 128,
    },
    {
        Name:       "web_hd",
        Resolution: "1080x1920",
        Bitrate:    2500,
        FPS:        30,
        Format:     "mp4",
        Codec:      "h264",
        AudioCodec: "aac",
        AudioBitrate: 192,
    },
}

func (vo *VideoOptimizer) ProcessProfileVideo(userID, videoID string, originalVideo []byte) error {
    // Process multiple quality levels
    var wg sync.WaitGroup
    errChan := make(chan error, len(SparkVideoProfiles))
    
    for _, profile := range SparkVideoProfiles {
        wg.Add(1)
        go func(p EncodingProfile) {
            defer wg.Done()
            
            encoded, err := vo.encoder.Encode(originalVideo, EncodingOptions{
                Profile:    p,
                StartTime:  0,
                Duration:   30, // Limit to 30 seconds for dating app
                Watermark:  false,
                Normalize:  true, // Normalize audio levels
            })
            
            if err != nil {
                errChan <- fmt.Errorf("encoding failed for profile %s: %w", p.Name, err)
                return
            }
            
            // Upload to CDN
            cdnKey := fmt.Sprintf("videos/%s/%s/%s.%s", userID, p.Name, videoID, p.Format)
            err = vo.cdnManager.Upload(cdnKey, encoded, CDNUploadOptions{
                ContentType: fmt.Sprintf("video/%s", p.Format),
                CacheControl: "public, max-age=2592000", // 30 days
            })
            
            if err != nil {
                errChan <- fmt.Errorf("CDN upload failed for %s: %w", p.Name, err)
            }
        }(profile)
    }
    
    wg.Wait()
    close(errChan)
    
    // Check for errors
    for err := range errChan {
        if err != nil {
            return err
        }
    }
    
    // Generate HLS playlist for adaptive streaming
    return vo.generateHLSPlaylist(userID, videoID)
}

// HLS Playlist generation
func (vo *VideoOptimizer) generateHLSPlaylist(userID, videoID string) error {
    playlist := `#EXTM3U
#EXT-X-VERSION:3

#EXT-X-STREAM-INF:BANDWIDTH=564000,RESOLUTION=480x640,FRAME-RATE=24.000
https://cdn.sparkdating.com/videos/%s/mobile_low/%s.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=1328000,RESOLUTION=720x960,FRAME-RATE=30.000
https://cdn.sparkdating.com/videos/%s/mobile_high/%s.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=2628000,RESOLUTION=1080x1920,FRAME-RATE=30.000
https://cdn.sparkdating.com/videos/%s/web_hd/%s.m3u8
`
    
    playlistContent := fmt.Sprintf(playlist, userID, videoID, userID, videoID, userID, videoID)
    
    cdnKey := fmt.Sprintf("videos/%s/playlist/%s.m3u8", userID, videoID)
    return vo.cdnManager.Upload(cdnKey, []byte(playlistContent), CDNUploadOptions{
        ContentType:  "application/x-mpegURL",
        CacheControl: "public, max-age=300", // 5 minutes for playlist
    })
}
```

## CDN Configuration & Optimization

### CloudFlare Configuration
```yaml
# CloudFlare Workers for dynamic optimization
cloudflare_config:
  zones:
    - name: "sparkdating.com"
      plan: "pro"
      settings:
        security_level: "medium"
        cache_level: "aggressive"
        browser_cache_ttl: 31536000  # 1 year
        edge_cache_ttl: 604800      # 1 week
        
  page_rules:
    - pattern: "cdn.sparkdating.com/profiles/*"
      settings:
        cache_level: "cache_everything"
        edge_cache_ttl: 2592000     # 30 days
        browser_cache_ttl: 31536000  # 1 year
        
    - pattern: "cdn.sparkdating.com/videos/*"
      settings:
        cache_level: "cache_everything"
        edge_cache_ttl: 604800      # 1 week
        browser_cache_ttl: 2592000   # 30 days
        
  image_optimization:
    enabled: true
    formats: ["webp", "avif"]
    quality: "lossless"
    metadata: "remove"
    
  video_optimization:
    enabled: true
    formats: ["mp4", "webm"]
    adaptive_bitrate: true
```

### AWS CloudFront Distribution
```json
{
  "DistributionConfig": {
    "CallerReference": "spark-media-cdn-2024",
    "Comment": "Spark Dating App Media CDN",
    "DefaultRootObject": "index.html",
    "Origins": [
      {
        "Id": "spark-media-origin",
        "DomainName": "media-origin.sparkdating.com",
        "CustomOriginConfig": {
          "HTTPPort": 443,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "https-only",
          "OriginSSLProtocols": ["TLSv1.2"]
        }
      }
    ],
    "DefaultCacheBehavior": {
      "TargetOriginId": "spark-media-origin",
      "ViewerProtocolPolicy": "redirect-to-https",
      "CachePolicyId": "spark-media-cache-policy",
      "Compress": true,
      "ResponseHeadersPolicyId": "spark-security-headers"
    },
    "CacheBehaviors": [
      {
        "PathPattern": "/profiles/*",
        "TargetOriginId": "spark-media-origin",
        "ViewerProtocolPolicy": "https-only",
        "CachePolicyId": "long-term-cache-policy",
        "TTL": {
          "DefaultTTL": 2592000,
          "MaxTTL": 31536000
        }
      },
      {
        "PathPattern": "/videos/*",
        "TargetOriginId": "spark-media-origin",
        "ViewerProtocolPolicy": "https-only",
        "CachePolicyId": "video-cache-policy",
        "TTL": {
          "DefaultTTL": 604800,
          "MaxTTL": 2592000
        }
      }
    ],
    "PriceClass": "PriceClass_All",
    "Enabled": true,
    "HttpVersion": "http2"
  }
}
```

## Performance Monitoring for Media

### Media-Specific Metrics
```go
type MediaPerformanceMetrics struct {
    imageLoadTime         prometheus.Histogram
    videoBufferTime       prometheus.Histogram
    cdnCacheHitRate      prometheus.CounterVec
    bandwidthUsage       prometheus.CounterVec
    formatOptimization   prometheus.CounterVec
    connectionSpeedDist  prometheus.Histogram
}

func NewMediaMetrics() *MediaPerformanceMetrics {
    return &MediaPerformanceMetrics{
        imageLoadTime: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name: "spark_image_load_duration_seconds",
            Help: "Time to load profile images",
            Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 5, 10},
        }),
        
        videoBufferTime: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name: "spark_video_buffer_duration_seconds", 
            Help: "Video buffering time",
            Buckets: []float64{0.5, 1, 2, 3, 5, 10, 15},
        }),
        
        cdnCacheHitRate: prometheus.NewCounterVec(prometheus.CounterOpts{
            Name: "spark_cdn_requests_total",
            Help: "CDN requests by cache result",
        }, []string{"content_type", "result", "region"}),
        
        formatOptimization: prometheus.NewCounterVec(prometheus.CounterOpts{
            Name: "spark_format_served_total",
            Help: "Image/video formats served",
        }, []string{"format", "content_type", "quality"}),
    }
}

// Track media performance
func (m *MediaPerformanceMetrics) RecordImageLoad(duration time.Duration, format string, size string, cacheHit bool) {
    m.imageLoadTime.Observe(duration.Seconds())
    
    result := "miss"
    if cacheHit {
        result = "hit"
    }
    
    m.cdnCacheHitRate.With(prometheus.Labels{
        "content_type": "image",
        "result":       result,
        "region":       m.getRegion(),
    }).Inc()
    
    m.formatOptimization.With(prometheus.Labels{
        "format":       format,
        "content_type": "image",
        "quality":      size,
    }).Inc()
}
```

### Media Performance Alerts
```yaml
media_alerts:
  - alert: SlowImageLoading
    expr: histogram_quantile(0.95, rate(spark_image_load_duration_seconds_bucket[5m])) > 2
    for: 3m
    labels:
      severity: warning
      component: media
    annotations:
      summary: "Profile images loading slowly"
      description: "P95 image load time is {{ $value }}s"
      
  - alert: LowCDNCacheHitRate
    expr: rate(spark_cdn_requests_total{result="hit"}[10m]) / rate(spark_cdn_requests_total[10m]) < 0.9
    for: 5m
    labels:
      severity: critical
      component: cdn
    annotations:
      summary: "CDN cache hit rate below target"
      description: "Cache hit rate is {{ $value | humanizePercentage }}"
      
  - alert: HighVideoBandwidthUsage
    expr: rate(spark_video_bandwidth_bytes_total[5m]) > 1000000000  # 1GB/s
    for: 2m
    labels:
      severity: warning
      component: video
    annotations:
      summary: "High video bandwidth usage"
      description: "Video bandwidth usage is {{ $value | humanizeBytes }}/s"
```

## Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|---------|--------|-------------|
| Profile Image Load Time | 2.5s | 0.4s | 84% faster |
| CDN Cache Hit Rate | 80% | 95% | 19% improvement |
| Bandwidth Usage | 100% | 40% | 60% reduction |
| Video Start Time | 5s | 1.2s | 76% faster |
| Mobile Data Usage | 100% | 35% | 65% reduction |

This comprehensive media optimization strategy ensures optimal image and video delivery performance while minimizing bandwidth usage and providing excellent user experience across all devices and connection speeds.