# AI Fallback Strategies for NovaCron

## Overview

NovaCron's scheduler and migration orchestrator are designed to provide robust fallback mechanisms when AI services are unavailable. This document outlines the comprehensive fallback strategies implemented to ensure system reliability and graceful degradation.

## Architecture

### Safe AI Provider Pattern

Both the scheduler and migration orchestrator use a "Safe AI Provider" pattern that wraps AI services with fallback logic:

```
[Application] → [SafeAIProvider] → [HTTPAIProvider] (if available)
                      ↓
               [FallbackStrategy] (if AI fails)
```

### Key Components

1. **SafeAIProvider**: Wrapper that handles AI service failures and timeouts
2. **FallbackStrategy**: Heuristic-based algorithms for when AI is unavailable
3. **Timeout Management**: Prevents hanging on slow/unresponsive AI services
4. **Metrics Collection**: Tracks fallback usage and AI failure rates

## Scheduler Fallback Strategies

### Resource Demand Prediction

When AI-based prediction fails, the system uses:

- **Time-based patterns**: Higher utilization during business hours (9-17), lower at night (0-6)
- **Resource-specific variance**: CPU is more variable, memory is stable, network can be bursty
- **Mathematical models**: Exponential smoothing with sine/cosine variations

```go
// Business hours have higher baseline utilization
if currentHour >= 9 && currentHour <= 17 {
    baseUtilization = 0.7
} else if currentHour >= 0 && currentHour <= 6 {
    baseUtilization = 0.3
}
```

### Performance Optimization

Fallback uses simple rule-based optimization:

- **CPU > 80%**: Scale up (add 2 nodes)
- **CPU < 30%**: Scale down (remove 1 node)
- **Multi-node clusters**: Enable rebalancing
- **Fair-share allocation**: Equal resource distribution

### Anomaly Detection

Threshold-based detection with configurable limits:

- **CPU usage > 90%**: High utilization anomaly
- **Memory usage > 95%**: Memory pressure anomaly
- **Error rate > 5%**: Quality degradation
- **Rapid changes > 30%**: Spike detection

### Scaling Recommendations

Rule-based scaling based on resource utilization:

- **Scale up** when utilization > 80%
- **Scale down** when utilization < 20%
- **Memory-specific** scaling when > 90% usage
- **Confidence scoring** based on threshold proximity

## Migration Fallback Strategies

### Migration Time Prediction

Heuristic-based time estimation:

```go
// Base durations by VM size
switch vmSize {
case "small":  2 minutes
case "medium": 5 minutes
case "large":  10 minutes
case "xlarge": 20 minutes
}

// Network overhead: +30% for cross-node migrations
```

### Bandwidth Requirements

Calculated based on VM size and time constraints:

1. Estimate VM size in GB
2. Calculate required transfer rate for target completion time
3. Adjust for network conditions (congested +50%, optimal -20%)
4. Cap at reasonable limits (10 Gbps max)

### Migration Strategy Selection

Workload-aware strategy selection:

- **Database VMs**: Post-copy migration (minimal downtime)
- **Web servers**: Pre-copy migration (tolerates brief downtime)
- **Batch processing**: Cold migration (cost-effective)
- **Large memory VMs**: Hybrid migration (adaptive approach)

### Compression Optimization

Data-type aware compression:

- **Text/logs**: GZIP level 7 (high compression ratio)
- **Binary/executable**: Zstd level 3 (balanced performance)
- **Media/video**: LZ4 level 1 (already compressed content)

### Network Path Selection

Simple path optimization:

1. **Direct connection**: Preferred when available
2. **Relay nodes**: Used when no direct path exists
3. **Topology awareness**: Considers network topology data
4. **Cost minimization**: Selects lowest-cost paths

## Error Handling and Recovery

### Timeout Management

Aggressive timeouts prevent system hanging:

- **Resource prediction**: 5 second timeout
- **Performance optimization**: 5 second timeout
- **Anomaly detection**: 3 second timeout (faster response needed)

### Retry Logic

Limited retries with exponential backoff:

```go
for attempt := 0; attempt <= maxRetries; attempt++ {
    if attempt > 0 {
        time.Sleep(time.Duration(math.Pow(2, float64(attempt))) * time.Second)
    }
    // Attempt AI call...
}
```

### Graceful Degradation

System continues operating with reduced intelligence:

1. **Fallback activation**: Automatic when AI fails
2. **Confidence scoring**: Lower confidence scores for heuristic results
3. **User notification**: Logs indicate fallback mode operation
4. **Metric tracking**: Fallback usage rates monitored

## Metrics and Monitoring

### Fallback Metrics

Each SafeAIProvider tracks:

- `total_calls`: Total AI function calls
- `fallback_calls`: Calls handled by fallback
- `ai_failures`: AI service failures
- `fallback_rate`: Percentage using fallback
- `avg_fallback_time`: Average fallback execution time

### Health Indicators

System health monitoring includes:

- **AI service availability**: Connection success rate
- **Response times**: AI service performance
- **Fallback effectiveness**: Fallback result quality
- **System stability**: Overall system performance under AI failure

## Configuration Options

### AI Configuration

```go
type AIConfig struct {
    Enabled                     bool
    Endpoint                    string
    Timeout                     time.Duration
    ConfidenceThreshold         float64
    RetryAttempts               int
    EnableProactiveScaling      bool
    EnableAnomalyDetection      bool
    EnablePredictiveAdjustments bool
}
```

### Fallback Tuning

Fallback strategies can be tuned via:

- **Threshold values**: Anomaly detection limits
- **Time patterns**: Business hour definitions
- **Scaling multipliers**: Resource scaling ratios
- **Confidence scores**: Heuristic confidence levels

## Testing Strategy

### Unit Tests

- Mock failing AI providers
- Test all fallback code paths
- Verify timeout handling
- Check metric collection

### Integration Tests

- Simulate AI service downtime
- Test under high load
- Verify graceful degradation
- Monitor system stability

### Load Testing

- Sustained operation without AI
- Performance comparison (AI vs fallback)
- Memory/CPU usage under fallback
- Concurrent operation handling

## Deployment Guidelines

### Production Deployment

1. **Enable fallbacks by default**: Always use SafeAIProvider
2. **Monitor fallback rates**: Alert on high fallback usage
3. **Tune timeouts appropriately**: Balance responsiveness vs reliability
4. **Log fallback activations**: Track when and why fallbacks occur

### AI Service Management

1. **Health check endpoints**: Monitor AI service health
2. **Circuit breaker pattern**: Temporarily disable AI after repeated failures
3. **Gradual retry**: Slowly increase AI usage after recovery
4. **A/B testing**: Compare AI vs fallback performance

## Performance Characteristics

### Resource Usage

- **Memory**: Fallback uses minimal additional memory
- **CPU**: Heuristic calculations are lightweight
- **Network**: No external calls for fallback operations
- **Latency**: Sub-millisecond fallback execution

### Accuracy Trade-offs

| Function | AI Accuracy | Fallback Accuracy | Confidence |
|----------|-------------|-------------------|------------|
| Resource Prediction | 90-95% | 60-70% | High vs Medium |
| Anomaly Detection | 95-98% | 70-80% | High vs Medium |
| Performance Optimization | 85-90% | 50-60% | Medium vs Low |
| Migration Planning | 90-95% | 65-75% | High vs Medium |

## Best Practices

### Development

1. **Defensive programming**: Always assume AI might fail
2. **Timeout everything**: Set reasonable timeouts for all AI calls
3. **Test fallback paths**: Ensure fallback code is well-tested
4. **Monitor metrics**: Track fallback usage in production

### Operations

1. **Baseline fallback performance**: Know what to expect without AI
2. **Alert on AI failures**: Monitor AI service health
3. **Plan for AI outages**: Have runbooks for AI service recovery
4. **Regular testing**: Periodically test system without AI

### Troubleshooting

1. **Check AI connectivity**: Verify network access to AI services
2. **Review timeout settings**: Ensure timeouts are appropriate
3. **Monitor fallback metrics**: Look for patterns in failures
4. **Validate fallback results**: Compare fallback vs AI accuracy

## Future Improvements

### Enhanced Heuristics

- **Machine learning fallbacks**: Train lightweight models for offline use
- **Historical pattern analysis**: Use past data for better predictions
- **Adaptive thresholds**: Self-tuning anomaly detection
- **Context awareness**: Consider more environmental factors

### Smart Caching

- **Result caching**: Cache recent AI predictions
- **Stale data tolerance**: Use cached results when AI unavailable
- **Intelligent refresh**: Refresh cache based on confidence decay
- **Hybrid approaches**: Combine cached AI results with heuristics

### Advanced Monitoring

- **Predictive health**: Predict AI service failures
- **Quality scoring**: Measure fallback result quality
- **Automatic recovery**: Self-healing AI service connections
- **Performance optimization**: Continuously improve fallback algorithms

## Conclusion

NovaCron's AI fallback strategies ensure robust operation even when AI services are unavailable. The system provides graceful degradation with reasonable accuracy using heuristic approaches, comprehensive error handling, and extensive monitoring. This architecture ensures high availability and reliability for critical infrastructure operations.