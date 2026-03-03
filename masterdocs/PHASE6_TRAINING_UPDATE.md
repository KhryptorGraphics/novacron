# DWCP v3 Phase 6 Training Materials - Production Insights

**Version**: 1.0.0  
**Last Updated**: 2025-11-10  
**Audience**: Engineering Teams, New Hires, Operations  
**Classification**: Internal Training Use

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Production Operations Training](#production-operations-training)
3. [Incident Response Certification](#incident-response-certification)
4. [Advanced Troubleshooting Workshop](#advanced-troubleshooting-workshop)
5. [Performance Tuning Course](#performance-tuning-course)
6. [Security Best Practices Training](#security-best-practices-training)
7. [On-Call Readiness Program](#on-call-readiness-program)
8. [Hands-On Labs](#hands-on-labs)
9. [Certification Requirements](#certification-requirements)
10. [Training Assessment](#training-assessment)

---

## Training Overview

### Training Philosophy

**Learning by Doing**: All training includes hands-on exercises using production-like scenarios

**Progressive Complexity**: Start with basics, progressively increase difficulty

**Real-World Scenarios**: Use actual incidents and issues from production

**Continuous Learning**: Training never stops - new scenarios added monthly

### Training Tracks

```
New Hire Track (2 weeks)
├── Week 1: System Architecture & Development
│   ├── Day 1-2: DWCP v3 Architecture
│   ├── Day 3-4: Development Environment Setup
│   └── Day 5: First Deployment
└── Week 2: Operations & Production
    ├── Day 1-2: Production Systems Overview
    ├── Day 3-4: Monitoring & Troubleshooting
    └── Day 5: First On-Call Shadow

SRE Track (4 weeks)
├── Week 1: Production Operations
├── Week 2: Incident Response
├── Week 3: Performance & Reliability
└── Week 4: Advanced Topics & Certification

Ongoing Track (continuous)
├── Monthly: New incident post-mortems
├── Quarterly: Technology deep dives
└── Annual: Certification renewal
```

---

## Production Operations Training

### Module 1: Production System Architecture

**Duration**: 4 hours

**Learning Objectives**:
- Understand DWCP v3 production topology
- Know all system components and their interactions
- Identify critical vs non-critical systems
- Understand data flow and dependencies

**Topics Covered**:
```yaml
architecture:
  - Control plane vs data plane
  - Multi-region deployment
  - Service mesh architecture
  - Database topology
  - Message queue infrastructure
  - Caching layers
  - Load balancing strategy
  - CDN configuration

dependencies:
  - Internal service dependencies
  - External service dependencies
  - Database dependencies
  - Infrastructure dependencies

failure_modes:
  - Single point of failure analysis
  - Cascading failure scenarios
  - Circuit breaker patterns
  - Graceful degradation
```

**Lab Exercise 1.1: Architecture Walkthrough**
```bash
#!/bin/bash
# Lab: Trace a request through the system

# Task 1: Identify all services involved in user login
echo "1. User sends login request to Load Balancer"
echo "2. Load Balancer routes to API Gateway"
echo "3. API Gateway authenticates with Auth Service"
echo "4. Auth Service queries User Database"
echo "5. Session stored in Redis"
echo "6. Response returned to user"

# Task 2: Map dependencies
kubectl get services --all-namespaces
kubectl get deployments --all-namespaces

# Task 3: Identify failure points
# Q: What happens if Redis is down?
# A: Authentication still works, but sessions not cached
# Impact: Increased database load, slower response times
```

**Lab Exercise 1.2: Dependency Mapping**
```bash
# Create dependency graph
./scripts/create-dependency-graph.sh > /tmp/dependencies.dot
dot -Tpng /tmp/dependencies.dot -o dependencies.png

# Questions to answer:
# 1. Which service has the most dependencies?
# 2. What is the impact if database goes down?
# 3. Which services can run in degraded mode?
```

**Assessment**:
- Quiz: 20 questions on architecture
- Practical: Draw dependency diagram
- Pass score: 80%

---

### Module 2: Daily Operations

**Duration**: 3 hours

**Learning Objectives**:
- Perform daily operational tasks
- Use monitoring dashboards effectively
- Identify normal vs abnormal patterns
- Execute routine maintenance procedures

**Topics Covered**:
```yaml
daily_tasks:
  - Morning health check
  - Log review
  - Metric analysis
  - Capacity check
  - Backup verification
  - Security scan review

tools:
  - Grafana dashboards
  - Kubectl commands
  - Log analysis (Kibana)
  - Alerting (AlertManager)
  - Incident management (PagerDuty)

procedures:
  - Deployment verification
  - Configuration updates
  - Secret rotation
  - Certificate renewal
  - Performance tuning
```

**Lab Exercise 2.1: Morning Checklist**
```bash
#!/bin/bash
# Lab: Complete morning operational checklist

echo "=== MORNING CHECKLIST LAB ==="

# 1. System health
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running

# 2. Review overnight alerts
kubectl logs -n monitoring alertmanager-0 --since=8h | grep firing

# 3. Check key metrics
# Navigate to: https://grafana.dwcp.io/d/system-health
# Record:
# - API latency p95: ____ms
# - Error rate: ____%
# - CPU utilization: ____%
# - Memory utilization: ____%

# 4. Verify backups
./scripts/verify-backups.sh --last 24h

# 5. Security scan results
trivy image --severity HIGH,CRITICAL dwcp/api:latest

# Complete checklist and submit results
```

**Lab Exercise 2.2: Metric Interpretation**
```
Scenario: You observe these metrics at 9am Monday

Metric              | Value      | Normal Range
--------------------|------------|-------------
API Latency (p95)   | 450ms      | <200ms
Error Rate          | 2%         | <0.1%
CPU Utilization     | 88%        | 40-70%
Memory Utilization  | 92%        | 50-75%
Request Rate        | 15k RPS    | 8k RPS

Questions:
1. What is likely happening?
2. Is immediate action required?
3. What would you do first?
4. How would you investigate?

Expected Answer:
1. System experiencing higher than normal load
2. Yes - high error rate requires immediate attention
3. Check if it's due to traffic spike or system issue
4. Check recent deployments, review error logs, verify external dependencies
```

---

## Incident Response Certification

### Module 3: Incident Response Framework

**Duration**: 6 hours

**Learning Objectives**:
- Understand incident severity levels
- Follow incident response playbooks
- Communicate effectively during incidents
- Conduct post-incident reviews

**Topics Covered**:
```yaml
incident_framework:
  - Severity classification
  - Response team roles
  - Communication protocols
  - Escalation procedures
  - Documentation requirements

playbooks:
  - High error rate
  - Service outage
  - Database performance
  - Security breach
  - Data corruption

communication:
  - Stakeholder updates
  - User communication
  - Internal coordination
  - Post-mortem reports
```

**Lab Exercise 3.1: Simulated Incident**
```
Scenario: API Error Rate Spike

Time: 14:35
Alert: "High error rate detected - 5%"

Initial Information:
- Error rate jumped from 0.1% to 5%
- Started 5 minutes ago
- Deployment completed 10 minutes ago
- Service: dwcp-api

Your Task:
1. Assess severity (you have 5 minutes)
2. Declare incident with appropriate severity
3. Form response team
4. Investigate root cause
5. Implement fix
6. Verify resolution
7. Document timeline

Expected Response:
1. Severity: SEV-2 (High - service degraded)
2. Declare: `/incident declare --severity SEV-2 --title "API error spike post-deployment"`
3. Team: On-call SRE (you), Engineering lead, Service owner
4. Investigation:
   - Check recent deployment
   - Review error logs
   - Compare metrics before/after deployment
5. Fix: Rollback deployment
6. Verification: Error rate returns to <0.1%
7. Timeline documented in incident channel

Evaluation Criteria:
- Time to acknowledge: <5 minutes ✓
- Severity classification: Correct ✓
- Communication: Updates every 5 minutes ✓
- Resolution time: <30 minutes ✓
- Documentation: Complete ✓
```

**Lab Exercise 3.2: War Game Exercise**
```
Scenario: Multi-Region Database Failure

Time: 03:00 (night shift)
Situation:
- Primary database in us-east-1 is down
- Replica lag in us-west-2 is 30 seconds
- eu-west-1 replica is healthy
- Users reporting errors globally
- On-call: You (alone for first 15 minutes)

Challenges:
- Middle of night, limited team available
- Multi-region impact
- Data replication concerns
- User impact across time zones

Your Actions:
1. Immediate response (0-5 min)
2. Assessment (5-10 min)
3. Decision making (10-15 min)
4. Execution (15-30 min)
5. Verification (30-45 min)
6. Communication (continuous)

Trainer Notes:
- Observe decision-making under pressure
- Evaluate communication clarity
- Assess technical troubleshooting
- Review use of runbooks
- Grade on: Response time, decisions, communication, resolution
```

**Certification Requirements**:
- Complete all 3 simulated incident exercises
- Score >85% on incident response quiz
- Shadow 3 real incidents
- Conduct 1 post-incident review presentation
- Valid for: 6 months

---

## Advanced Troubleshooting Workshop

### Module 4: Deep Troubleshooting Techniques

**Duration**: 8 hours

**Learning Objectives**:
- Use advanced debugging tools
- Analyze system behavior at low level
- Trace requests across services
- Profile application performance
- Debug production issues safely

**Topics Covered**:
```yaml
debugging_tools:
  - kubectl debug
  - Distributed tracing (Jaeger)
  - Performance profiling
  - Heap analysis
  - Network debugging
  - Log correlation

techniques:
  - Binary search debugging
  - Correlation analysis
  - Hypothesis testing
  - Root cause analysis
  - Performance profiling

production_safety:
  - Read-only debugging
  - Minimal impact troubleshooting
  - Safe state collection
  - Non-invasive monitoring
```

**Lab Exercise 4.1: Memory Leak Detective**
```bash
#!/bin/bash
# Lab: Find and fix a memory leak

# Scenario:
# Service memory usage increases 100MB/hour
# After 24 hours, pods are OOMKilled
# Deployment restarts temporarily fix it

# Task 1: Detect the leak
kubectl top pods -n production -l app=leaky-service --watch

# Task 2: Collect heap dump
POD=$(kubectl get pods -n production -l app=leaky-service -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD -- curl http://localhost:6060/debug/pprof/heap > heap.prof

# Task 3: Analyze heap dump
go tool pprof heap.prof
# Commands to use:
# top     - Show top memory consumers
# list    - Show source code
# web     - Visualize call graph

# Task 4: Identify leak pattern
# Expected finding: Goroutine leak - not properly closing connections

# Task 5: Verify fix
# After applying fix, monitor for 6 hours
# Memory should stabilize

# Success Criteria:
# - Correctly identified leak source
# - Proposed valid fix
# - Verified fix effectiveness
```

**Lab Exercise 4.2: Distributed Tracing Challenge**
```
Scenario: Slow Request Mystery

User reports: "Checkout takes 30 seconds"
Normal checkout time: 2 seconds

Trace ID: abc123-def456-ghi789

Task:
1. Open Jaeger: https://jaeger.dwcp.io
2. Load trace: abc123-def456-ghi789
3. Analyze trace spans
4. Identify slow component
5. Determine root cause
6. Propose fix

Trace Breakdown:
- Total time: 30.2 seconds
  - API Gateway: 0.1s
  - Auth Service: 0.2s
  - Order Service: 0.3s
  - Payment Service: 29.5s ← PROBLEM
    - Payment Gateway API Call: 29.4s
  - Notification Service: 0.1s

Analysis:
- Payment Gateway timeout set to 30s
- Payment Gateway actually responding in 0.5s
- Network issue causing 29s delay

Root Cause:
- Service mesh routing issue
- Traffic going through wrong network path
- Adding 29s of latency

Fix:
- Update service mesh virtual service
- Correct network path
- Verify latency returns to normal

Evaluation:
- Correctly used Jaeger: ✓
- Identified slow span: ✓
- Found root cause: ✓
- Proposed correct fix: ✓
```

---

## Performance Tuning Course

### Module 5: Performance Optimization

**Duration**: 6 hours

**Learning Objectives**:
- Identify performance bottlenecks
- Optimize database queries
- Tune application performance
- Configure caching effectively
- Optimize resource utilization

**Lab Exercise 5.1: Database Query Optimization**
```sql
-- Scenario: Dashboard loads slowly (15 seconds)
-- Task: Optimize these queries

-- BEFORE (Slow Query):
SELECT 
    u.id,
    u.name,
    u.email,
    (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count,
    (SELECT SUM(total) FROM orders WHERE user_id = u.id) as total_spent,
    (SELECT MAX(created_at) FROM orders WHERE user_id = u.id) as last_order
FROM users u
WHERE u.created_at > NOW() - INTERVAL '30 days'
ORDER BY u.created_at DESC;

-- Execution time: 15 seconds
-- Problem: N+1 queries, missing indexes

-- AFTER (Optimized Query):
SELECT 
    u.id,
    u.name,
    u.email,
    COUNT(o.id) as order_count,
    COALESCE(SUM(o.total), 0) as total_spent,
    MAX(o.created_at) as last_order
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id, u.name, u.email
ORDER BY u.created_at DESC;

-- Add indexes:
CREATE INDEX idx_users_created ON users(created_at);
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Execution time: 0.3 seconds
-- Improvement: 50x faster

-- Questions:
-- 1. Why was the first query slow?
-- 2. How does the optimization help?
-- 3. What indexes are critical?
-- 4. What else could be optimized?
```

**Certification**: Performance Tuning Specialist
- Complete all optimization labs
- Achieve >10x improvement in at least one scenario
- Present optimization case study
- Valid for: 1 year

---

## Security Best Practices Training

### Module 6: Production Security

**Duration**: 4 hours

**Learning Objectives**:
- Understand security threats
- Implement security best practices
- Handle security incidents
- Audit security configurations

**Lab Exercise 6.1: Security Audit**
```bash
#!/bin/bash
# Lab: Conduct security audit

# Task 1: Scan for secrets in code
git-secrets --scan

# Task 2: Check for vulnerabilities
trivy image dwcp/api:latest --severity HIGH,CRITICAL

# Task 3: Audit RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:production:api

# Task 4: Check network policies
kubectl get networkpolicies --all-namespaces

# Task 5: Review exposed services
kubectl get services --all-namespaces -o wide | grep LoadBalancer

# Expected Findings (training environment):
# - 2 HIGH vulnerabilities in dependencies
# - 1 service account with excessive permissions
# - 3 namespaces without network policies
# - 1 service unnecessarily exposed

# Remediation:
# 1. Update vulnerable dependencies
# 2. Reduce RBAC permissions to minimum
# 3. Create network policies
# 4. Change service type to ClusterIP
```

---

## On-Call Readiness Program

### Module 7: On-Call Preparation

**Duration**: 8 hours + 1 week shadowing

**Requirements**:
- Complete all previous modules
- 3 months tenure
- Manager approval
- Pass on-call readiness exam

**Training Components**:
```yaml
preparation:
  - On-call responsibilities
  - Alert triage
  - Runbook usage
  - Escalation procedures
  - Communication protocols

shadowing:
  - Week 1: Shadow primary on-call
  - Week 2: Shadow secondary on-call
  - Week 3: Respond with supervision
  - Week 4: Solo with backup available

certification:
  - Written exam: 50 questions
  - Practical exam: 3 scenarios
  - Shadowing completion
  - Mentor approval
```

**On-Call Certification Exam**:
```
Scenario 1: Database Outage (30 minutes)
- Primary database is down
- Users cannot login
- What do you do?

Scenario 2: Memory Leak (45 minutes)
- Pods restarting every 2 hours
- Memory usage climbing
- How do you troubleshoot?

Scenario 3: Security Alert (30 minutes)
- Suspicious login attempts
- Possible breach
- What are your steps?

Pass score: 85% (All 3 scenarios)
```

---

## Hands-On Labs

### Lab 1: Deploy to Production
```
Objective: Complete a production deployment

Steps:
1. Get code reviewed and merged
2. Wait for CI/CD pipeline
3. Verify staging deployment
4. Deploy to production using canary
5. Monitor deployment
6. Complete post-deployment checklist
7. Document deployment

Time limit: 2 hours
Pass criteria: Successful deployment with zero errors
```

### Lab 2: Troubleshoot Production Issue
```
Objective: Debug and fix a production issue

Scenario: Users reporting slow checkout

Steps:
1. Identify the problem
2. Analyze metrics and logs
3. Form hypothesis
4. Test hypothesis
5. Implement fix
6. Verify resolution
7. Document findings

Time limit: 1 hour
Pass criteria: Correctly identify root cause and propose valid fix
```

### Lab 3: Incident Response Drill
```
Objective: Handle a simulated incident

Scenario: SEV-1 outage

Steps:
1. Acknowledge alert
2. Declare incident
3. Form response team
4. Investigate
5. Implement mitigation
6. Communicate with stakeholders
7. Resolve incident
8. Conduct post-incident review

Time limit: 90 minutes
Pass criteria: Successful resolution within SLA
```

---

## Certification Requirements

### Production Engineer Certification

**Level 1: Associate**
- Complete Modules 1-2
- Pass quizzes (>80%)
- Complete 3 labs
- Valid: 6 months

**Level 2: Professional**
- Hold Level 1 certification
- Complete Modules 3-5
- Pass certification exam (>85%)
- Complete 10 production deployments
- Shadow 5 incidents
- Valid: 1 year

**Level 3: Expert**
- Hold Level 2 certification
- Complete Modules 6-7
- On-call certified
- Lead 3 incident responses
- Mentor 2 engineers
- Valid: 2 years

---

## Training Assessment

### Quiz Example

**Production Operations Quiz (20 questions)**

1. What is the maximum acceptable error rate in production?
   - a) 0%
   - b) 0.1% ✓
   - c) 1%
   - d) 5%

2. When should you rollback a deployment?
   - a) Error rate >1% for 2 minutes ✓
   - b) Error rate >5% for 5 minutes
   - c) Latency >500ms
   - d) CPU >80%

3. What is SEV-1 incident?
   - a) Minor issue
   - b) Service degraded
   - c) Complete outage ✓
   - d) Warning only

[... 17 more questions ...]

Pass score: 16/20 (80%)

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-10
- **Next Review**: Quarterly
- **Owner**: Training Team

*Training materials are continuously updated based on production learnings.*
