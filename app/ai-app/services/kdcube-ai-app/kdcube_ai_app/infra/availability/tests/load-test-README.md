# Gateway Load Testing Guide

## Overview

This comprehensive load testing suite helps you understand your gateway's capacity, budget requirements, and policy effectiveness. The tests are designed to work with your existing authentication, rate limiting, backpressure, and circuit breaker systems.

## Test Suite Components

### 1. 📊 Comprehensive Load Tests (`comprehensive_load_tests.py`)
Basic load testing with detailed analysis of gateway performance.

### 2. 💰 Budget-Aware Load Tests (`budget_aware_load_tests.py`)
Advanced testing with cost analysis and capacity planning for budget-conscious deployments.

### 3. 🛡️ Policy Testing Suite (`policy_testing_suite.py`)
Specialized tests for validating and tuning gateway policies.

## Quick Start

### Prerequisites
```bash
pip install httpx asyncio
```

### Basic Usage

1. **Test current capacity:**
```bash
python comprehensive_load_tests.py --test capacity
```

2. **Budget-aware capacity planning:**
```bash
python budget_aware_load_tests.py --hourly-budget 1000 --daily-budget 20000
```

3. **Test and reset policies:**
```bash
python policy_testing_suite.py --test all
```

## Test Types Explained

### 🔥 Single User Burst Test
**Purpose:** Find rate limiting thresholds for individual users.

**What it does:**
- Sends rapid requests from single user/session
- Identifies when rate limiting kicks in
- Measures recovery time

**Interpretation:**
- ✅ **Good:** Rate limiting triggers after 5-10 requests for anonymous, 15-25 for registered
- ⚠️ **Concerning:** No rate limiting after 50+ requests
- 🔧 **Action:** Adjust `ANON_BURST_LIMIT` and `REG_BURST_LIMIT` in your config

### 👥 Concurrent Users Test
**Purpose:** Find backpressure and queue capacity limits.

**What it does:**
- Simulates multiple users making requests simultaneously
- Identifies when 503 errors start appearing
- Tests queue management

**Interpretation:**
- ✅ **Good:** 503s start appearing at reasonable load (20-50 concurrent users)
- ⚠️ **Concerning:** System accepts unlimited load or fails immediately
- 🔧 **Action:** Adjust `MAX_QUEUE_SIZE` and instance scaling

### 🌐 Mixed Load Test
**Purpose:** Test realistic usage patterns with different user types.

**What it does:**
- Runs anonymous, registered, and admin users simultaneously
- Tests priority handling
- Measures system behavior under mixed load

**Interpretation:**
- ✅ **Good:** Admin users get priority, registered users perform better than anonymous
- ⚠️ **Concerning:** No differentiation between user types
- 🔧 **Action:** Review rate limiting configurations and queue prioritization

### ⚡ Circuit Breaker Test
**Purpose:** Validate circuit breaker functionality and recovery.

**What it does:**
- Triggers circuit breaker by overwhelming rate limiter
- Tests fail-fast behavior
- Measures recovery time

**Interpretation:**
- ✅ **Good:** Circuit breaker opens after sustained failures, recovers within 1-5 minutes
- ⚠️ **Concerning:** Circuit breaker never opens or takes too long to recover
- 🔧 **Action:** Tune circuit breaker thresholds and recovery timeouts

## Key Metrics Explained

### 📈 Performance Metrics

**Requests per Second (RPS):**
- **Anonymous:** 5-20 RPS per instance typically sustainable
- **Registered:** 10-50 RPS per instance typically sustainable
- **Admin:** 50+ RPS per instance typically sustainable

**Response Times:**
- **Good:** < 500ms average, < 2s 95th percentile
- **Acceptable:** < 1s average, < 5s 95th percentile
- **Poor:** > 2s average, > 10s 95th percentile

**Error Rates:**
- **Excellent:** < 1% error rate under normal load
- **Good:** < 5% error rate under peak load
- **Poor:** > 10% error rate

### 🛡️ Policy Effectiveness

**Rate Limiting:**
- **Effective:** 95%+ of excess requests get 429 status
- **Ineffective:** < 80% proper rate limiting

**Backpressure:**
- **Effective:** 503s appear before system crashes
- **Ineffective:** System becomes unresponsive without 503s

**Circuit Breaker:**
- **Effective:** Opens within 30s of sustained failures, recovers within 5 minutes
- **Ineffective:** Never opens or takes > 10 minutes to recover

### 💰 Budget Analysis

**Cost per Request Types:**
- **Anonymous:** Highest cost (full processing + higher resource usage)
- **Registered:** Medium cost (cached auth + user history)
- **Admin:** Lowest cost (privileged processing + minimal validation)

**Budget Efficiency:**
- **Good:** 70%+ of budget goes to successful requests
- **Poor:** < 50% budget efficiency due to failed/throttled requests

## Capacity Planning Results

### 🎯 What the Results Tell You

**Request Capacity:**
```
Max Anonymous Requests/Hour: 2,400
Max Registered Requests/Hour: 12,000
Max Concurrent Anonymous Users: 30
Max Concurrent Registered Users: 60
```

**Interpretation:**
- Your system can handle 2,400 anonymous requests per hour
- Registered users get 5x higher limits
- Beyond 30 concurrent anonymous users, expect backpressure

**Resource Recommendations:**
```
Recommended Instances: 3
Estimated CPU Utilization: 75%
Estimated Memory Utilization: 60%
```

**Interpretation:**
- Scale to 3 instances for optimal performance
- Current configuration will use 75% CPU at peak load
- Memory usage is healthy

## Common Issues and Solutions

### ❌ No Rate Limiting Observed
**Symptoms:** Tests send 50+ requests without 429 responses
**Causes:**
- Rate limits set too high
- Rate limiting disabled
- Token bucket not working

**Solutions:**
- Lower `ANON_BURST_LIMIT` to 5-10
- Lower `REG_BURST_LIMIT` to 15-25
- Check Redis connectivity for token buckets

### ❌ Immediate 503 Errors
**Symptoms:** 503 errors with minimal load
**Causes:**
- Queue size too small
- Not enough instances
- Circuit breaker too sensitive

**Solutions:**
- Increase `MAX_QUEUE_SIZE`
- Scale up instances
- Increase circuit breaker failure threshold

### ❌ No User Type Differentiation
**Symptoms:** Anonymous and registered users get same treatment
**Causes:**
- Authentication not working
- Same rate limits for all users
- Queue priority not implemented

**Solutions:**
- Verify authentication tokens
- Check rate limit configuration per user type
- Implement queue prioritization

### ❌ Circuit Breaker Never Opens
**Symptoms:** System becomes unresponsive but circuit breaker stays closed
**Causes:**
- Failure threshold too high
- Wrong failure detection
- Circuit breaker disabled

**Solutions:**
- Lower failure threshold to 5-10
- Check failure detection logic
- Verify circuit breaker integration

## Budget Planning Examples

### Small Deployment (Single Instance)
```
Hourly Budget: $100
Recommended Mix:
- Anonymous: 800 requests/hour
- Registered: 1,200 requests/hour
- Estimated Cost: $95/hour
```

### Medium Deployment (3 Instances)
```
Hourly Budget: $500
Recommended Mix:
- Anonymous: 2,400 requests/hour
- Registered: 6,000 requests/hour
- Estimated Cost: $475/hour
```

### Large Deployment (5 Instances)
```
Hourly Budget: $1,000
Recommended Mix:
- Anonymous: 4,000 requests/hour
- Registered: 12,000 requests/hour
- Estimated Cost: $950/hour
```

## Monitoring Dashboard Integration

The load tests provide data that integrates with your monitoring dashboard:

**Real-time Metrics to Watch:**
- Queue utilization percentage
- Circuit breaker states
- Throttle rate percentage
- Instance health scores

**Alerts to Configure:**
- Queue utilization > 80%
- Any circuit breakers open
- Throttle rate > 10%
- Error rate > 5%

## Regular Testing Schedule

### Daily Tests
- Quick burst test for each user type (5 minutes)
- Monitor for configuration drift

### Weekly Tests
- Full concurrent user test (15 minutes)
- Capacity planning validation

### Monthly Tests
- Complete policy test suite (45 minutes)
- Budget analysis and optimization
- Circuit breaker recovery validation

### After Changes
- Always run policy tests after configuration changes
- Validate capacity after infrastructure changes
- Test user experience after auth system updates

## Best Practices

1. **Always reset before testing:** Use the reset functionality to ensure clean test conditions
2. **Test during off-peak hours:** Avoid impacting real users
3. **Save test results:** Keep historical data for trend analysis
4. **Test incrementally:** Don't jump from 10 to 100 users immediately
5. **Monitor system health:** Watch CPU, memory, and queue metrics during tests
6. **Document changes:** Record configuration changes and their impact
7. **Regular validation:** Rerun tests after any system changes

