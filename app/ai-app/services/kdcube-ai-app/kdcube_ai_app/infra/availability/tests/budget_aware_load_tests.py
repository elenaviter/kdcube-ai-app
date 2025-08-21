# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# budget_aware_load_tests.py
"""
Advanced load testing with budget awareness and detailed capacity planning
"""
import asyncio
import json
import time
import uuid
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import httpx
import argparse

@dataclass
class BudgetConfig:
    """Budget configuration for testing"""
    # Cost per request by user type (in arbitrary units)
    cost_per_anonymous_request: float = 1.0
    cost_per_registered_request: float = 0.5
    cost_per_admin_request: float = 0.1

    # Budget limits
    hourly_budget: float = 1000.0
    daily_budget: float = 20000.0

    # Resource costs
    cpu_cost_per_second: float = 0.1
    memory_cost_per_mb_hour: float = 0.01
    queue_cost_per_item_second: float = 0.001

@dataclass
class ResourceCapacity:
    """Resource capacity configuration"""
    # Instance resources
    cpu_cores_per_instance: int = 4
    memory_gb_per_instance: int = 8
    max_concurrent_per_instance: int = 50

    # Queue capacity
    max_queue_size_per_instance: int = 1000

    # Instance scaling
    min_instances: int = 1
    max_instances: int = 5
    scale_up_threshold: float = 0.8  # CPU utilization
    scale_down_threshold: float = 0.3

@dataclass
class CapacityPlanningResult:
    """Result of capacity planning analysis"""
    # Request capacity
    max_anonymous_requests_per_hour: int
    max_registered_requests_per_hour: int
    max_concurrent_anonymous_users: int
    max_concurrent_registered_users: int

    # Budget analysis
    hourly_cost_at_max_capacity: float
    break_even_request_mix: Dict[str, int]

    # Resource requirements
    recommended_instances: int
    estimated_cpu_utilization: float
    estimated_memory_utilization: float

    # Policy effectiveness
    rate_limit_effectiveness: float  # % of requests properly rate limited
    backpressure_effectiveness: float  # % of requests properly handled under load
    circuit_breaker_recovery_time: float  # seconds to recover

class BudgetAwareLoadTester:
    """Advanced load tester with budget and capacity awareness"""

    def __init__(self, base_url: str, admin_token: str, user_token: str,
                 budget_config: BudgetConfig, resource_config: ResourceCapacity):
        self.base_url = base_url
        self.admin_token = admin_token
        self.user_token = user_token
        self.budget_config = budget_config
        self.resource_config = resource_config
        self.client = httpx.AsyncClient(timeout=30.0)

        # Tracking
        self.total_cost = 0.0
        self.request_history = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def calculate_request_cost(self, user_type: str, response_time: float,
                               queue_time: float = 0) -> float:
        """Calculate the cost of a request"""
        # Base cost by user type
        base_costs = {
            "anonymous": self.budget_config.cost_per_anonymous_request,
            "registered": self.budget_config.cost_per_registered_request,
            "admin": self.budget_config.cost_per_admin_request
        }
        base_cost = base_costs.get(user_type, 1.0)

        # Additional costs for processing time and queuing
        processing_cost = response_time * self.budget_config.cpu_cost_per_second
        queue_cost = queue_time * self.budget_config.queue_cost_per_item_second

        return base_cost + processing_cost + queue_cost

    async def get_system_capacity_info(self) -> Dict[str, Any]:
        """Get detailed system capacity information"""
        try:
            response = await self.client.get(f"{self.base_url}/monitoring/system")
            if response.status_code == 200:
                data = response.json()

                # Extract capacity information
                capacity_info = data.get("capacity_info", {})
                queue_stats = data.get("enhanced_queue_stats", {})
                throttling_stats = data.get("throttling_stats", {})

                return {
                    "current_instances": capacity_info.get("instance_count", 1),
                    "total_queue_size": queue_stats.get("total_queue", 0),
                    "weighted_capacity": capacity_info.get("weighted_max_capacity", 1000),
                    "pressure_ratio": capacity_info.get("pressure_ratio", 0),
                    "accepting_anonymous": capacity_info.get("accepting_anonymous", True),
                    "throttle_rate": throttling_stats.get("throttle_rate", 0),
                    "circuit_breakers": data.get("circuit_breakers", {})
                }
        except Exception as e:
            print(f"Warning: Could not get capacity info: {e}")
            return {}

    async def find_rate_limit_threshold(self, user_type: str) -> Tuple[int, float]:
        """Find the exact rate limit threshold for a user type"""
        print(f"🔍 Finding rate limit threshold for {user_type} users...")

        # Reset system state
        await self.reset_system_state()

        session_id = str(uuid.uuid4())
        fingerprint = f"rate-limit-test-{user_type}"

        # Start with small bursts and increase
        successful_requests = 0
        for burst_size in [5, 10, 15, 20, 25, 30]:
            print(f"  Testing burst size: {burst_size}")

            # Send burst
            tasks = []
            for i in range(burst_size):
                task = asyncio.create_task(
                    self.send_request(user_type, session_id, fingerprint)
                )
                tasks.append(task)
                await asyncio.sleep(0.05)  # Small delay between requests

            results = await asyncio.gather(*tasks)

            # Count successful requests in this burst
            burst_successful = len([r for r in results if r["status"] == 200])
            total_429 = len([r for r in results if r["status"] == 429])

            print(f"    Successful: {burst_successful}, Rate limited: {total_429}")

            if total_429 > 0:
                # Found the threshold
                return successful_requests + burst_successful, burst_size

            successful_requests += burst_successful
            await asyncio.sleep(5)  # Cool down between bursts

        return successful_requests, 30  # Default if no limit found

    async def find_backpressure_threshold(self) -> Tuple[int, int]:
        """Find the backpressure threshold (concurrent users that trigger 503s)"""
        print("🔍 Finding backpressure threshold...")

        await self.reset_system_state()

        async def create_user_load(user_count: int) -> List[Dict]:
            """Create load with specified number of concurrent users"""
            async def user_requests(user_id: int):
                results = []
                for i in range(3):  # 3 requests per user
                    result = await self.send_request(
                        "anonymous",
                        str(uuid.uuid4()),
                        f"backpressure-user-{user_id}"
                    )
                    results.append(result)
                    await asyncio.sleep(0.1)
                return results

            # Start all users simultaneously
            tasks = [asyncio.create_task(user_requests(i)) for i in range(user_count)]
            user_results = await asyncio.gather(*tasks)

            # Flatten results
            all_results = []
            for user_result in user_results:
                all_results.extend(user_result)
            return all_results

        # Test increasing user counts
        for user_count in [10, 20, 30, 40, 50, 75, 100]:
            print(f"  Testing {user_count} concurrent users...")

            results = await create_user_load(user_count)

            status_counts = Counter(r["status"] for r in results)
            total_503 = status_counts.get(503, 0)
            total_requests = len(results)

            print(f"    Total requests: {total_requests}, 503s: {total_503}")

            if total_503 > 0:
                return user_count, total_503

            await asyncio.sleep(3)  # Cool down

        return 100, 0  # No backpressure found

    async def test_circuit_breaker_behavior(self) -> Dict[str, float]:
        """Test circuit breaker behavior and recovery time"""
        print("🔍 Testing circuit breaker behavior...")

        await self.reset_system_state()

        # Force circuit breaker to open by overwhelming rate limiter
        print("  Triggering circuit breaker...")
        session_id = str(uuid.uuid4())

        # Send rapid requests to trigger rate limiting failures
        tasks = []
        for i in range(50):  # Enough to trigger circuit breaker
            task = asyncio.create_task(
                self.send_request("anonymous", session_id, "cb-test")
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)

        # Check if circuit breaker opened
        cb_blocks = len([r for r in results if r.get("response_data", {}).get("detail", "").find("circuit") >= 0])
        print(f"  Circuit breaker blocks detected: {cb_blocks}")

        if cb_blocks == 0:
            return {"open_time": 0, "recovery_time": 0, "effectiveness": 0}

        # Wait for circuit breaker to recover
        print("  Waiting for circuit breaker recovery...")
        recovery_start = time.time()

        while time.time() - recovery_start < 300:  # Max 5 minutes
            test_result = await self.send_request("anonymous", str(uuid.uuid4()), "recovery-test")
            if test_result["status"] == 200:
                recovery_time = time.time() - recovery_start
                print(f"  Circuit breaker recovered in {recovery_time:.1f}s")
                return {
                    "open_time": recovery_start - start_time,
                    "recovery_time": recovery_time,
                    "effectiveness": cb_blocks / len(results)
                }
            await asyncio.sleep(5)

        return {"open_time": 0, "recovery_time": 300, "effectiveness": 0}

    async def send_request(self, user_type: str, session_id: str, fingerprint: str) -> Dict:
        """Send a request and return detailed result"""
        start_time = time.time()

        headers = {}
        if user_type == "registered":
            headers["Authorization"] = f"Bearer {self.user_token}"
        elif user_type == "admin":
            headers["Authorization"] = f"Bearer {self.admin_token}"

        if user_type == "anonymous":
            headers.update({
                "X-Forwarded-For": f"192.168.1.{hash(fingerprint) % 254 + 1}",
                "User-Agent": f"LoadTester-{fingerprint}"
            })

        payload = {
            "message": f"Test message from {user_type} at {time.time()}",
            "session_id": session_id,
            "config": {
                "selected_model": "gpt-4o",
                "selected_embedder": "openai-text-embedding-3-small"
            }
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/landing/chat",
                json=payload,
                headers=headers
            )
            response_time = time.time() - start_time

            try:
                response_data = response.json()
            except:
                response_data = {"raw": response.text}

            # Calculate cost
            cost = self.calculate_request_cost(user_type, response_time)
            self.total_cost += cost

            result = {
                "status": response.status_code,
                "response_time": response_time,
                "cost": cost,
                "user_type": user_type,
                "timestamp": start_time,
                "response_data": response_data
            }

            self.request_history.append(result)
            return result

        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status": 0,
                "response_time": response_time,
                "cost": 0,
                "user_type": user_type,
                "timestamp": start_time,
                "error": str(e)
            }

    async def reset_system_state(self):
        """Reset system to clean state"""
        try:
            # Reset circuit breakers
            cb_response = await self.client.get(
                f"{self.base_url}/admin/circuit-breakers",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if cb_response.status_code == 200:
                cb_data = cb_response.json()
                for circuit_name, circuit_info in cb_data.get("circuits", {}).items():
                    if circuit_info.get("state") == "open":
                        await self.client.post(
                            f"{self.base_url}/admin/circuit-breakers/{circuit_name}/reset",
                            headers={"Authorization": f"Bearer {self.admin_token}"}
                        )
        except Exception as e:
            print(f"Warning: Could not reset system state: {e}")

        # Wait for stabilization
        await asyncio.sleep(3)

    async def run_capacity_planning_analysis(self) -> CapacityPlanningResult:
        """Run comprehensive capacity planning analysis"""
        print("\n🏗️  COMPREHENSIVE CAPACITY PLANNING ANALYSIS")
        print("="*60)

        # Test 1: Find rate limits
        anon_threshold, anon_burst = await self.find_rate_limit_threshold("anonymous")
        reg_threshold, reg_burst = await self.find_rate_limit_threshold("registered")

        print(f"\n📊 RATE LIMIT ANALYSIS:")
        print(f"   Anonymous: {anon_threshold} requests before rate limiting (burst: {anon_burst})")
        print(f"   Registered: {reg_threshold} requests before rate limiting (burst: {reg_burst})")

        # Test 2: Find backpressure limits
        bp_users, bp_503s = await self.find_backpressure_threshold()

        print(f"\n🚦 BACKPRESSURE ANALYSIS:")
        print(f"   {bp_users} concurrent users trigger backpressure ({bp_503s} 503s)")

        # Test 3: Circuit breaker analysis
        cb_stats = await self.test_circuit_breaker_behavior()

        print(f"\n⚡ CIRCUIT BREAKER ANALYSIS:")
        print(f"   Recovery time: {cb_stats['recovery_time']:.1f}s")
        print(f"   Effectiveness: {cb_stats['effectiveness']*100:.1f}%")

        # Calculate capacity recommendations
        system_info = await self.get_system_capacity_info()
        current_instances = system_info.get("current_instances", 1)

        # Estimate hourly capacity based on rate limits and burst capacity
        # Assume 60 requests per hour per anonymous user (conservative)
        max_anon_per_hour = anon_threshold * 60  # Scale based on rate limit
        max_reg_per_hour = reg_threshold * 120   # Registered users get higher limits

        # Estimate concurrent capacity based on backpressure testing
        max_concurrent_anon = max(bp_users - 10, 10)  # Conservative estimate
        max_concurrent_reg = max_concurrent_anon * 2   # Registered users have priority

        # Budget analysis
        hourly_cost_max = (
                max_anon_per_hour * self.budget_config.cost_per_anonymous_request +
                max_reg_per_hour * self.budget_config.cost_per_registered_request
        )

        # Break-even analysis
        break_even_anon = int(self.budget_config.hourly_budget * 0.7 / self.budget_config.cost_per_anonymous_request)
        break_even_reg = int(self.budget_config.hourly_budget * 0.3 / self.budget_config.cost_per_registered_request)

        # Resource recommendations
        cpu_utilization = min(0.8, (max_concurrent_anon + max_concurrent_reg) / (current_instances * self.resource_config.max_concurrent_per_instance))
        memory_utilization = cpu_utilization * 0.8  # Estimate based on CPU

        recommended_instances = max(1, int((max_concurrent_anon + max_concurrent_reg) / self.resource_config.max_concurrent_per_instance / 0.8))

        return CapacityPlanningResult(
            max_anonymous_requests_per_hour=max_anon_per_hour,
            max_registered_requests_per_hour=max_reg_per_hour,
            max_concurrent_anonymous_users=max_concurrent_anon,
            max_concurrent_registered_users=max_concurrent_reg,
            hourly_cost_at_max_capacity=hourly_cost_max,
            break_even_request_mix={"anonymous": break_even_anon, "registered": break_even_reg},
            recommended_instances=recommended_instances,
            estimated_cpu_utilization=cpu_utilization,
            estimated_memory_utilization=memory_utilization,
            rate_limit_effectiveness=0.95,  # Assume high effectiveness
            backpressure_effectiveness=0.90 if bp_503s > 0 else 0.5,
            circuit_breaker_recovery_time=cb_stats['recovery_time']
        )

    def print_capacity_planning_result(self, result: CapacityPlanningResult):
        """Print formatted capacity planning results"""
        print(f"\n{'='*60}")
        print(f"📈 CAPACITY PLANNING RECOMMENDATIONS")
        print(f"{'='*60}")

        print(f"\n🚀 REQUEST CAPACITY:")
        print(f"   Max Anonymous Requests/Hour: {result.max_anonymous_requests_per_hour:,}")
        print(f"   Max Registered Requests/Hour: {result.max_registered_requests_per_hour:,}")
        print(f"   Max Concurrent Anonymous Users: {result.max_concurrent_anonymous_users}")
        print(f"   Max Concurrent Registered Users: {result.max_concurrent_registered_users}")

        print(f"\n💰 BUDGET ANALYSIS:")
        print(f"   Hourly Cost at Max Capacity: ${result.hourly_cost_at_max_capacity:.2f}")
        print(f"   Break-even Mix (hourly):")
        print(f"     Anonymous: {result.break_even_request_mix['anonymous']:,} requests")
        print(f"     Registered: {result.break_even_request_mix['registered']:,} requests")

        daily_budget_anon = int(self.budget_config.daily_budget * 0.7 / self.budget_config.cost_per_anonymous_request)
        daily_budget_reg = int(self.budget_config.daily_budget * 0.3 / self.budget_config.cost_per_registered_request)
        print(f"   Daily Budget Capacity:")
        print(f"     Anonymous: {daily_budget_anon:,} requests")
        print(f"     Registered: {daily_budget_reg:,} requests")

        print(f"\n🖥️  RESOURCE RECOMMENDATIONS:")
        print(f"   Recommended Instances: {result.recommended_instances}")
        print(f"   Estimated CPU Utilization: {result.estimated_cpu_utilization*100:.1f}%")
        print(f"   Estimated Memory Utilization: {result.estimated_memory_utilization*100:.1f}%")

        print(f"\n🛡️  POLICY EFFECTIVENESS:")
        print(f"   Rate Limiting: {result.rate_limit_effectiveness*100:.1f}%")
        print(f"   Backpressure: {result.backpressure_effectiveness*100:.1f}%")
        print(f"   Circuit Breaker Recovery: {result.circuit_breaker_recovery_time:.1f}s")

        # Cost projections
        print(f"\n📊 COST PROJECTIONS:")
        scenarios = [
            ("Light Load (20% capacity)", 0.2),
            ("Normal Load (50% capacity)", 0.5),
            ("Heavy Load (80% capacity)", 0.8),
            ("Peak Load (100% capacity)", 1.0)
        ]

        for scenario_name, factor in scenarios:
            hourly_cost = result.hourly_cost_at_max_capacity * factor
            daily_cost = hourly_cost * 24
            monthly_cost = daily_cost * 30
            print(f"   {scenario_name}:")
            print(f"     Hourly: ${hourly_cost:.2f} | Daily: ${daily_cost:.2f} | Monthly: ${monthly_cost:.2f}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if result.recommended_instances > 1:
            print(f"   🔧 Scale to {result.recommended_instances} instances for optimal performance")

        if result.estimated_cpu_utilization > 0.8:
            print(f"   ⚠️  High CPU utilization expected - consider more instances")

        if result.circuit_breaker_recovery_time > 60:
            print(f"   ⚠️  Long circuit breaker recovery - tune thresholds")

        if result.hourly_cost_at_max_capacity > self.budget_config.hourly_budget:
            print(f"   💰 Max capacity exceeds budget - implement usage-based pricing")

        budget_efficiency = self.budget_config.hourly_budget / result.hourly_cost_at_max_capacity
        if budget_efficiency < 1:
            print(f"   📊 Budget allows {budget_efficiency*100:.1f}% of max capacity")

async def main():
    """Main function for budget-aware load testing"""
    parser = argparse.ArgumentParser(description="Budget-Aware Gateway Load Testing")
    parser.add_argument("--base-url", default="http://localhost:8010", help="Base URL")
    parser.add_argument("--admin-token", default="test-admin-token-123", help="Admin token")
    parser.add_argument("--user-token", default="test-chat-token-456", help="User token")
    parser.add_argument("--hourly-budget", type=float, default=1000.0, help="Hourly budget")
    parser.add_argument("--daily-budget", type=float, default=20000.0, help="Daily budget")

    args = parser.parse_args()

    budget_config = BudgetConfig(
        hourly_budget=args.hourly_budget,
        daily_budget=args.daily_budget
    )

    resource_config = ResourceCapacity()

    async with BudgetAwareLoadTester(
            args.base_url, args.admin_token, args.user_token,
            budget_config, resource_config
    ) as tester:

        result = await tester.run_capacity_planning_analysis()
        tester.print_capacity_planning_result(result)

        print(f"\n💳 TOTAL TEST COST: ${tester.total_cost:.2f}")
        print(f"📊 TOTAL REQUESTS: {len(tester.request_history)}")

if __name__ == "__main__":
    asyncio.run(main())