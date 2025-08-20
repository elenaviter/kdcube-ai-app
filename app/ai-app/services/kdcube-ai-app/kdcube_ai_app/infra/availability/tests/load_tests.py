# load_tests.py
"""
Comprehensive load testing suite for gateway system
Tests rate limiting, backpressure, circuit breakers, and capacity planning
"""
import asyncio
import json
import time
import uuid
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import httpx
import argparse

# Configuration
BASE_URL = "http://localhost:8010"
CHAT_ENDPOINT = f"{BASE_URL}/landing/chat"
MONITORING_ENDPOINT = f"{BASE_URL}/monitoring/system"
CIRCUIT_BREAKER_ENDPOINT = f"{BASE_URL}/admin/circuit-breakers"

# Test tokens (replace with your actual test tokens)
ADMIN_TOKEN = "test-admin-token-123"
CHAT_USER_TOKEN = "test-chat-token-456"

@dataclass
class TestConfig:
    """Configuration for load tests"""
    base_url: str = BASE_URL
    admin_token: str = ADMIN_TOKEN
    chat_user_token: str = CHAT_USER_TOKEN
    request_timeout: float = 30.0
    monitor_interval: float = 1.0

@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: str
    user_type: str
    status_code: int
    response_time: float
    timestamp: float
    error: Optional[str] = None
    response_data: Optional[Dict] = None
    fingerprint: Optional[str] = None

@dataclass
class TestResult:
    """Result of a complete test"""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    status_codes: Dict[int, int]
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    requests_per_second: float
    error_rate: float
    rate_limited_requests: int
    backpressure_requests: int
    circuit_breaker_blocks: int
    system_stats_before: Optional[Dict] = None
    system_stats_after: Optional[Dict] = None

class GatewayLoadTester:
    """Main load testing class"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.request_timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def reset_system_state(self) -> Dict[str, Any]:
        """Reset circuit breakers and get baseline system state"""
        print("🔄 Resetting system state...")

        # Get current circuit breaker status
        try:
            cb_response = await self.client.get(
                f"{self.config.base_url}/admin/circuit-breakers",
                headers={"Authorization": f"Bearer {self.config.admin_token}"}
            )
            if cb_response.status_code == 200:
                cb_data = cb_response.json()

                # Reset any open circuit breakers
                for circuit_name, circuit_info in cb_data.get("circuits", {}).items():
                    if circuit_info.get("state") == "open":
                        print(f"  Resetting circuit breaker: {circuit_name}")
                        await self.client.post(
                            f"{self.config.base_url}/admin/circuit-breakers/{circuit_name}/reset",
                            headers={"Authorization": f"Bearer {self.config.admin_token}"}
                        )
        except Exception as e:
            print(f"  Warning: Could not reset circuit breakers: {e}")

        # Wait for system to stabilize
        await asyncio.sleep(2)

        # Get baseline system stats
        try:
            monitor_response = await self.client.get(MONITORING_ENDPOINT,
                                                     headers={"Authorization": f"Bearer {self.config.admin_token}"})
            if monitor_response.status_code == 200:
                return monitor_response.json()
        except Exception as e:
            print(f"  Warning: Could not get baseline stats: {e}")

        return {}

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            response = await self.client.get(MONITORING_ENDPOINT,
                                             headers={"Authorization": f"Bearer {self.config.admin_token}"})
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Warning: Could not get system stats: {e}")
        return {}

    async def send_request(self,
                           user_type: str = "anonymous",
                           session_id: Optional[str] = None,
                           fingerprint_suffix: str = "",
                           message: Optional[str] = None) -> RequestResult:
        """Send a single request"""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Prepare headers based on user type
        headers = {}
        if user_type == "registered":
            headers["Authorization"] = f"Bearer {self.config.chat_user_token}"
        elif user_type == "admin":
            headers["Authorization"] = f"Bearer {self.config.admin_token}"

        # Create unique fingerprint for anonymous users
        if user_type == "anonymous":
            fake_ip = f"192.168.1.{hash(fingerprint_suffix) % 254 + 1}"
            user_agent = f"LoadTester-{fingerprint_suffix}"
            headers.update({
                "X-Forwarded-For": fake_ip,
                "User-Agent": user_agent
            })

        # Prepare payload
        payload = {
            "message": message or f"Load test message from {user_type} at {time.time()}",
            "session_id": session_id or str(uuid.uuid4()),
            "config": {
                "selected_model": "gpt-4o",
                "selected_embedder": "openai-text-embedding-3-small"
            }
        }

        try:
            response = await self.client.post(CHAT_ENDPOINT, json=payload, headers=headers)
            response_time = time.time() - start_time

            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}

            return RequestResult(
                request_id=request_id,
                user_type=user_type,
                status_code=response.status_code,
                response_time=response_time,
                timestamp=start_time,
                response_data=response_data,
                fingerprint=fingerprint_suffix if user_type == "anonymous" else None
            )

        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                request_id=request_id,
                user_type=user_type,
                status_code=0,
                response_time=response_time,
                timestamp=start_time,
                error=str(e),
                fingerprint=fingerprint_suffix if user_type == "anonymous" else None
            )

    def analyze_results(self, test_name: str, results: List[RequestResult],
                        duration: float, system_before: Dict, system_after: Dict) -> TestResult:
        """Analyze test results and generate report"""
        total_requests = len(results)
        successful_requests = len([r for r in results if 200 <= r.status_code < 300])

        # Status code analysis
        status_codes = Counter(r.status_code for r in results)

        # Response time analysis
        response_times = [r.response_time for r in results if r.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max_response_time

        # Rate analysis
        requests_per_second = total_requests / duration if duration > 0 else 0
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0

        # Policy analysis
        rate_limited_requests = status_codes.get(429, 0)
        backpressure_requests = status_codes.get(503, 0)
        circuit_breaker_blocks = len([r for r in results if r.response_data and
                                      "circuit_breaker" in str(r.response_data)])

        return TestResult(
            test_name=test_name,
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            status_codes=dict(status_codes),
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            rate_limited_requests=rate_limited_requests,
            backpressure_requests=backpressure_requests,
            circuit_breaker_blocks=circuit_breaker_blocks,
            system_stats_before=system_before,
            system_stats_after=system_after
        )

    def print_test_result(self, result: TestResult):
        """Print formatted test results"""
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS: {result.test_name}")
        print(f"{'='*60}")

        # Basic metrics
        print(f"⏱️  Duration: {result.duration:.2f}s")
        print(f"📨 Total Requests: {result.total_requests}")
        print(f"✅ Successful: {result.successful_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
        print(f"❌ Error Rate: {result.error_rate*100:.1f}%")
        print(f"🚀 Requests/sec: {result.requests_per_second:.2f}")

        # Response times
        print(f"\n⏱️  RESPONSE TIMES:")
        print(f"   Average: {result.avg_response_time*1000:.0f}ms")
        print(f"   Min: {result.min_response_time*1000:.0f}ms")
        print(f"   Max: {result.max_response_time*1000:.0f}ms")
        print(f"   95th percentile: {result.p95_response_time*1000:.0f}ms")

        # Status codes
        print(f"\n📋 STATUS CODES:")
        for status, count in sorted(result.status_codes.items()):
            percentage = count / result.total_requests * 100
            status_name = {
                200: "OK",
                429: "Rate Limited",
                503: "Service Unavailable",
                401: "Unauthorized",
                403: "Forbidden",
                500: "Internal Error"
            }.get(status, "Other")
            print(f"   {status} ({status_name}): {count} ({percentage:.1f}%)")

        # Policy enforcement
        print(f"\n🛡️  POLICY ENFORCEMENT:")
        print(f"   Rate Limited (429): {result.rate_limited_requests}")
        print(f"   Backpressure (503): {result.backpressure_requests}")
        print(f"   Circuit Breaker Blocks: {result.circuit_breaker_blocks}")

        # System stats comparison
        if result.system_stats_before and result.system_stats_after:
            self._print_system_comparison(result.system_stats_before, result.system_stats_after)

    def _print_system_comparison(self, before: Dict, after: Dict):
        """Print system statistics comparison"""
        print(f"\n🖥️  SYSTEM IMPACT:")

        # Queue stats
        queue_before = before.get("queue_stats", {})
        queue_after = after.get("queue_stats", {})

        for queue_type in ["anonymous", "registered", "privileged"]:
            before_size = queue_before.get(queue_type, 0)
            after_size = queue_after.get(queue_type, 0)
            delta = after_size - before_size
            print(f"   {queue_type.title()} Queue: {before_size} → {after_size} (Δ{delta:+d})")

        # Throttling stats
        throttling_before = before.get("throttling_stats", {})
        throttling_after = after.get("throttling_stats", {})

        total_before = throttling_before.get("total_requests", 0)
        total_after = throttling_after.get("total_requests", 0)
        throttled_before = throttling_before.get("total_throttled", 0)
        throttled_after = throttling_after.get("total_throttled", 0)

        print(f"   Total Requests: {total_before} → {total_after} (Δ{total_after - total_before:+d})")
        print(f"   Total Throttled: {throttled_before} → {throttled_after} (Δ{throttled_after - throttled_before:+d})")

        # Circuit breaker stats
        if "circuit_breakers" in after:
            cb_stats = after["circuit_breakers"]
            summary = cb_stats.get("summary", {})
            print(f"   Circuit Breakers: {summary.get('closed_circuits', 0)} closed, {summary.get('open_circuits', 0)} open")

    async def test_single_user_burst(self, user_type: str = "anonymous",
                                     burst_size: int = 10, burst_delay: float = 0.1) -> TestResult:
        """Test burst from single user to trigger rate limiting"""
        print(f"\n🔥 Testing single {user_type} user burst ({burst_size} requests)")

        system_before = await self.reset_system_state()
        session_id = str(uuid.uuid4())
        fingerprint = str(uuid.uuid4())[:8]

        start_time = time.time()

        # Send burst of requests
        tasks = []
        for i in range(burst_size):
            task = asyncio.create_task(
                self.send_request(user_type=user_type, session_id=session_id,
                                  fingerprint_suffix=fingerprint)
            )
            tasks.append(task)
            if burst_delay > 0:
                await asyncio.sleep(burst_delay)

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        system_after = await self.get_system_stats()

        return self.analyze_results(f"Single {user_type} burst", results, duration,
                                    system_before, system_after)

    async def test_concurrent_users(self, user_type: str = "anonymous",
                                    num_users: int = 20, requests_per_user: int = 5,
                                    ramp_up_time: float = 2.0) -> TestResult:
        """Test concurrent users to trigger backpressure"""
        print(f"\n👥 Testing {num_users} concurrent {user_type} users ({requests_per_user} req each)")

        system_before = await self.reset_system_state()

        async def user_session(user_id: int):
            """Simulate one user's session"""
            session_id = str(uuid.uuid4())
            fingerprint = f"user-{user_id}"
            results = []

            for req_num in range(requests_per_user):
                result = await self.send_request(
                    user_type=user_type,
                    session_id=session_id,
                    fingerprint_suffix=fingerprint,
                    message=f"Request {req_num+1} from user {user_id}"
                )
                results.append(result)
                # Small delay between requests from same user
                await asyncio.sleep(0.1)

            return results

        start_time = time.time()

        # Ramp up users gradually
        user_tasks = []
        for user_id in range(num_users):
            task = asyncio.create_task(user_session(user_id))
            user_tasks.append(task)
            if ramp_up_time > 0:
                await asyncio.sleep(ramp_up_time / num_users)

        # Wait for all users to complete
        user_results = await asyncio.gather(*user_tasks)

        # Flatten results
        all_results = []
        for user_result in user_results:
            all_results.extend(user_result)

        duration = time.time() - start_time
        system_after = await self.get_system_stats()

        return self.analyze_results(f"{num_users} concurrent {user_type} users",
                                    all_results, duration, system_before, system_after)

    async def test_mixed_load(self, anon_users: int = 10, reg_users: int = 5,
                              admin_users: int = 2, duration_seconds: float = 30.0) -> TestResult:
        """Test mixed load with different user types"""
        print(f"\n🌐 Testing mixed load for {duration_seconds}s (A:{anon_users}, R:{reg_users}, Ad:{admin_users})")

        system_before = await self.reset_system_state()
        results = []

        async def continuous_user(user_type: str, user_id: int, stop_event: asyncio.Event):
            """Continuously send requests until stopped"""
            session_id = str(uuid.uuid4())
            fingerprint = f"{user_type}-{user_id}"
            request_count = 0

            while not stop_event.is_set():
                result = await self.send_request(
                    user_type=user_type,
                    session_id=session_id,
                    fingerprint_suffix=fingerprint,
                    message=f"Continuous request {request_count} from {user_type} {user_id}"
                )
                results.append(result)
                request_count += 1

                # Different request rates for different user types
                delay = {
                    "anonymous": 2.0,    # Slower for anonymous
                    "registered": 1.0,   # Medium for registered
                    "admin": 0.5        # Faster for admin
                }.get(user_type, 1.0)

                await asyncio.sleep(delay)

        # Start all users
        stop_event = asyncio.Event()
        tasks = []

        for i in range(anon_users):
            task = asyncio.create_task(continuous_user("anonymous", i, stop_event))
            tasks.append(task)

        for i in range(reg_users):
            task = asyncio.create_task(continuous_user("registered", i, stop_event))
            tasks.append(task)

        for i in range(admin_users):
            task = asyncio.create_task(continuous_user("admin", i, stop_event))
            tasks.append(task)

        start_time = time.time()

        # Run for specified duration
        await asyncio.sleep(duration_seconds)

        # Stop all users
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time
        system_after = await self.get_system_stats()

        return self.analyze_results("Mixed load test", results, duration,
                                    system_before, system_after)

    async def test_capacity_planning(self) -> Dict[str, TestResult]:
        """Run comprehensive capacity planning tests"""
        print(f"\n📊 COMPREHENSIVE CAPACITY PLANNING TESTS")
        print(f"{'='*60}")

        results = {}

        # Test 1: Single user burst limits
        for user_type in ["anonymous", "registered"]:
            result = await self.test_single_user_burst(user_type=user_type, burst_size=20)
            self.print_test_result(result)
            results[f"burst_{user_type}"] = result
            await asyncio.sleep(5)  # Cool down between tests

        # Test 2: Concurrent user limits
        for user_type in ["anonymous", "registered"]:
            num_users = 30 if user_type == "anonymous" else 15
            result = await self.test_concurrent_users(user_type=user_type, num_users=num_users)
            self.print_test_result(result)
            results[f"concurrent_{user_type}"] = result
            await asyncio.sleep(5)

        # Test 3: Mixed load test
        result = await self.test_mixed_load()
        self.print_test_result(result)
        results["mixed_load"] = result

        return results

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Gateway Load Testing Suite")
    parser.add_argument("--test", choices=[
        "burst-anon", "burst-reg", "concurrent-anon", "concurrent-reg",
        "mixed", "capacity", "all"
    ], default="all", help="Test to run")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL for testing")
    parser.add_argument("--admin-token", default=ADMIN_TOKEN, help="Admin token")
    parser.add_argument("--user-token", default=CHAT_USER_TOKEN, help="User token")

    args = parser.parse_args()

    config = TestConfig(
        base_url=args.base_url,
        admin_token=args.admin_token,
        chat_user_token=args.user_token
    )

    async with GatewayLoadTester(config) as tester:
        if args.test == "burst-anon":
            result = await tester.test_single_user_burst("anonymous")
            tester.print_test_result(result)
        elif args.test == "burst-reg":
            result = await tester.test_single_user_burst("registered")
            tester.print_test_result(result)
        elif args.test == "concurrent-anon":
            result = await tester.test_concurrent_users("anonymous")
            tester.print_test_result(result)
        elif args.test == "concurrent-reg":
            result = await tester.test_concurrent_users("registered")
            tester.print_test_result(result)
        elif args.test == "mixed":
            result = await tester.test_mixed_load()
            tester.print_test_result(result)
        elif args.test == "capacity":
            await tester.test_capacity_planning()
        elif args.test == "all":
            await tester.test_capacity_planning()

if __name__ == "__main__":
    asyncio.run(main())