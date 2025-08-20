# gateway_policy_tests.py
"""
Specialized test suite for testing and resetting gateway policies
"""
import asyncio
import json
import time
import uuid
from typing import Dict, List, Tuple, Optional
import httpx
import argparse
from datetime import datetime

class PolicyTester:
    """Test and manage gateway policies"""

    def __init__(self, base_url: str, admin_token: str, user_token: str):
        self.base_url = base_url
        self.admin_token = admin_token
        self.user_token = user_token
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def reset_all_policies(self) -> Dict[str, bool]:
        """Reset all gateway policies to clean state"""
        print("🔄 Resetting all gateway policies...")

        results = {}

        # Reset circuit breakers
        try:
            cb_response = await self.client.get(
                f"{self.base_url}/admin/circuit-breakers",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )

            if cb_response.status_code == 200:
                cb_data = cb_response.json()
                circuits = cb_data.get("circuits", {})

                for circuit_name, circuit_info in circuits.items():
                    if circuit_info.get("state") in ["open", "half_open"]:
                        print(f"  Resetting circuit breaker: {circuit_name}")
                        reset_response = await self.client.post(
                            f"{self.base_url}/admin/circuit-breakers/{circuit_name}/reset",
                            headers={"Authorization": f"Bearer {self.admin_token}"}
                        )
                        results[f"circuit_breaker_{circuit_name}"] = reset_response.status_code == 200
                    else:
                        results[f"circuit_breaker_{circuit_name}"] = True
            else:
                print(f"  Warning: Could not get circuit breaker status: {cb_response.status_code}")
                results["circuit_breakers"] = False

        except Exception as e:
            print(f"  Error resetting circuit breakers: {e}")
            results["circuit_breakers"] = False

        # Wait for rate limit windows to reset (this takes time)
        print("  Rate limit buckets will reset naturally over time")
        results["rate_limits"] = True

        # Queue should drain naturally
        print("  Queues will drain naturally")
        results["queues"] = True

        # Wait for system stabilization
        print("  Waiting for system stabilization...")
        await asyncio.sleep(5)

        return results

    async def get_system_state(self) -> Dict[str, any]:
        """Get current system state"""
        try:
            response = await self.client.get(f"{self.base_url}/monitoring/system")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting system state: {e}")
        return {}

    async def test_rate_limiting_policy(self, user_type: str = "anonymous") -> Dict[str, any]:
        """Test rate limiting policy specifically"""
        print(f"\n🚦 Testing rate limiting policy for {user_type} users")

        session_id = str(uuid.uuid4())
        fingerprint = f"rate-test-{user_type}-{int(time.time())}"

        results = {
            "user_type": user_type,
            "requests_sent": 0,
            "successful_requests": 0,
            "rate_limited_requests": 0,
            "first_rate_limit_at": None,
            "recovery_time": None
        }

        # Send requests until we hit rate limit
        for i in range(50):  # Max 50 requests
            result = await self.send_test_request(user_type, session_id, fingerprint)
            results["requests_sent"] += 1

            if result["status"] == 200:
                results["successful_requests"] += 1
                print(f"  Request {i+1}: ✅ Success")
            elif result["status"] == 429:
                results["rate_limited_requests"] += 1
                if results["first_rate_limit_at"] is None:
                    results["first_rate_limit_at"] = i + 1
                print(f"  Request {i+1}: 🚫 Rate limited")
                break
            else:
                print(f"  Request {i+1}: ❌ Error {result['status']}")

            await asyncio.sleep(0.1)  # Small delay between requests

        # Test recovery
        if results["rate_limited_requests"] > 0:
            print("  Testing rate limit recovery...")
            recovery_start = time.time()

            # Wait and test recovery (rate limits typically reset after 1 minute)
            for wait_time in [10, 30, 60, 120]:
                print(f"    Waiting {wait_time}s...")
                await asyncio.sleep(wait_time - (10 if wait_time > 10 else 0))

                recovery_result = await self.send_test_request(user_type, str(uuid.uuid4()), f"recovery-{wait_time}")
                if recovery_result["status"] == 200:
                    results["recovery_time"] = time.time() - recovery_start
                    print(f"    ✅ Recovered after {results['recovery_time']:.1f}s")
                    break
                else:
                    print(f"    🚫 Still rate limited")

        return results

    async def test_backpressure_policy(self) -> Dict[str, any]:
        """Test backpressure policy"""
        print("\n🌊 Testing backpressure policy")

        results = {
            "concurrent_users_tested": 0,
            "backpressure_triggered_at": None,
            "total_503_responses": 0,
            "anonymous_blocked": False
        }

        # Gradually increase concurrent load
        for user_count in [5, 10, 20, 30, 50]:
            print(f"  Testing with {user_count} concurrent users...")

            async def user_load():
                return await self.send_test_request("anonymous", str(uuid.uuid4()), f"bp-test-{uuid.uuid4()}")

            # Send concurrent requests
            tasks = [asyncio.create_task(user_load()) for _ in range(user_count)]
            user_results = await asyncio.gather(*tasks)

            # Analyze results
            status_counts = {}
            for result in user_results:
                status = result["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

            results["concurrent_users_tested"] = user_count
            current_503s = status_counts.get(503, 0)
            results["total_503_responses"] += current_503s

            print(f"    Results: {dict(status_counts)}")

            if current_503s > 0 and results["backpressure_triggered_at"] is None:
                results["backpressure_triggered_at"] = user_count
                print(f"    🌊 Backpressure triggered at {user_count} users")

            # Check if anonymous users are being blocked
            if current_503s > 0:
                results["anonymous_blocked"] = True

            await asyncio.sleep(2)  # Cool down between tests

        return results

    async def test_circuit_breaker_policy(self) -> Dict[str, any]:
        """Test circuit breaker policy"""
        print("\n⚡ Testing circuit breaker policy")

        results = {
            "circuit_breaker_triggered": False,
            "trigger_method": None,
            "requests_to_trigger": 0,
            "recovery_time": None,
            "effectiveness": 0.0
        }

        session_id = str(uuid.uuid4())
        fingerprint = f"cb-test-{int(time.time())}"

        print("  Attempting to trigger circuit breaker via rate limit failures...")

        # Method 1: Trigger via rapid rate limit violations
        for i in range(100):
            result = await self.send_test_request("anonymous", session_id, fingerprint)
            results["requests_to_trigger"] += 1

            # Check if this is a circuit breaker response
            response_data = result.get("response_data", {})
            if (result["status"] == 503 and
                    "circuit" in str(response_data.get("detail", "")).lower()):
                results["circuit_breaker_triggered"] = True
                results["trigger_method"] = "rate_limit_failures"
                print(f"    ⚡ Circuit breaker triggered after {results['requests_to_trigger']} requests")
                break

            # Small delay to create rapid requests
            await asyncio.sleep(0.05)

        # Test circuit breaker effectiveness
        if results["circuit_breaker_triggered"]:
            print("  Testing circuit breaker effectiveness...")
            blocked_requests = 0

            # Send more requests - they should be blocked quickly
            for i in range(10):
                result = await self.send_test_request("anonymous", str(uuid.uuid4()), f"cb-effectiveness-{i}")
                response_data = result.get("response_data", {})
                if (result["status"] == 503 and
                        "circuit" in str(response_data.get("detail", "")).lower()):
                    blocked_requests += 1

            results["effectiveness"] = blocked_requests / 10
            print(f"    Circuit breaker blocked {blocked_requests}/10 requests ({results['effectiveness']*100:.1f}%)")

            # Test recovery
            print("  Testing circuit breaker recovery...")
            recovery_start = time.time()

            for wait_minutes in [1, 2, 3, 5]:
                print(f"    Waiting {wait_minutes} minute(s)...")
                await asyncio.sleep(wait_minutes * 60)

                # Test if circuit breaker has recovered
                recovery_result = await self.send_test_request("anonymous", str(uuid.uuid4()), "recovery-test")
                if recovery_result["status"] == 200:
                    results["recovery_time"] = time.time() - recovery_start
                    print(f"    ✅ Circuit breaker recovered after {results['recovery_time']/60:.1f} minutes")
                    break
                else:
                    print(f"    🚫 Circuit breaker still open")

        return results

    async def send_test_request(self, user_type: str, session_id: str, fingerprint: str) -> Dict:
        """Send a test request"""
        headers = {}
        if user_type == "registered":
            headers["Authorization"] = f"Bearer {self.user_token}"
        elif user_type == "admin":
            headers["Authorization"] = f"Bearer {self.admin_token}"

        if user_type == "anonymous":
            headers.update({
                "X-Forwarded-For": f"192.168.1.{hash(fingerprint) % 254 + 1}",
                "User-Agent": f"PolicyTester-{fingerprint}"
            })

        payload = {
            "message": f"Policy test from {user_type} at {time.time()}",
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

            try:
                response_data = response.json()
            except:
                response_data = {"raw": response.text}

            return {
                "status": response.status_code,
                "response_data": response_data
            }

        except Exception as e:
            return {
                "status": 0,
                "error": str(e)
            }

    async def run_comprehensive_policy_test(self) -> Dict[str, any]:
        """Run comprehensive test of all policies"""
        print("🧪 COMPREHENSIVE POLICY TEST SUITE")
        print("="*50)

        # Reset system first
        reset_results = await self.reset_all_policies()
        print(f"System reset results: {reset_results}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "reset_results": reset_results,
            "rate_limiting": {},
            "backpressure": {},
            "circuit_breaker": {},
            "system_state_before": await self.get_system_state(),
            "system_state_after": None
        }

        # Test 1: Rate Limiting
        for user_type in ["anonymous", "registered"]:
            rate_result = await self.test_rate_limiting_policy(user_type)
            results["rate_limiting"][user_type] = rate_result
            await asyncio.sleep(10)  # Cool down between user types

        # Test 2: Backpressure
        backpressure_result = await self.test_backpressure_policy()
        results["backpressure"] = backpressure_result
        await asyncio.sleep(15)  # Cool down after backpressure test

        # Test 3: Circuit Breaker
        circuit_breaker_result = await self.test_circuit_breaker_policy()
        results["circuit_breaker"] = circuit_breaker_result

        # Get final system state
        results["system_state_after"] = await self.get_system_state()

        return results

    def print_policy_test_results(self, results: Dict[str, any]):
        """Print formatted policy test results"""
        print(f"\n{'='*60}")
        print(f"📋 POLICY TEST RESULTS")
        print(f"{'='*60}")

        # Rate Limiting Results
        print(f"\n🚦 RATE LIMITING TESTS:")
        for user_type, rate_data in results["rate_limiting"].items():
            print(f"\n  {user_type.upper()} Users:")
            print(f"    Requests sent: {rate_data['requests_sent']}")
            print(f"    Successful: {rate_data['successful_requests']}")
            print(f"    Rate limited: {rate_data['rate_limited_requests']}")

            if rate_data['first_rate_limit_at']:
                print(f"    ✅ Rate limit triggered at request #{rate_data['first_rate_limit_at']}")
                limit_effectiveness = rate_data['rate_limited_requests'] / max(1, rate_data['requests_sent'])
                print(f"    Rate limit effectiveness: {limit_effectiveness*100:.1f}%")
            else:
                print(f"    ❌ Rate limit NOT triggered")

            if rate_data['recovery_time']:
                print(f"    ✅ Recovered in {rate_data['recovery_time']:.1f}s")
            elif rate_data['rate_limited_requests'] > 0:
                print(f"    ⏳ Recovery time exceeded test duration")

        # Backpressure Results
        print(f"\n🌊 BACKPRESSURE TESTS:")
        bp_data = results["backpressure"]
        print(f"    Max concurrent users tested: {bp_data['concurrent_users_tested']}")
        print(f"    Total 503 responses: {bp_data['total_503_responses']}")

        if bp_data['backpressure_triggered_at']:
            print(f"    ✅ Backpressure triggered at {bp_data['backpressure_triggered_at']} concurrent users")
            print(f"    Anonymous users blocked: {'Yes' if bp_data['anonymous_blocked'] else 'No'}")
        else:
            print(f"    ❌ Backpressure NOT triggered")

        # Circuit Breaker Results
        print(f"\n⚡ CIRCUIT BREAKER TESTS:")
        cb_data = results["circuit_breaker"]

        if cb_data['circuit_breaker_triggered']:
            print(f"    ✅ Circuit breaker triggered after {cb_data['requests_to_trigger']} requests")
            print(f"    Trigger method: {cb_data['trigger_method']}")
            print(f"    Effectiveness: {cb_data['effectiveness']*100:.1f}%")

            if cb_data['recovery_time']:
                print(f"    ✅ Recovered in {cb_data['recovery_time']/60:.1f} minutes")
            else:
                print(f"    ⏳ Recovery time exceeded test duration")
        else:
            print(f"    ❌ Circuit breaker NOT triggered")

        # System Impact Analysis
        self._print_system_impact_analysis(results)

        # Policy Recommendations
        self._print_policy_recommendations(results)

    def _print_system_impact_analysis(self, results: Dict[str, any]):
        """Print system impact analysis"""
        print(f"\n🖥️  SYSTEM IMPACT ANALYSIS:")

        before = results.get("system_state_before", {})
        after = results.get("system_state_after", {})

        # Queue analysis
        queue_before = before.get("queue_stats", {})
        queue_after = after.get("queue_stats", {})

        total_before = sum(queue_before.values())
        total_after = sum(queue_after.values())

        print(f"    Queue size change: {total_before} → {total_after} (Δ{total_after - total_before:+d})")

        # Throttling analysis
        throttling_before = before.get("throttling_stats", {})
        throttling_after = after.get("throttling_stats", {})

        requests_before = throttling_before.get("total_requests", 0)
        requests_after = throttling_after.get("total_requests", 0)
        throttled_before = throttling_before.get("total_throttled", 0)
        throttled_after = throttling_after.get("total_throttled", 0)

        print(f"    Total requests: {requests_before} → {requests_after} (Δ{requests_after - requests_before:+d})")
        print(f"    Total throttled: {throttled_before} → {throttled_after} (Δ{throttled_after - throttled_before:+d})")

        if requests_after > requests_before:
            new_throttle_rate = (throttled_after - throttled_before) / (requests_after - requests_before) * 100
            print(f"    Test throttle rate: {new_throttle_rate:.1f}%")

        # Circuit breaker analysis
        if "circuit_breakers" in after:
            cb_summary = after["circuit_breakers"].get("summary", {})
            open_circuits = cb_summary.get("open_circuits", 0)
            total_circuits = cb_summary.get("total_circuits", 0)

            if open_circuits > 0:
                print(f"    ⚠️  {open_circuits}/{total_circuits} circuit breakers are open")
            else:
                print(f"    ✅ All {total_circuits} circuit breakers are closed")

    def _print_policy_recommendations(self, results: Dict[str, any]):
        """Print policy recommendations based on test results"""
        print(f"\n💡 POLICY RECOMMENDATIONS:")

        recommendations = []

        # Rate limiting recommendations
        rate_results = results["rate_limiting"]

        for user_type, data in rate_results.items():
            if not data.get("first_rate_limit_at"):
                recommendations.append(f"🔧 Consider lowering rate limits for {user_type} users - no rate limiting observed")
            elif data.get("first_rate_limit_at", 0) > 30:
                recommendations.append(f"⚡ Rate limits for {user_type} users may be too high - took {data['first_rate_limit_at']} requests to trigger")

            if data.get("recovery_time", 0) > 120:
                recommendations.append(f"⏰ Rate limit recovery time for {user_type} users is long ({data['recovery_time']:.1f}s)")

        # Backpressure recommendations
        bp_data = results["backpressure"]
        if not bp_data.get("backpressure_triggered_at"):
            recommendations.append("🌊 Backpressure threshold may be too high - consider lowering queue limits")
        elif bp_data.get("backpressure_triggered_at", 0) < 10:
            recommendations.append("🌊 Backpressure threshold may be too low - triggered with only few users")

        # Circuit breaker recommendations
        cb_data = results["circuit_breaker"]
        if not cb_data.get("circuit_breaker_triggered"):
            recommendations.append("⚡ Circuit breaker may need tuning - did not trigger during test")
        elif cb_data.get("effectiveness", 0) < 0.8:
            recommendations.append(f"⚡ Circuit breaker effectiveness is low ({cb_data['effectiveness']*100:.1f}%)")

        if cb_data.get("recovery_time", 0) > 300:  # 5 minutes
            recommendations.append("⚡ Circuit breaker recovery time is very long - consider shorter timeout")

        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. {rec}")
        else:
            print("    ✅ All policies appear to be working correctly!")

        # Configuration suggestions
        print(f"\n🔧 SUGGESTED CONFIGURATION ADJUSTMENTS:")

        # Rate limit suggestions
        anon_data = rate_results.get("anonymous", {})
        reg_data = rate_results.get("registered", {})

        if anon_data.get("first_rate_limit_at"):
            suggested_anon_burst = max(3, anon_data["first_rate_limit_at"] - 2)
            print(f"    Anonymous burst limit: {suggested_anon_burst} requests")

        if reg_data.get("first_rate_limit_at"):
            suggested_reg_burst = max(5, reg_data["first_rate_limit_at"] - 2)
            print(f"    Registered burst limit: {suggested_reg_burst} requests")

        # Backpressure suggestions
        if bp_data.get("backpressure_triggered_at"):
            suggested_queue_size = bp_data["backpressure_triggered_at"] * 3  # 3 requests per user average
            print(f"    Queue capacity: {suggested_queue_size} requests")

        # Circuit breaker suggestions
        if cb_data.get("requests_to_trigger"):
            suggested_failure_threshold = max(3, cb_data["requests_to_trigger"] // 10)
            print(f"    Circuit breaker failure threshold: {suggested_failure_threshold} failures")

async def main():
    """Main function for policy testing"""
    parser = argparse.ArgumentParser(description="Gateway Policy Testing Suite")
    parser.add_argument("--base-url", default="http://localhost:8010", help="Base URL")
    parser.add_argument("--admin-token", default="test-admin-token-123", help="Admin token")
    parser.add_argument("--user-token", default="test-chat-token-456", help="User token")
    parser.add_argument("--test", choices=[
        "reset", "rate-limit", "backpressure", "circuit-breaker", "all"
    ], default="all", help="Test to run")
    parser.add_argument("--user-type", choices=["anonymous", "registered"],
                        default="anonymous", help="User type for specific tests")

    args = parser.parse_args()

    async with PolicyTester(args.base_url, args.admin_token, args.user_token) as tester:

        if args.test == "reset":
            reset_results = await tester.reset_all_policies()
            print("Reset results:", reset_results)

        elif args.test == "rate-limit":
            rate_result = await tester.test_rate_limiting_policy(args.user_type)
            print(f"Rate limiting test result: {rate_result}")

        elif args.test == "backpressure":
            bp_result = await tester.test_backpressure_policy()
            print(f"Backpressure test result: {bp_result}")

        elif args.test == "circuit-breaker":
            cb_result = await tester.test_circuit_breaker_policy()
            print(f"Circuit breaker test result: {cb_result}")

        elif args.test == "all":
            comprehensive_results = await tester.run_comprehensive_policy_test()
            tester.print_policy_test_results(comprehensive_results)

            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"policy_test_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            print(f"\n📄 Results saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())