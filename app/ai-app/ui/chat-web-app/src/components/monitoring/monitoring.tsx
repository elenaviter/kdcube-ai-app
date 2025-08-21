/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React, {useState, useEffect, useCallback} from 'react';
import {
    Activity,
    Server,
    MessageSquare,
    Database,
    Zap,
    AlertTriangle,
    CheckCircle,
    XCircle,
    RefreshCw,
    Users,
    List,
    TrendingUp,
    Clock,
    BarChart3,
    UserCheck,
    UserX,
    X,
    Scale,
    Globe,
    Shield,
    Layers,
    Power,
    PowerOff,
    RotateCcw,
    AlertCircle,
    Pause,
    Play,
    Eye,
    EyeOff,
    Filter,
    Timer,
    Gauge
} from 'lucide-react';
import {useAuthManagerContext} from "../auth/AuthManager.tsx";
import {GatewayConfigurationComponent} from "./GatewayConfigurationComponent.tsx";
import {CapacityTransparencyPanel} from "./CapacityTransparencyPanel.tsx";
import {getMonitoringBaseAddress} from "../../AppConfig.ts";

// Enhanced interfaces for configuration data
interface GatewayConfiguration {
    current_profile: string;
    instance_id: string;
    tenant_id: string;
    display_name: string;
    rate_limits: {
        anonymous: {
            hourly: number;
            burst: number;
            burst_window: number;
        };
        registered: {
            hourly: number;
            burst: number;
            burst_window: number;
        };
        privileged: {
            hourly: number;
            burst: number;
            burst_window: number;
        };
    };
    service_capacity: {
        concurrent_requests_per_instance: number;
        avg_processing_time_seconds: number;
        requests_per_hour: number;
    };
    backpressure_settings: {
        capacity_buffer: number;
        queue_depth_multiplier: number;
        anonymous_pressure_threshold: number;
        registered_pressure_threshold: number;
        hard_limit_threshold: number;
    };
    circuit_breaker_settings: {
        authentication: {
            failure_threshold: number;
            recovery_timeout: number;
            success_threshold: number;
            window_size: number;
        };
        rate_limiter: {
            failure_threshold: number;
            recovery_timeout: number;
            success_threshold: number;
            window_size: number;
        };
        backpressure: {
            failure_threshold: number;
            recovery_timeout: number;
            success_threshold: number;
            window_size: number;
        };
    };
    monitoring_settings: {
        throttling_events_retention_hours: number;
        session_analytics_enabled: boolean;
        queue_analytics_enabled: boolean;
        heartbeat_timeout_seconds: number;
        instance_cache_ttl_seconds: number;
    };
}

interface ComputedMetrics {
    base_queue_size_per_instance: number;
    theoretical_throughput_per_instance: number;
    effective_concurrent_capacity: number;
    queue_capacity_per_instance: number;
}

// Enhanced interfaces for queue data (keeping existing ones)
interface QueueAnalytics {
    anonymous: {
        size: number;
        avg_wait: number;
        throughput: number;
        blocked: boolean;
    };
    registered: {
        size: number;
        avg_wait: number;
        throughput: number;
        blocked: boolean;
    };
    privileged: {
        size: number;
        avg_wait: number;
        throughput: number;
        blocked: boolean;
    };
}

interface BackpressurePolicy {
    thresholds: {
        anonymous_threshold: number;
        registered_threshold: number;
        hard_limit_threshold: number;
    };
    current_effects: {
        anonymous_blocked: boolean;
        registered_blocked: boolean;
        all_blocked: boolean;
        pressure_level: 'low' | 'medium' | 'high' | 'critical';
    };
    capacity_scaling: {
        base_per_instance: number;
        instances_detected: number;
        total_weighted_capacity: number;
        utilization_percent: number;
    };
}

interface EnhancedQueueStats {
    anonymous_queue: number;
    registered_queue: number;
    privileged_queue: number;
    total_queue: number;
    base_capacity_per_instance: number;
    alive_instances: string[];
    instance_count: number;
    weighted_max_capacity: number;
    pressure_ratio: number;
    accepting_anonymous: boolean;
    accepting_registered: boolean;
    accepting_privileged: boolean;
}

// Circuit breaker interfaces (from previous code)
interface CircuitBreakerStats {
    name: string;
    state: 'closed' | 'open' | 'half_open';
    failure_count: number;
    success_count: number;
    total_requests: number;
    total_failures: number;
    consecutive_failures: number;
    current_window_failures: number;
    last_failure_time?: number;
    last_success_time?: number;
    opened_at?: number;
}

interface CircuitBreakerSummary {
    total_circuits: number;
    open_circuits: number;
    half_open_circuits: number;
    closed_circuits: number;
}


const SystemMonitor = () => {
    // Your existing state
    const [instances, setInstances] = useState({});
    const [globalStats, setGlobalStats] = useState({load: 0, capacity: 0, healthy: 0, total: 0});
    const [queueStats, setQueueStats] = useState({anonymous: 0, registered: 0, privileged: 0});

    // Enhanced queue state
    const [enhancedQueueStats, setEnhancedQueueStats] = useState<EnhancedQueueStats | null>(null);
    const [backpressurePolicy, setBackpressurePolicy] = useState<BackpressurePolicy | null>(null);
    const [queueAnalytics, setQueueAnalytics] = useState<{individual_queues: QueueAnalytics} | null>(null);

    // Enhanced configuration state
    const [gatewayConfiguration, setGatewayConfiguration] = useState<GatewayConfiguration | null>(null);

    // Other state
    const [capacityInfo, setCapacityInfo] = useState(null);
    const [sessionAnalytics, setSessionAnalytics] = useState(null);
    const [throttlingStats, setThrottlingStats] = useState(null);
    const [throttlingTimePeriod, setThrottlingTimePeriod] = useState(1); // hours

    const [recentThrottlingEvents, setRecentThrottlingEvents] = useState([]);
    const [throttlingByPeriod, setThrottlingByPeriod] = useState({});

    // New circuit breaker state
    const [circuitBreakers, setCircuitBreakers] = useState<Record<string, CircuitBreakerStats>>({});
    const [circuitBreakerSummary, setCircuitBreakerSummary] = useState<CircuitBreakerSummary>({
        total_circuits: 0,
        open_circuits: 0,
        half_open_circuits: 0,
        closed_circuits: 0
    });
    const [capacityTransparency, setCapacityTransparency] = useState(null);
    const [showCapacityTransparency, setShowCapacityTransparency] = useState(true);

    const [isLoading, setIsLoading] = useState(false);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);

    // New UI state for enhanced features
    const [showCircuitBreakers, setShowCircuitBreakers] = useState(true);
    const [showQueueDetails, setShowQueueDetails] = useState(true);
    const [showBackpressurePolicy, setShowBackpressurePolicy] = useState(true);
    const [showConfiguration, setShowConfiguration] = useState(false); // New toggle for configuration
    const [queueFilter, setQueueFilter] = useState<'all' | 'blocked' | 'active'>('all');

    // Configuration
    const monitoring_backend = getMonitoringBaseAddress();
    const MONITOR_ENDPOINT = `${monitoring_backend}/monitoring/system`;
    const ADMIN_CIRCUIT_BREAKER_ENDPOINT = `${monitoring_backend}/admin/circuit-breakers`;
    const REFRESH_INTERVAL = 5000; // 5 seconds

    const authContext = useAuthManagerContext();

    // Fetch data from monitoring endpoints
    const fetchSystemData = useCallback(async () => {
        setIsLoading(true);
        setError(null);

        try {
            const headers: HeadersInit = [
                ['Content-Type', 'application/json']
            ];
            authContext.appendAuthHeader(headers);

            // Fetch system monitoring data (includes all throttling data)
            const systemResponse = await fetch(MONITOR_ENDPOINT, {
                method: 'GET',
                headers,
            });

            if (!systemResponse.ok) {
                throw new Error(`HTTP ${systemResponse.status}: ${systemResponse.statusText}`);
            }

            const systemData = await systemResponse.json();

            // Update existing state...
            setInstances(systemData.instances || {});
            setGlobalStats(systemData.global_stats || {load: 0, capacity: 0, healthy: 0, total: 0});
            setQueueStats(systemData.queue_stats || {anonymous: 0, registered: 0, privileged: 0});
            setCapacityInfo(systemData.capacity_info || null);
            setSessionAnalytics(systemData.session_analytics || null);
            setEnhancedQueueStats(systemData.enhanced_queue_stats || null);
            setBackpressurePolicy(systemData.backpressure_policy || null);
            setQueueAnalytics(systemData.queue_analytics || null);
            setGatewayConfiguration(systemData.gateway_configuration || null);
            setCapacityTransparency(systemData.capacity_transparency || null);

            if (systemData.circuit_breakers) {
                setCircuitBreakers(systemData.circuit_breakers.circuits || {});
                setCircuitBreakerSummary(systemData.circuit_breakers.summary || {
                    total_circuits: 0,
                    open_circuits: 0,
                    half_open_circuits: 0,
                    closed_circuits: 0
                });
            }

            // Update throttling data from existing endpoint
            setThrottlingStats(systemData.throttling_stats || null);
            setRecentThrottlingEvents(systemData.recent_throttling_events || []);
            setThrottlingByPeriod(systemData.throttling_by_period || {});

            setLastUpdate(new Date());

        } catch (err) {
            setError(err.message);
            console.error('Failed to fetch system data:', err);
        } finally {
            setIsLoading(false);
        }
    }, [authContext]);

    // Reset circuit breaker function
    const resetCircuitBreaker = useCallback(async (circuitName: string) => {
        try {
            const headers: HeadersInit = [
                ['Content-Type', 'application/json']
            ];
            authContext.appendAuthHeader(headers);

            const response = await fetch(`${ADMIN_CIRCUIT_BREAKER_ENDPOINT}/${circuitName}/reset`, {
                method: 'POST',
                headers,
            });

            if (!response.ok) {
                throw new Error(`Failed to reset circuit breaker: ${response.statusText}`);
            }

            // Refresh data after reset
            await fetchSystemData();

        } catch (err) {
            console.error(`Error resetting circuit breaker ${circuitName}:`, err);
            setError(`Failed to reset circuit breaker: ${err.message}`);
        }
    }, [authContext, fetchSystemData]);

    const renderThrottlingSection = () => {
        // Get stats for selected time period
        const selectedPeriodKey = `${throttlingTimePeriod}h`;
        const stats = throttlingByPeriod[selectedPeriodKey] || throttlingStats;
        const events = recentThrottlingEvents; // Always show recent events from last hour

        if (!stats) return null;

        const throttleRate = stats.throttle_rate || 0;

        return (
            <div className="bg-white rounded border p-3 mb-3">
                <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4"/>
                    <span className="font-semibold text-xs">Throttling & Rate Limits</span>

                    {/* Time Period Selector */}
                    <select
                        value={throttlingTimePeriod}
                        onChange={(e) => setThrottlingTimePeriod(parseInt(e.target.value))}
                        className="text-xs border rounded px-2 py-1 ml-2"
                    >
                        <option value={1}>Last 1 Hour</option>
                        <option value={3}>Last 3 Hours</option>
                        <option value={6}>Last 6 Hours</option>
                        <option value={12}>Last 12 Hours</option>
                        <option value={24}>Last 24 Hours</option>
                    </select>

                    <span className={`text-xs px-1 rounded ${
                        throttleRate > 5 ? 'bg-red-100 text-red-700' :
                            throttleRate > 2 ? 'bg-orange-100 text-orange-700' :
                                'bg-green-100 text-green-700'
                    }`}>
                        {throttleRate.toFixed(1)}% rate
                    </span>

                    {events.length > 0 && (
                        <span className="text-xs bg-yellow-100 text-yellow-700 px-1 rounded">
                            {events.length} recent events
                        </span>
                    )}
                </div>

                {/* Time Period Info */}
                <div className="text-xs text-gray-600 mb-2">
                    Showing data for: Last {throttlingTimePeriod} hour{throttlingTimePeriod !== 1 ? 's' : ''}
                    {stats && (
                        <span className="ml-2">
                            ({stats.total_throttled} events,
                            {stats.events_per_hour?.toFixed(1) || 0} per hour)
                        </span>
                    )}
                </div>

                {/* Throttling Overview */}
                <div className="grid grid-cols-4 gap-3 mb-3">
                    <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                        <TrendingUp className="w-3 h-3 text-blue-600"/>
                        <span className="text-xs">Total Requests</span>
                        <span className="font-bold text-blue-700">{stats.total_requests || 0}</span>
                    </div>
                    <div className="flex items-center gap-2 p-2 bg-yellow-50 rounded">
                        <AlertTriangle className="w-3 h-3 text-yellow-600"/>
                        <span className="text-xs">Rate Limited (429)</span>
                        <span className="font-bold text-yellow-700">{stats.rate_limit_429 || 0}</span>
                    </div>
                    <div className="flex items-center gap-2 p-2 bg-red-50 rounded">
                        <XCircle className="w-3 h-3 text-red-600"/>
                        <span className="text-xs">Service Busy (503)</span>
                        <span className="font-bold text-red-700">{stats.backpressure_503 || 0}</span>
                    </div>
                    <div className="flex items-center gap-2 p-2 bg-purple-50 rounded">
                        <Shield className="w-3 h-3 text-purple-600"/>
                        <span className="text-xs">Total Throttled</span>
                        <span className="font-bold text-purple-700">{stats.total_throttled || 0}</span>
                    </div>
                </div>

                {/* Recent Throttling Events - always show last hour events */}
                {events.length > 0 && (
                    <div className="border-t pt-2">
                        <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-3 h-3"/>
                            <span className="font-semibold text-xs">Recent Events (Last Hour)</span>
                        </div>
                        <div className="space-y-1 max-h-24 overflow-y-auto">
                            {events.slice(0, 5).map((event, index) => {
                                const eventTime = new Date(event.timestamp * 1000);
                                const timeAgo = Math.round((Date.now() - eventTime.getTime()) / 1000 / 60); // minutes ago

                                return (
                                    <div key={event.event_id || index}
                                         className="flex items-center justify-between text-xs">
                                        <div className="flex items-center gap-2">
                                            <span
                                                className={`px-1 py-0.5 rounded text-xs ${getThrottlingReasonColor(event.reason)}`}>
                                                {event.reason?.replace(/_/g, ' ') || 'Unknown'}
                                            </span>
                                            <span className="text-gray-600">{event.user_type}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span
                                                className={`font-medium ${event.http_status === 429 ? 'text-yellow-600' : 'text-red-600'}`}>
                                                {event.http_status}
                                            </span>
                                            <span className="text-gray-500">
                                                {timeAgo}m ago
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {/* Throttling Breakdown by Reason */}
                {stats.throttled_by_reason && Object.keys(stats.throttled_by_reason).length > 0 && (
                    <div className="border-t pt-2 mt-2">
                        <div className="text-xs text-gray-600 mb-1">
                            Breakdown by Reason (Last {throttlingTimePeriod}h):
                        </div>
                        <div className="flex flex-wrap gap-1">
                            {Object.entries(stats.throttled_by_reason)
                                .sort(([,a], [,b]) => b - a) // Sort by count descending
                                .map(([reason, count]) => (
                                    <span key={reason}
                                          className={`px-1 py-0.5 rounded text-xs ${getThrottlingReasonColor(reason)}`}>
                                    {reason.replace(/_/g, ' ')}: {count}
                                </span>
                                ))}
                        </div>
                    </div>
                )}

                {/* Period Comparison */}
                {Object.keys(throttlingByPeriod).length > 1 && (
                    <div className="border-t pt-2 mt-2">
                        <div className="text-xs text-gray-600 mb-1">Period Comparison:</div>
                        <div className="grid grid-cols-5 gap-1 text-xs">
                            {[1, 3, 6, 12, 24].map(hours => {
                                const periodKey = `${hours}h`;
                                const periodData = throttlingByPeriod[periodKey];
                                if (!periodData) return null;

                                return (
                                    <div key={hours}
                                         className={`text-center p-1 rounded ${
                                             hours === throttlingTimePeriod ? 'bg-blue-100 border border-blue-300' : 'bg-gray-50'
                                         }`}>
                                        <div className="text-gray-600">{hours}h</div>
                                        <div className="font-medium">{periodData.total_throttled}</div>
                                        <div className="text-xs text-gray-500">{periodData.throttle_rate?.toFixed(1)}%</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        );
    };
    // Manual refresh
    const handleRefresh = () => {
        fetchSystemData();
    };

    // Auto refresh setup
    useEffect(() => {
        fetchSystemData(); // Initial load

        let interval;
        if (autoRefresh) {
            interval = setInterval(fetchSystemData, REFRESH_INTERVAL);
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [autoRefresh, fetchSystemData]);

    // Helper functions
    const HealthIcon = ({health}) => {
        switch (health) {
            case 'healthy':
                return <CheckCircle className="w-3 h-3 text-green-500"/>;
            case 'degraded':
                return <AlertTriangle className="w-3 h-3 text-yellow-500"/>;
            case 'unhealthy':
                return <XCircle className="w-3 h-3 text-red-500"/>;
            default:
                return <Activity className="w-3 h-3 text-gray-400"/>;
        }
    };

    const ServiceIcon = ({type}) => {
        const icons = {
            chat_rest: MessageSquare,
            chat_socket: MessageSquare,
            kb_rest: Database,
            orchestrator: Zap
        };
        const Icon = icons[type] || Activity;
        return <Icon className="w-3 h-3"/>;
    };

    const LoadBar = ({load, capacity}) => {
        const pct = capacity > 0 ? (load / capacity) * 100 : 0;
        const color = pct >= 90 ? 'bg-red-400' : pct >= 70 ? 'bg-yellow-400' : 'bg-green-400';
        return (
            <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                <div className={`h-1.5 rounded-full ${color}`} style={{width: `${Math.min(pct, 100)}%`}}/>
            </div>
        );
    };

    const CircuitBreakerIcon = ({state}) => {
        switch (state) {
            case 'closed':
                return <Power className="w-3 h-3 text-green-500"/>;
            case 'open':
                return <PowerOff className="w-3 h-3 text-red-500"/>;
            case 'half_open':
                return <Pause className="w-3 h-3 text-yellow-500"/>;
            default:
                return <AlertCircle className="w-3 h-3 text-gray-400"/>;
        }
    };

    const getCircuitBreakerColor = (state: string): string => {
        switch (state) {
            case 'closed':
                return 'text-green-600 bg-green-50 border-green-200';
            case 'open':
                return 'text-red-600 bg-red-50 border-red-200';
            case 'half_open':
                return 'text-yellow-600 bg-yellow-50 border-yellow-200';
            default:
                return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    };

    const getPressureLevelColor = (level: string): string => {
        switch (level) {
            case 'low':
                return 'text-green-600 bg-green-50';
            case 'medium':
                return 'text-yellow-600 bg-yellow-50';
            case 'high':
                return 'text-orange-600 bg-orange-50';
            case 'critical':
                return 'text-red-600 bg-red-50';
            default:
                return 'text-gray-600 bg-gray-50';
        }
    };

    const formatTime = (seconds: number): string => {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
        return `${(seconds / 3600).toFixed(1)}h`;
    };

    const formatTimestamp = (timestamp) => {
        return new Date(timestamp * 1000).toLocaleTimeString();
    };

    const formatHeartbeat = (timestamp) => {
        const now = Date.now() / 1000;
        const diff = now - timestamp;

        if (diff < 60) return `${Math.floor(diff)}s`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m`;
        return `${Math.floor(diff / 3600)}h`;
    };

    const getHeartbeatColor = (timestamp) => {
        const diff = (Date.now() / 1000) - timestamp;
        if (diff > 30) return 'text-red-500';
        if (diff > 15) return 'text-yellow-500';
        return 'text-green-500';
    };

    const getProfileColor = (profile: string): string => {
        switch (profile.toLowerCase()) {
            case 'development':
                return 'text-green-600 bg-green-50';
            case 'testing':
                return 'text-blue-600 bg-blue-50';
            case 'production':
                return 'text-red-600 bg-red-50';
            case 'load_test':
                return 'text-purple-600 bg-purple-50';
            default:
                return 'text-gray-600 bg-gray-50';
        }
    };

    const getThrottlingReasonColor = (reason) => {
        const colorMap = {
            'session_rate_limit': 'bg-yellow-100 text-yellow-800',
            'hourly_rate_limit': 'bg-orange-100 text-orange-800',
            'burst_rate_limit': 'bg-red-100 text-red-800',
            'system_backpressure': 'bg-purple-100 text-purple-800',
            'anonymous_rejected': 'bg-blue-100 text-blue-800'
        };
        return colorMap[reason] || 'bg-gray-100 text-gray-800';
    };

    // Calculate system metrics
    const totalQueueSize = queueStats.anonymous + queueStats.registered + (queueStats.privileged || 0);
    const queueUtilization = enhancedQueueStats
        ? Math.round((enhancedQueueStats.total_queue / enhancedQueueStats.weighted_max_capacity) * 100)
        : Math.round((totalQueueSize / 50) * 100);

    const throttleRate = throttlingStats && throttlingStats.total_requests > 0
        ? throttlingStats.throttle_rate || 0
        : 0;

    // Calculate system health
    const getSystemHealthStatus = () => {
        const hasOpenCircuits = circuitBreakerSummary.open_circuits > 0;
        const hasHalfOpenCircuits = circuitBreakerSummary.half_open_circuits > 0;
        const isCriticalPressure = backpressurePolicy?.current_effects.pressure_level === 'critical';
        const isHighPressure = backpressurePolicy?.current_effects.pressure_level === 'high';

        if (hasOpenCircuits || isCriticalPressure) return 'unhealthy';
        if (hasHalfOpenCircuits || isHighPressure) return 'degraded';
        return 'healthy';
    };

    const systemHealth = getSystemHealthStatus();

    // Filter queue analytics
    const getFilteredQueues = () => {
        if (!queueAnalytics?.individual_queues) return {};

        const queues = queueAnalytics.individual_queues;

        switch (queueFilter) {
            case 'blocked':
                return Object.fromEntries(
                    Object.entries(queues).filter(([_, queue]) => queue.blocked)
                );
            case 'active':
                return Object.fromEntries(
                    Object.entries(queues).filter(([_, queue]) => !queue.blocked && queue.size > 0)
                );
            default:
                return queues;
        }
    };

    return (
        <div className="p-4 bg-gray-50 font-mono text-xs">
            {/* Header */}
            <div className="flex items-center justify-between mb-3 pb-2 border-b">
                <h1 className="text-sm font-bold flex items-center gap-1">
                    <Server className="w-4 h-4"/>
                    System Monitor
                    {gatewayConfiguration && (
                        <span className={`ml-2 px-2 py-1 rounded text-xs ${getProfileColor(gatewayConfiguration.current_profile)}`}>
                            {gatewayConfiguration.current_profile.toUpperCase()}
                        </span>
                    )}
                    {capacityTransparency && (
                        <span className="ml-2 px-2 py-1 bg-green-100 text-green-700 rounded text-xs">
                            DYNAMIC-CAP
                        </span>
                    )}
                    {capacityTransparency?.instance_scaling && (
                        <span className={`ml-2 px-2 py-1 rounded text-xs ${
                            capacityTransparency.instance_scaling.process_health_ratio > 0.8 ? 'bg-green-100 text-green-700' :
                                capacityTransparency.instance_scaling.process_health_ratio > 0.6 ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-red-100 text-red-700'
                        }`}>
                            {capacityTransparency.instance_scaling.total_healthy_processes}/{capacityTransparency.instance_scaling.total_actual_processes} PROC
                        </span>
                    )}
                    {/* Capacity Warnings */}
                    {capacityTransparency?.capacity_warnings && capacityTransparency.capacity_warnings.length > 0 && (
                        <span className="ml-2 px-2 py-1 bg-red-100 text-red-700 rounded text-xs">
                            {capacityTransparency.capacity_warnings.length} WARN
                        </span>
                    )}
                    {circuitBreakerSummary.open_circuits > 0 && (
                        <span className="ml-2 px-2 py-1 bg-red-100 text-red-700 rounded text-xs">
                            {circuitBreakerSummary.open_circuits} CB OPEN
                        </span>
                    )}
                    {backpressurePolicy?.current_effects.all_blocked && (
                        <span className="ml-2 px-2 py-1 bg-red-100 text-red-700 rounded text-xs">
                            ALL BLOCKED
                        </span>
                    )}
                </h1>
                <div className="flex items-center gap-3 text-xs">
                    <span>Load: {globalStats.load}/{globalStats.capacity}</span>
                    <span>Health: {globalStats.healthy}/{globalStats.total}</span>
                    {globalStats.actual !== undefined && globalStats.actual !== globalStats.total && (
                        <span className="text-orange-600">({globalStats.actual} running)</span>
                    )}
                    <span>Queue: {totalQueueSize}</span>
                    {capacityTransparency?.instance_scaling && (
                        <span className="text-blue-600">
                            Cap: {capacityTransparency.instance_scaling.total_system_capacity}
                            ({capacityTransparency.instance_scaling.total_healthy_processes}h/{capacityTransparency.instance_scaling.total_configured_processes}c)
                        </span>
                    )}
                    <span className={`px-2 py-1 rounded ${
                        systemHealth === 'healthy' ? 'bg-green-100 text-green-700' :
                            systemHealth === 'degraded' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-red-100 text-red-700'
                    }`}>
                        {systemHealth.toUpperCase()}
                    </span>
                    <span className={throttleRate > 5 ? 'text-red-600' : throttleRate > 2 ? 'text-orange-600' : 'text-green-600'}>
                        Throttle: {throttleRate.toFixed(1)}%
                    </span>
                    {enhancedQueueStats && (
                        <span className="text-blue-600">({enhancedQueueStats.instance_count} inst)</span>
                    )}

                    {/* View toggles */}
                    <div className="flex items-center gap-1">
                        {Object.keys(circuitBreakers).length > 0 && (
                            <button
                                onClick={() => setShowCircuitBreakers(!showCircuitBreakers)}
                                className={`px-2 py-1 rounded text-xs ${
                                    showCircuitBreakers ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                                }`}
                                title="Toggle circuit breaker display"
                            >
                                <Power className="w-3 h-3 inline mr-1"/>
                                CB
                            </button>
                        )}

                        <button
                            onClick={() => setShowQueueDetails(!showQueueDetails)}
                            className={`px-2 py-1 rounded text-xs ${
                                showQueueDetails ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                            }`}
                            title="Toggle queue details"
                        >
                            <List className="w-3 h-3 inline mr-1"/>
                            Q
                        </button>

                        <button
                            onClick={() => setShowBackpressurePolicy(!showBackpressurePolicy)}
                            className={`px-2 py-1 rounded text-xs ${
                                showBackpressurePolicy ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-600'
                            }`}
                            title="Toggle backpressure policy display"
                        >
                            <Shield className="w-3 h-3 inline mr-1"/>
                            BP
                        </button>
                        <button
                            onClick={() => setShowCapacityTransparency(!showCapacityTransparency)}
                            className={`px-2 py-1 rounded text-xs ${
                                showCapacityTransparency ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                            }`}
                            title="Toggle capacity transparency display"
                        >
                            <Gauge className="w-3 h-3 inline mr-1"/>
                            CAP
                        </button>
                        <button
                            onClick={() => setShowConfiguration(!showConfiguration)}
                            className={`px-2 py-1 rounded text-xs ${
                                showConfiguration ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600'
                            }`}
                            title="Toggle configuration display"
                        >
                            <Layers className="w-3 h-3 inline mr-1"/>
                            CFG
                        </button>
                    </div>

                    {/* Auto refresh toggle */}
                    <label className="flex items-center gap-1 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            className="w-3 h-3"
                        />
                        <span className="text-xs">Auto</span>
                    </label>

                    {/* Manual refresh button */}
                    <button
                        onClick={handleRefresh}
                        disabled={isLoading}
                        className={`p-1 rounded hover:bg-gray-200 transition-colors ${isLoading ? 'animate-spin' : ''}`}
                        title="Refresh now"
                    >
                        <RefreshCw className="w-3 h-3"/>
                    </button>

                    {/* Status indicator */}
                    <div
                        className={`w-2 h-2 rounded-full ${error ? 'bg-red-400' : isLoading ? 'bg-yellow-400' : 'bg-green-400'} ${!error && !isLoading ? 'animate-pulse' : ''}`}/>

                    {/* Last update time */}
                    {lastUpdate && (
                        <span className="text-gray-500">
                            {lastUpdate.toLocaleTimeString()}
                        </span>
                    )}
                </div>
            </div>

            {/* Error display */}
            {error && (
                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                    Error: {error}
                </div>
            )}

            {/* Gateway Configuration Section */}
            {showConfiguration && (
                <GatewayConfigurationComponent
                    gatewayConfiguration={gatewayConfiguration}
                    capacityTransparency={capacityTransparency}
                />
            )}

            {/* Enhanced Queue Analytics Section */}
            {showQueueDetails && enhancedQueueStats && (
                <div className="bg-white rounded border p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <List className="w-4 h-4"/>
                        <span className="font-semibold text-xs">Enhanced Queue Analytics</span>
                        <span className="text-gray-500">({queueUtilization}% capacity)</span>
                        {!enhancedQueueStats.accepting_anonymous && (
                            <span className="text-red-500 text-xs bg-red-50 px-1 rounded">ANON BLOCKED</span>
                        )}
                        {!enhancedQueueStats.accepting_registered && (
                            <span className="text-orange-500 text-xs bg-orange-50 px-1 rounded">REG BLOCKED</span>
                        )}

                        {/* Queue filter */}
                        <div className="ml-auto flex items-center gap-2">
                            <Filter className="w-3 h-3"/>
                            <select
                                value={queueFilter}
                                onChange={(e) => setQueueFilter(e.target.value as any)}
                                className="text-xs border rounded px-1 py-0.5"
                            >
                                <option value="all">All Queues</option>
                                <option value="blocked">Blocked Only</option>
                                <option value="active">Active Only</option>
                            </select>
                        </div>
                    </div>

                    {/* Overall Queue Stats */}
                    <div className="grid grid-cols-4 gap-3 mb-3">
                        <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                            <UserCheck className="w-3 h-3 text-blue-600"/>
                            <span className="text-xs">Registered</span>
                            <span className="font-bold text-blue-700">{enhancedQueueStats.registered_queue}</span>
                            {!enhancedQueueStats.accepting_registered && <XCircle className="w-3 h-3 text-red-500"/>}
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                            <UserX className="w-3 h-3 text-gray-600"/>
                            <span className="text-xs">Anonymous</span>
                            <span className="font-bold text-gray-700">{enhancedQueueStats.anonymous_queue}</span>
                            {!enhancedQueueStats.accepting_anonymous && <XCircle className="w-3 h-3 text-red-500"/>}
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-purple-50 rounded">
                            <Shield className="w-3 h-3 text-purple-600"/>
                            <span className="text-xs">Privileged</span>
                            <span className="font-bold text-purple-700">{enhancedQueueStats.privileged_queue}</span>
                            {!enhancedQueueStats.accepting_privileged && <XCircle className="w-3 h-3 text-red-500"/>}
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-green-50 rounded">
                            <TrendingUp className="w-3 h-3 text-green-600"/>
                            <span className="text-xs">Total</span>
                            <span className="font-bold text-green-700">{enhancedQueueStats.total_queue}</span>
                        </div>
                    </div>

                    {/* Individual Queue Details */}
                    {queueAnalytics && (
                        <div className="border-t pt-2">
                            <div className="flex items-center gap-2 mb-2">
                                <BarChart3 className="w-3 h-3"/>
                                <span className="font-semibold text-xs">Queue Performance</span>
                            </div>
                            <div className="space-y-2">
                                {Object.entries(getFilteredQueues()).map(([queueType, queue]) => (
                                    <div key={queueType} className={`p-2 rounded border ${
                                        queue.blocked ? 'border-red-200 bg-red-50' : 'border-gray-200 bg-gray-50'
                                    }`}>
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="flex items-center gap-2">
                                                <span className={`text-xs font-medium ${
                                                    queueType === 'anonymous' ? 'text-gray-700' :
                                                    queueType === 'registered' ? 'text-blue-700' :
                                                    'text-purple-700'
                                                }`}>
                                                    {queueType.charAt(0).toUpperCase() + queueType.slice(1)}
                                                </span>
                                                {queue.blocked && (
                                                    <span className="px-1 py-0.5 bg-red-100 text-red-700 text-xs rounded">
                                                        BLOCKED
                                                    </span>
                                                )}
                                                <span className="text-xs text-gray-600">
                                                    {queue.size} items
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-3 text-xs">
                                                <div className="flex items-center gap-1">
                                                    <Timer className="w-3 h-3"/>
                                                    <span>Wait: {formatTime(queue.avg_wait)}</span>
                                                </div>
                                                <div className="flex items-center gap-1">
                                                    <TrendingUp className="w-3 h-3"/>
                                                    <span>T/hr: {queue.throughput}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Queue size visual */}
                                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                                            <div className={`h-1.5 rounded-full transition-all duration-300 ${
                                                queue.blocked ? 'bg-red-500' :
                                                queue.size === 0 ? 'bg-green-500' :
                                                queue.size < 10 ? 'bg-yellow-500' : 'bg-orange-500'
                                            }`} style={{
                                                width: `${Math.min((queue.size / Math.max(enhancedQueueStats.total_queue, 1)) * 100, 100)}%`
                                            }}/>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Capacity Information */}
                    <div className="border-t pt-2 mt-2">
                        <div className="flex items-center gap-2 mb-2">
                            <Scale className="w-3 h-3"/>
                            <span className="font-semibold text-xs">Dynamic Capacity</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Base per instance:</span>
                                    <span className="font-medium">{enhancedQueueStats.base_capacity_per_instance}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Alive instances:</span>
                                    <span className="font-medium text-green-600">{enhancedQueueStats.instance_count}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Weighted capacity:</span>
                                    <span className="font-medium text-blue-600">{enhancedQueueStats.weighted_max_capacity}</span>
                                </div>
                            </div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Pressure ratio:</span>
                                    <span className={`font-medium ${
                                        enhancedQueueStats.pressure_ratio > 0.8 ? 'text-red-600' : 
                                        enhancedQueueStats.pressure_ratio > 0.6 ? 'text-yellow-600' : 'text-green-600'
                                    }`}>
                                        {Math.round(enhancedQueueStats.pressure_ratio * 100)}%
                                    </span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Anonymous access:</span>
                                    <span className={`font-medium ${enhancedQueueStats.accepting_anonymous ? 'text-green-600' : 'text-red-600'}`}>
                                        {enhancedQueueStats.accepting_anonymous ? 'OPEN' : 'BLOCKED'}
                                    </span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-600">Active instances:</span>
                                    <span className="font-medium text-gray-700 text-xs truncate">
                                        {enhancedQueueStats.alive_instances && enhancedQueueStats.alive_instances.length > 0
                                            ? enhancedQueueStats.alive_instances.slice(0, 2).join(', ') +
                                              (enhancedQueueStats.alive_instances.length > 2 ? '...' : '')
                                            : 'None'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Enhanced capacity visual bar */}
                        <div className="mt-2">
                            <div className="flex justify-between text-xs mb-1">
                                <span>Queue Utilization</span>
                                <span>{enhancedQueueStats.total_queue}/{enhancedQueueStats.weighted_max_capacity}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                                <div className={`h-2 rounded-full transition-all duration-300 ${
                                    enhancedQueueStats.pressure_ratio > 0.9 ? 'bg-red-500' :
                                    enhancedQueueStats.pressure_ratio > 0.8 ? 'bg-orange-500' :
                                    enhancedQueueStats.pressure_ratio > 0.6 ? 'bg-yellow-500' :
                                    'bg-green-500'
                                }`} style={{width: `${Math.min(enhancedQueueStats.pressure_ratio * 100, 100)}%`}}/>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Backpressure Policy Transparency */}
            {showBackpressurePolicy && backpressurePolicy && (
                <div className="bg-white rounded border p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <Shield className="w-4 h-4"/>
                        <span className="font-semibold text-xs">Backpressure Policy</span>
                        <span className={`px-2 py-1 rounded text-xs ${getPressureLevelColor(backpressurePolicy.current_effects.pressure_level)}`}>
                            {backpressurePolicy.current_effects.pressure_level.toUpperCase()} PRESSURE
                        </span>
                    </div>

                    {/* Policy Thresholds */}
                    <div className="grid grid-cols-3 gap-3 mb-3">
                        <div className="p-2 bg-blue-50 rounded">
                            <div className="text-xs text-gray-600">Anonymous Threshold</div>
                            <div className="font-bold text-blue-700">{backpressurePolicy.thresholds.anonymous_threshold}</div>
                            <div className="text-xs text-gray-500">
                                {backpressurePolicy.current_effects.anonymous_blocked ? 'BLOCKED' : 'OPEN'}
                            </div>
                        </div>
                        <div className="p-2 bg-orange-50 rounded">
                            <div className="text-xs text-gray-600">Registered Threshold</div>
                            <div className="font-bold text-orange-700">{backpressurePolicy.thresholds.registered_threshold}</div>
                            <div className="text-xs text-gray-500">
                                {backpressurePolicy.current_effects.registered_blocked ? 'BLOCKED' : 'OPEN'}
                            </div>
                        </div>
                        <div className="p-2 bg-red-50 rounded">
                            <div className="text-xs text-gray-600">Hard Limit</div>
                            <div className="font-bold text-red-700">{backpressurePolicy.thresholds.hard_limit_threshold}</div>
                            <div className="text-xs text-gray-500">
                                {backpressurePolicy.current_effects.all_blocked ? 'ALL BLOCKED' : 'ACCEPTING'}
                            </div>
                        </div>
                    </div>

                    {/* Capacity Scaling Info */}
                    <div className="border-t pt-2">
                        <div className="text-xs text-gray-600 mb-1">Capacity Scaling:</div>
                        <div className="grid grid-cols-4 gap-2 text-xs">
                            <div className="text-center">
                                <div className="text-gray-600">Base/Instance</div>
                                <div className="font-medium">{backpressurePolicy.capacity_scaling.base_per_instance}</div>
                            </div>
                            <div className="text-center">
                                <div className="text-gray-600">Instances</div>
                                <div className="font-medium">{backpressurePolicy.capacity_scaling.instances_detected}</div>
                            </div>
                            <div className="text-center">
                                <div className="text-gray-600">Total Capacity</div>
                                <div className="font-medium">{backpressurePolicy.capacity_scaling.total_weighted_capacity}</div>
                            </div>
                            <div className="text-center">
                                <div className="text-gray-600">Utilization</div>
                                <div className={`font-medium ${
                                    backpressurePolicy.capacity_scaling.utilization_percent > 80 ? 'text-red-600' :
                                    backpressurePolicy.capacity_scaling.utilization_percent > 60 ? 'text-yellow-600' : 'text-green-600'
                                }`}>
                                    {backpressurePolicy.capacity_scaling.utilization_percent}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Circuit Breakers Section */}
            {showCircuitBreakers && Object.keys(circuitBreakers).length > 0 && (
                <div className="bg-white rounded border p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <Power className="w-4 h-4"/>
                        <span className="font-semibold text-xs">Circuit Breakers</span>
                        <span className={`text-xs px-1 rounded ${
                            circuitBreakerSummary.open_circuits > 0 ? 'bg-red-100 text-red-700' :
                                circuitBreakerSummary.half_open_circuits > 0 ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-green-100 text-green-700'
                        }`}>
                            {circuitBreakerSummary.closed_circuits}/{circuitBreakerSummary.total_circuits} healthy
                        </span>
                    </div>

                    {/* Circuit Breaker Overview */}
                    <div className="grid grid-cols-4 gap-3 mb-3">
                        <div className="flex items-center gap-2 p-2 bg-green-50 rounded">
                            <Power className="w-3 h-3 text-green-600"/>
                            <span className="text-xs">Closed</span>
                            <span className="font-bold text-green-700">{circuitBreakerSummary.closed_circuits}</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-yellow-50 rounded">
                            <Pause className="w-3 h-3 text-yellow-600"/>
                            <span className="text-xs">Half-Open</span>
                            <span className="font-bold text-yellow-700">{circuitBreakerSummary.half_open_circuits}</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-red-50 rounded">
                            <PowerOff className="w-3 h-3 text-red-600"/>
                            <span className="text-xs">Open</span>
                            <span className="font-bold text-red-700">{circuitBreakerSummary.open_circuits}</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                            <TrendingUp className="w-3 h-3 text-blue-600"/>
                            <span className="text-xs">Total</span>
                            <span className="font-bold text-blue-700">{circuitBreakerSummary.total_circuits}</span>
                        </div>
                    </div>

                    {/* Individual Circuit Breakers */}
                    <div className="border-t pt-2">
                        <div className="space-y-2">
                            {Object.entries(circuitBreakers).map(([name, stats]) => (
                                <div key={name} className={`p-2 rounded border ${getCircuitBreakerColor(stats.state)}`}>
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex items-center gap-2">
                                            <CircuitBreakerIcon state={stats.state}/>
                                            <span className="font-medium text-xs">{name.replace('_', ' ')}</span>
                                            <span className="text-xs opacity-75 uppercase">{stats.state}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {stats.state === 'open' && (
                                                <button
                                                    onClick={() => resetCircuitBreaker(name)}
                                                    className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600 transition-colors"
                                                    title="Reset circuit breaker"
                                                >
                                                    <RotateCcw className="w-3 h-3"/>
                                                </button>
                                            )}
                                            <span className="text-xs">
                                                {stats.consecutive_failures > 0 ? `${stats.consecutive_failures} fails` : 'OK'}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-4 gap-2 text-xs">
                                        <div className="text-center">
                                            <div className="text-gray-600">Requests</div>
                                            <div className="font-medium">{stats.total_requests}</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-gray-600">Failures</div>
                                            <div className="font-medium text-red-600">{stats.total_failures}</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-gray-600">Window</div>
                                            <div className="font-medium">{stats.current_window_failures || 0}</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-gray-600">Success Rate</div>
                                            <div className={`font-medium ${
                                                stats.total_requests > 0 ?
                                                    ((stats.total_requests - stats.total_failures) / stats.total_requests) > 0.8 ? 'text-green-600' :
                                                        ((stats.total_requests - stats.total_failures) / stats.total_requests) > 0.6 ? 'text-yellow-600' : 'text-red-600'
                                                    : 'text-gray-600'
                                            }`}>
                                                {stats.total_requests > 0 ?
                                                    Math.round(((stats.total_requests - stats.total_failures) / stats.total_requests) * 100) : 0}%
                                            </div>
                                        </div>
                                    </div>

                                    {/* Additional timing info */}
                                    {(stats.last_failure_time || stats.opened_at) && (
                                        <div className="mt-1 pt-1 border-t border-current border-opacity-20 text-xs text-gray-600">
                                            {stats.opened_at && (
                                                <span className="mr-3">
                                                    Opened: {formatTimestamp(stats.opened_at)}
                                                </span>
                                            )}
                                            {stats.last_failure_time && (
                                                <span>
                                                    Last failure: {formatTimestamp(stats.last_failure_time)}
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
            {/* Capacity Transparency Section */}
            {showCapacityTransparency && capacityTransparency && (
                <CapacityTransparencyPanel
                    capacityTransparency={capacityTransparency}
                    showDetails={showConfiguration}
                />
            )}

            {/* Keep ALL your existing throttling stats section exactly as is */}
            {renderThrottlingSection()}


            {/* Session Analytics */}
            {sessionAnalytics && (
                <div className="bg-white rounded border p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <BarChart3 className="w-4 h-4"/>
                        <span className="font-semibold text-xs">Session Analytics</span>
                        <span className="text-gray-500">(Last 24h)</span>
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                        <div className="text-center p-2 bg-green-50 rounded">
                            <div className="text-xs text-gray-600">Sessions/Hour</div>
                            <div className="font-bold text-green-700">{sessionAnalytics.sessions_per_hour || 0}</div>
                        </div>
                        <div className="text-center p-2 bg-blue-50 rounded">
                            <div className="text-xs text-gray-600">Total Today</div>
                            <div className="font-bold text-blue-700">{sessionAnalytics.total_today || 0}</div>
                        </div>
                        <div className="text-center p-2 bg-orange-50 rounded">
                            <div className="text-xs text-gray-600">Avg Duration</div>
                            <div className="font-bold text-orange-700">{sessionAnalytics.avg_duration || '0m'}</div>
                        </div>
                        <div className="text-center p-2 bg-purple-50 rounded">
                            <div className="text-xs text-gray-600">Active Now</div>
                            <div className="font-bold text-purple-700">{sessionAnalytics.active_sessions || 0}</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Instances Grid */}
            <div className="space-y-2">
                {Object.entries(instances).map(([instanceId, services]) => (
                    <div key={instanceId} className="bg-white rounded border p-2">
                        <div className="flex items-center justify-between mb-2">
                            <span className="font-semibold text-xs">{instanceId}</span>
                            <div className="flex gap-1">
                                {Object.values(services).map((service, i) => (
                                    <HealthIcon key={i} health={service.health}/>
                                ))}
                            </div>
                        </div>

                        <div className="grid grid-cols-1 gap-2">
                            {Object.entries(services).map(([serviceKey, service]) => (
                                <div key={serviceKey} className="p-1 bg-gray-50 rounded">
                                    <div className="flex items-center gap-2 text-xs mb-1">
                                        <ServiceIcon type={serviceKey}/>
                                        <span className="min-w-0 flex-1 truncate">{serviceKey.replace('_', '/')}</span>

                                        <div className="flex items-center gap-1">
                                            <span className="text-gray-600">
                                                {service.healthy_processes || 0}/{service.processes}proc
                                            </span>
                                            {service.missing_processes > 0 && (
                                                <span
                                                    className="text-red-500 text-xs">(-{service.missing_processes})</span>
                                            )}
                                        </div>

                                        <LoadBar load={service.load} capacity={service.capacity}/>
                                        <span className="text-gray-700 w-12 text-right text-xs">
                                            {service.load}/{service.capacity}cap
                                        </span>
                                        <HealthIcon health={service.health}/>
                                    </div>

                                    <div className="text-xs text-gray-500 ml-5 space-y-1">
                                        {service.pids && service.pids.map((pid, i) => {
                                            const status = service.health_statuses && service.health_statuses[i] || 'unknown';
                                            const statusColor = {
                                                'healthy': 'text-green-600',
                                                'degraded': 'text-yellow-600',
                                                'unhealthy': 'text-red-600',
                                                'stale': 'text-gray-400',
                                                'unknown': 'text-gray-400'
                                            }[status] || 'text-gray-400';

                                            return (
                                                <div key={pid} className="flex items-center justify-between">
                                                    <div className="flex items-center gap-1">
                                                        <span className="text-gray-400">PID {pid}</span>
                                                        {service.ports && service.ports[i] && (
                                                            <span className="text-blue-500">:{service.ports[i]}</span>
                                                        )}
                                                        <span
                                                            className="text-gray-600">({service.loads && service.loads[i] || 0})</span>
                                                        <span className={`text-xs font-medium ${statusColor}`}>
                                                            {status}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-gray-400">
                                                            {service.heartbeats && service.heartbeats[i] ? formatTimestamp(service.heartbeats[i]) : 'Unknown'}
                                                        </span>
                                                        <span
                                                            className={`${service.heartbeats && service.heartbeats[i] ? getHeartbeatColor(service.heartbeats[i]) : 'text-gray-400'} font-medium`}>
                                                            {service.heartbeats && service.heartbeats[i] ? formatHeartbeat(service.heartbeats[i]) : 'N/A'}
                                                        </span>
                                                    </div>
                                                </div>
                                            );
                                        })}

                                        {service.missing_processes > 0 && (
                                            <div className="flex items-center justify-between text-red-500">
                                                <div className="flex items-center gap-1">
                                                    <span
                                                        className="text-red-400">Missing {service.missing_processes} process{service.missing_processes > 1 ? 'es' : ''}</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-red-400">No heartbeat</span>
                                                    <span className="font-medium text-red-500">MISSING</span>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Enhanced Stats Footer */}
            <div className="mt-3 pt-2 border-t">
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Instances: {Object.keys(instances).length}</span>
                    <span>Services: {Object.values(instances).reduce((acc, inst) => acc + Object.keys(inst).length, 0)}</span>
                    <span>Processes: {Object.values(instances).reduce((acc, inst) =>
                        acc + Object.values(inst).reduce((s, svc) => s + (svc.processes || 0), 0), 0)}</span>
                    <span>System Load: {Math.round((globalStats.load / globalStats.capacity) * 100) || 0}%</span>
                    {enhancedQueueStats && (
                        <span>Queue Pressure: {Math.round(enhancedQueueStats.pressure_ratio * 100)}%</span>
                    )}
                    {throttlingStats && (
                        <span
                            className={throttleRate > 5 ? 'text-red-600' : throttleRate > 2 ? 'text-orange-600' : 'text-green-600'}>
                            Throttle: {throttleRate.toFixed(1)}%
                        </span>
                    )}
                    {/* Add circuit breaker info to footer */}
                    {circuitBreakerSummary.total_circuits > 0 && (
                        <span className={circuitBreakerSummary.open_circuits > 0 ? 'text-red-600' : 'text-green-600'}>
                            CB: {circuitBreakerSummary.closed_circuits}/{circuitBreakerSummary.total_circuits}
                        </span>
                    )}
                    {capacityTransparency?.instance_scaling && (
                        <span className={`${
                            capacityTransparency.instance_scaling.process_health_ratio < 0.8 ? 'text-red-600' :
                                capacityTransparency.instance_scaling.process_health_ratio < 0.9 ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                            Proc: {capacityTransparency.instance_scaling.total_healthy_processes}h/{capacityTransparency.instance_scaling.total_configured_processes}c
                        </span>
                    )}
                </div>
                <div className="flex justify-center text-xs text-gray-500">
                    <Clock className="w-3 h-3 mr-1"/>
                    Last updated: {lastUpdate ? lastUpdate.toLocaleString() : 'Never'}
                    {error && <span className="ml-2 text-red-500">• Connection Issues</span>}
                    {throttleRate > 5 && (
                        <span className="ml-2 text-red-500">• High Throttle Rate!</span>
                    )}
                    {circuitBreakerSummary.open_circuits > 0 && (
                        <span className="ml-2 text-red-500">• Circuit Breakers Open!</span>
                    )}
                    {backpressurePolicy?.current_effects.all_blocked && (
                        <span className="ml-2 text-red-500">• All Requests Blocked!</span>
                    )}
                    {backpressurePolicy?.current_effects.anonymous_blocked && !backpressurePolicy?.current_effects.all_blocked && (
                        <span className="ml-2 text-orange-500">• Anonymous Blocked</span>
                    )}
                    {capacityTransparency?.capacity_warnings && capacityTransparency.capacity_warnings.length > 0 && (
                        <span className="ml-2 text-red-500">• {capacityTransparency.capacity_warnings.length} Process Warning{capacityTransparency.capacity_warnings.length > 1 ? 's' : ''}!</span>
                    )}
                </div>

                {enhancedQueueStats && (
                    <div className="flex justify-center text-xs text-gray-500 mt-1">
                        <Globe className="w-3 h-3 mr-1"/>
                        Static Capacity: {enhancedQueueStats.weighted_max_capacity}
                        (base: {enhancedQueueStats.base_capacity_per_instance} × {enhancedQueueStats.instance_count} instances)
                        {capacityTransparency && (
                            <span className="text-blue-500 ml-1">
                                → Dynamic: {capacityTransparency.instance_scaling.total_system_capacity}
                            </span>
                        )}
                    </div>
                )}

                {throttlingStats && throttlingStats.total_requests > 0 && (
                    <div className="flex justify-center text-xs text-gray-500 mt-1">
                        <Shield className="w-3 h-3 mr-1"/>
                        Requests: {throttlingStats.total_requests} | Throttled: {throttlingStats.total_throttled} |
                        429s: {throttlingStats.rate_limit_429} | 503s: {throttlingStats.backpressure_503}
                    </div>
                )}

                {backpressurePolicy && (
                    <div className="flex justify-center text-xs text-gray-500 mt-1">
                        <Gauge className="w-3 h-3 mr-1"/>
                        Backpressure: {backpressurePolicy.current_effects.pressure_level} pressure |
                        Anon: {backpressurePolicy.thresholds.anonymous_threshold} |
                        Reg: {backpressurePolicy.thresholds.registered_threshold} |
                        Hard: {backpressurePolicy.thresholds.hard_limit_threshold}
                        {capacityTransparency && (
                            <span className="text-green-500 ml-1">(Based on {capacityTransparency.instance_scaling.total_healthy_processes} actual processes)</span>
                        )}
                    </div>
                )}
                {gatewayConfiguration && (
                    <div className="flex justify-center text-xs text-gray-500 mt-1">
                        <Layers className="w-3 h-3 mr-1"/>
                        Profile: {gatewayConfiguration.current_profile} |
                        Config: {gatewayConfiguration.service_capacity.concurrent_requests_per_instance}×{gatewayConfiguration.service_capacity.avg_processing_time_seconds}s |
                        Limits: A:{gatewayConfiguration.rate_limits.anonymous.hourly}/R:{gatewayConfiguration.rate_limits.registered.hourly}
                        {capacityTransparency?.capacity_metrics.health_metrics && (
                            <span className="text-blue-500 ml-1">
                                | Actual: {capacityTransparency.capacity_metrics.health_metrics.processes_vs_configured.healthy}×{capacityTransparency.capacity_metrics.configuration.configured_concurrent_per_process}
                            </span>
                        )}
                    </div>
                )}
                {capacityTransparency?.instance_scaling && (
                    <div className="flex justify-center text-xs text-gray-500 mt-1">
                        <Gauge className="w-3 h-3 mr-1"/>
                        Dynamic Capacity: {capacityTransparency.instance_scaling.total_concurrent_capacity} concurrent
                        ({capacityTransparency.instance_scaling.total_healthy_processes} healthy × {capacityTransparency.capacity_metrics.configuration.configured_concurrent_per_process} each) |
                        Queue: {capacityTransparency.instance_scaling.total_queue_capacity} |
                        Total: {capacityTransparency.instance_scaling.total_system_capacity}
                        {capacityTransparency.instance_scaling.total_healthy_processes < capacityTransparency.instance_scaling.total_configured_processes && (
                            <span className="text-red-500 ml-1">
                                (Missing: {capacityTransparency.instance_scaling.total_configured_processes - capacityTransparency.instance_scaling.total_healthy_processes} processes)
                            </span>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

// Create a SystemMonitorPanel component
export const SystemMonitorPanel = ({onClose}) => {
    return (
        <div className="h-full w-full bg-white border-l border-gray-200 flex flex-col">
            <div className="flex items-center justify-between p-3 bg-gray-50 border-b border-gray-200 flex-shrink-0">
                <h3 className="text-sm font-semibold flex items-center text-gray-900">
                    <Server size={16} className="mr-2 text-green-600"/>
                    System Monitor
                </h3>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-700"
                        title="Close monitor"
                    >
                        <X size={16}/>
                    </button>
                )}
            </div>

            <div className="flex-1 overflow-y-auto">
                <SystemMonitor/>
            </div>
        </div>
    );
};

export default SystemMonitor;