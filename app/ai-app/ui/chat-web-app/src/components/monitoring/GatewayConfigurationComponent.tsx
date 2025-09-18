/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React from 'react';
import {
    Layers,
    Users,
    Scale,
    Power,
    Clock,
    Settings,
    Cpu,
    Zap,
    Shield,
    Timer,
    Gauge,
    Activity
} from 'lucide-react';

export const GatewayConfigurationComponent = ({ gatewayConfiguration, capacityTransparency }) => {
    if (!gatewayConfiguration) return null;

    const getProfileColor = (profile) => {
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

    // Extract capacity metrics if available
    const capacityMetrics = capacityTransparency?.capacity_metrics;
    const instanceScaling = capacityTransparency?.instance_scaling;
    const configuration = capacityMetrics?.configuration;
    const actualRuntime = capacityMetrics?.actual_runtime;
    const healthMetrics = capacityMetrics?.health_metrics;

    // Use capacity transparency data if available, otherwise fall back to gateway config
    const serviceCapacity = {
        concurrent_requests_per_instance: actualRuntime?.actual_concurrent_per_instance ||
                                        gatewayConfiguration.service_capacity?.concurrent_requests_per_instance || 0,
        avg_processing_time_seconds: configuration?.configured_avg_processing_time_seconds ||
                                   gatewayConfiguration.service_capacity?.avg_processing_time_seconds || 25.0,
        requests_per_hour: actualRuntime?.actual_theoretical_hourly_per_instance ||
                          gatewayConfiguration.service_capacity?.requests_per_hour || 0,
        concurrent_per_process: configuration?.configured_concurrent_per_process || 5,
        processes_per_instance: configuration?.configured_processes_per_instance || 1
    };

    return (
        <div className="bg-white rounded border p-3 mb-3">
            <div className="flex items-center gap-2 mb-2">
                <Layers className="w-4 h-4"/>
                <span className="font-semibold text-xs">Gateway Configuration</span>
                <span className={`px-2 py-1 rounded text-xs ${getProfileColor(gatewayConfiguration.current_profile)}`}>
                    {gatewayConfiguration.current_profile.toUpperCase()}
                </span>
                <span className="text-xs text-gray-500">({gatewayConfiguration.instance_id})</span>
                {capacityTransparency && (
                    <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                        DYNAMIC
                    </span>
                )}
            </div>

            {/* Instance & Profile Overview */}
            <div className="grid grid-cols-4 gap-3 mb-3">
                <div className="p-2 bg-blue-50 rounded">
                    <div className="text-xs text-gray-600">Instance</div>
                    <div className="font-bold text-blue-700 text-xs">{gatewayConfiguration.instance_id}</div>
                    <div className="text-xs text-gray-500">{gatewayConfiguration.tenant_id}</div>
                </div>
                <div className="p-2 bg-green-50 rounded">
                    <div className="text-xs text-gray-600">Profile</div>
                    <div className="font-bold text-green-700">{gatewayConfiguration.current_profile}</div>
                    <div className="text-xs text-gray-500">environment</div>
                </div>
                <div className="p-2 bg-orange-50 rounded">
                    <div className="text-xs text-gray-600">Per Process</div>
                    <div className="font-bold text-orange-700">{serviceCapacity.concurrent_per_process}</div>
                    <div className="text-xs text-gray-500">{serviceCapacity.processes_per_instance} processes</div>
                </div>
                <div className="p-2 bg-purple-50 rounded">
                    <div className="text-xs text-gray-600">Instance Total</div>
                    <div className="font-bold text-purple-700">{serviceCapacity.concurrent_requests_per_instance}</div>
                    <div className="text-xs text-gray-500">concurrent</div>
                </div>
            </div>

            {/* Service Capacity Configuration */}
            {capacityMetrics && (
                <div className="border-t pt-2 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <Cpu className="w-3 h-3"/>
                        <span className="font-semibold text-xs">Service Capacity (Multi-Process)</span>
                        {healthMetrics?.process_health_ratio < 1 && (
                            <span className="text-xs text-yellow-600 bg-yellow-50 px-1 rounded">
                                {Math.round(healthMetrics.process_health_ratio * 100)}% HEALTHY
                            </span>
                        )}
                    </div>
                    <div className="grid grid-cols-4 gap-3">
                        <div className="p-2 bg-gray-50 rounded">
                            <div className="text-xs text-gray-600 mb-1">Per Process Config</div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Concurrent:</span>
                                    <span className="font-medium">{serviceCapacity.concurrent_per_process}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span>Avg Time:</span>
                                    <span className="font-medium">{serviceCapacity.avg_processing_time_seconds}s</span>
                                </div>
                            </div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded">
                            <div className="text-xs text-gray-600 mb-1">Instance Scaling</div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Configured:</span>
                                    <span className="font-medium">{serviceCapacity.processes_per_instance}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span>Healthy:</span>
                                    <span className="font-medium">{healthMetrics?.processes_vs_configured?.healthy || serviceCapacity.processes_per_instance}</span>
                                </div>
                            </div>
                        </div>
                        <div className="p-2 bg-green-50 rounded">
                            <div className="text-xs text-gray-600 mb-1">Queue & Buffer</div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Queue Cap:</span>
                                    <span className="font-medium">{actualRuntime?.actual_queue_capacity_per_instance || 'N/A'}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span>Buffer:</span>
                                    <span className="font-medium">{configuration?.capacity_buffer_percent || 20}%</span>
                                </div>
                            </div>
                        </div>
                        <div className="p-2 bg-orange-50 rounded">
                            <div className="text-xs text-gray-600 mb-1">Throughput</div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Per Instance:</span>
                                    <span className="font-medium">{serviceCapacity.requests_per_hour}/hr</span>
                                </div>
                                {instanceScaling && (
                                    <div className="flex justify-between text-xs">
                                        <span>System Total:</span>
                                        <span className="font-medium">{instanceScaling.theoretical_system_hourly}/hr</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Rate Limits Configuration */}
            <div className="border-t pt-2 mb-3">
                <div className="flex items-center gap-2 mb-2">
                    <Users className="w-3 h-3"/>
                    <span className="font-semibold text-xs">Rate Limits</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-2 bg-gray-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Anonymous Users</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Hourly:</span>
                                <span className="font-medium">
                                    {gatewayConfiguration.rate_limits.anonymous.hourly === -1 ? '∞' : gatewayConfiguration.rate_limits.anonymous.hourly}
                                </span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Burst:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.anonymous.burst}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.anonymous.burst_window}s</span>
                            </div>
                        </div>
                    </div>
                    <div className="p-2 bg-blue-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Registered Users</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Hourly:</span>
                                <span className="font-medium">
                                    {gatewayConfiguration.rate_limits.registered.hourly === -1 ? '∞' : gatewayConfiguration.rate_limits.registered.hourly}
                                </span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Burst:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.registered.burst}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.registered.burst_window}s</span>
                            </div>
                        </div>
                    </div>
                    <div className="p-2 bg-purple-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Privileged Users</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Hourly:</span>
                                <span className="font-medium">
                                    {gatewayConfiguration.rate_limits.privileged.hourly === -1 ? '∞' : gatewayConfiguration.rate_limits.privileged.hourly}
                                </span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Burst:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.privileged.burst}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.rate_limits.privileged.burst_window}s</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Backpressure Thresholds */}
            <div className="border-t pt-2 mb-3">
                <div className="flex items-center gap-2 mb-2">
                    <Scale className="w-3 h-3"/>
                    <span className="font-semibold text-xs">Backpressure Thresholds</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-2 bg-blue-50 rounded">
                        <div className="text-xs text-gray-600">Anonymous Block</div>
                        <div className="font-bold text-blue-700">
                            {Math.round(gatewayConfiguration.backpressure_settings.anonymous_pressure_threshold * 100)}%
                        </div>
                        <div className="text-xs text-gray-500">of capacity</div>
                    </div>
                    <div className="p-2 bg-orange-50 rounded">
                        <div className="text-xs text-gray-600">Registered Block</div>
                        <div className="font-bold text-orange-700">
                            {Math.round(gatewayConfiguration.backpressure_settings.registered_pressure_threshold * 100)}%
                        </div>
                        <div className="text-xs text-gray-500">of capacity</div>
                    </div>
                    <div className="p-2 bg-red-50 rounded">
                        <div className="text-xs text-gray-600">Hard Limit</div>
                        <div className="font-bold text-red-700">
                            {Math.round(gatewayConfiguration.backpressure_settings.hard_limit_threshold * 100)}%
                        </div>
                        <div className="text-xs text-gray-500">of capacity</div>
                    </div>
                </div>

                {/* Additional backpressure settings */}
                <div className="mt-2 pt-2 border-t border-gray-400">
                    <div className="grid grid-cols-2 gap-3">
                        <div className="flex justify-between text-xs">
                            <span className="text-gray-600">Capacity Buffer:</span>
                            <span className="font-medium">{Math.round(gatewayConfiguration.backpressure_settings.capacity_buffer * 100)}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                            <span className="text-gray-600">Queue Depth Multiplier:</span>
                            <span className="font-medium">{gatewayConfiguration.backpressure_settings.queue_depth_multiplier}x</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Circuit Breaker Settings */}
            <div className="border-t pt-2 mb-3">
                <div className="flex items-center gap-2 mb-2">
                    <Power className="w-3 h-3"/>
                    <span className="font-semibold text-xs">Circuit Breaker Settings</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-2 bg-green-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Authentication</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Fail Threshold:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.authentication.failure_threshold}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Recovery:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.authentication.recovery_timeout}s</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.authentication.window_size}s</span>
                            </div>
                        </div>
                    </div>
                    <div className="p-2 bg-yellow-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Rate Limiter</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Fail Threshold:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.rate_limiter.failure_threshold}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Recovery:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.rate_limiter.recovery_timeout}s</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.rate_limiter.window_size}s</span>
                            </div>
                        </div>
                    </div>
                    <div className="p-2 bg-red-50 rounded">
                        <div className="text-xs text-gray-600 mb-1">Backpressure</div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span>Fail Threshold:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.backpressure.failure_threshold}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Recovery:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.backpressure.recovery_timeout}s</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span>Window:</span>
                                <span className="font-medium">{gatewayConfiguration.circuit_breaker_settings.backpressure.window_size}s</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Monitoring Settings */}
            <div className="border-t pt-2">
                <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-3 h-3"/>
                    <span className="font-semibold text-xs">Monitoring Settings</span>
                </div>
                <div className="grid grid-cols-5 gap-2 text-xs">
                    <div className="text-center">
                        <div className="text-gray-600">Events Retention</div>
                        <div className="font-medium">{gatewayConfiguration.monitoring_settings.throttling_events_retention_hours}h</div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-600">Session Analytics</div>
                        <div className={`font-medium ${gatewayConfiguration.monitoring_settings.session_analytics_enabled ? 'text-green-600' : 'text-red-600'}`}>
                            {gatewayConfiguration.monitoring_settings.session_analytics_enabled ? 'ON' : 'OFF'}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-600">Queue Analytics</div>
                        <div className={`font-medium ${gatewayConfiguration.monitoring_settings.queue_analytics_enabled ? 'text-green-600' : 'text-red-600'}`}>
                            {gatewayConfiguration.monitoring_settings.queue_analytics_enabled ? 'ON' : 'OFF'}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-600">Heartbeat Timeout</div>
                        <div className="font-medium">{gatewayConfiguration.monitoring_settings.heartbeat_timeout_seconds}s</div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-600">Cache TTL</div>
                        <div className="font-medium">{gatewayConfiguration.monitoring_settings.instance_cache_ttl_seconds}s</div>
                    </div>
                </div>
            </div>

            {/* Environment Sync Status */}
            {capacityMetrics && (
                <div className="border-t pt-2 mt-2">
                    <div className="flex items-center gap-2 mb-2">
                        <Settings className="w-3 h-3"/>
                        <span className="font-semibold text-xs">Environment Sync</span>
                        <span className="text-xs text-green-600 bg-green-50 px-1 rounded">LIVE</span>
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                        <div className="text-center">
                            <div className="text-gray-600">MAX_CONCURRENT_CHAT</div>
                            <div className="font-medium">{serviceCapacity.concurrent_per_process}</div>
                        </div>
                        <div className="text-center">
                            <div className="text-gray-600">CHAT_APP_PARALLELISM</div>
                            <div className="font-medium">{serviceCapacity.processes_per_instance}</div>
                        </div>
                        <div className="text-center">
                            <div className="text-gray-600">AVG_PROCESSING_TIME</div>
                            <div className="font-medium">{serviceCapacity.avg_processing_time_seconds}s</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
