/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import React from 'react';
import {Activity, AlertTriangle, Gauge, Layers} from "lucide-react";

export const CapacityTransparencyPanel = ({ capacityTransparency, showDetails = false }) => {
    if (!capacityTransparency) return null;

    const metrics = capacityTransparency.capacity_metrics;
    const scaling = capacityTransparency.instance_scaling;
    const thresholds = capacityTransparency.threshold_breakdown;
    const warnings = capacityTransparency.capacity_warnings || [];

    // Check if we have actual vs configured data
    const hasActualData = metrics.actual_runtime && metrics.health_metrics;
    const healthMetrics = metrics.health_metrics;

    return (
        <div className="bg-white rounded border p-3 mb-3">
            <div className="flex items-center gap-2 mb-2">
                <Gauge className="w-4 h-4"/>
                <span className="font-semibold text-xs">Dynamic Capacity (Actual Processes)</span>
                <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                    LIVE CALCULATED
                </span>
                {warnings.length > 0 && (
                    <span className="text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
                        {warnings.length} WARNING{warnings.length > 1 ? 'S' : ''}
                    </span>
                )}
            </div>

            {/* Warnings Section */}
            {warnings.length > 0 && (
                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded">
                    <div className="text-xs text-red-700 font-medium mb-1">Capacity Warnings:</div>
                    <div className="space-y-1">
                        {warnings.map((warning, index) => (
                            <div key={index} className="text-xs text-red-600 flex items-center gap-1">
                                <AlertTriangle className="w-3 h-3"/>
                                {warning}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Configured vs Actual Process Comparison */}
            {hasActualData && (
                <div className="grid grid-cols-4 gap-3 mb-3">
                    <div className="p-2 bg-blue-50 rounded">
                        <div className="text-xs text-gray-600">Configured Processes</div>
                        <div className="font-bold text-blue-700">
                            {healthMetrics.processes_vs_configured.configured}
                        </div>
                        <div className="text-xs text-gray-500">expected</div>
                    </div>
                    <div className="p-2 bg-green-50 rounded">
                        <div className="text-xs text-gray-600">Actual Running</div>
                        <div className="font-bold text-green-700">
                            {healthMetrics.processes_vs_configured.actual}
                        </div>
                        <div className="text-xs text-gray-500">detected</div>
                    </div>
                    <div className="p-2 bg-orange-50 rounded">
                        <div className="text-xs text-gray-600">Healthy Processes</div>
                        <div className="font-bold text-orange-700">
                            {healthMetrics.processes_vs_configured.healthy}
                        </div>
                        <div className="text-xs text-gray-500">
                            {Math.round(healthMetrics.process_health_ratio * 100)}% health
                        </div>
                    </div>
                    <div className="p-2 bg-purple-50 rounded">
                        <div className="text-xs text-gray-600">Process Deficit</div>
                        <div className={`font-bold ${healthMetrics.processes_vs_configured.process_deficit > 0 ? 'text-red-700' : 'text-green-700'}`}>
                            {healthMetrics.processes_vs_configured.process_deficit > 0 ?
                                `-${healthMetrics.processes_vs_configured.process_deficit}` :
                                '✓ OK'
                            }
                        </div>
                        <div className="text-xs text-gray-500">missing</div>
                    </div>
                </div>
            )}

            {/* Actual Capacity Calculation */}
            {hasActualData && (
                <div className="border-t pt-2 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity className="w-3 h-3"/>
                        <span className="font-semibold text-xs">Actual Capacity (Based on Healthy Processes)</span>
                    </div>
                    <div className="grid grid-cols-4 gap-3">
                        <div className="p-2 bg-gray-50 rounded">
                            <div className="text-xs text-gray-600">Per Process</div>
                            <div className="font-bold text-gray-700">
                                {metrics.configuration.configured_concurrent_per_process}
                            </div>
                            <div className="text-xs text-gray-500">
                                {metrics.configuration.configured_avg_processing_time_seconds}s avg
                            </div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded">
                            <div className="text-xs text-gray-600">Actual Concurrent</div>
                            <div className="font-bold text-blue-700">
                                {metrics.actual_runtime.actual_concurrent_per_instance}
                            </div>
                            <div className="text-xs text-gray-500">
                                {healthMetrics.processes_vs_configured.healthy} × {metrics.configuration.configured_concurrent_per_process}
                            </div>
                        </div>
                        <div className="p-2 bg-green-50 rounded">
                            <div className="text-xs text-gray-600">Effective (After Buffer)</div>
                            <div className="font-bold text-green-700">
                                {metrics.actual_runtime.actual_effective_concurrent_per_instance}
                            </div>
                            <div className="text-xs text-gray-500">
                                -{metrics.configuration.capacity_buffer_percent}% buffer
                            </div>
                        </div>
                        <div className="p-2 bg-orange-50 rounded">
                            <div className="text-xs text-gray-600">Total Instance</div>
                            <div className="font-bold text-orange-700">
                                {metrics.actual_runtime.actual_total_capacity_per_instance}
                            </div>
                            <div className="text-xs text-gray-500">
                                +{metrics.actual_runtime.actual_queue_capacity_per_instance} queue
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* System-Wide Scaling (Multi-Instance) */}
            <div className="border-t pt-2 mb-3">
                <div className="text-xs text-gray-600 mb-1">System-Wide Scaling (All Instances):</div>
                <div className="grid grid-cols-4 gap-3">
                    <div className="text-center">
                        <div className="text-xs text-gray-600">Instances</div>
                        <div className="font-bold text-blue-600">{scaling.detected_instances}</div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-600">Total Healthy</div>
                        <div className="font-bold text-green-600">
                            {scaling.total_healthy_processes}
                            <span className="text-xs text-gray-500">/{scaling.total_actual_processes}</span>
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-600">System Concurrent</div>
                        <div className="font-bold text-orange-600">{scaling.total_concurrent_capacity}</div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-600">System Total</div>
                        <div className="font-bold text-purple-600">{scaling.total_system_capacity}</div>
                    </div>
                </div>

                {/* System health indicator */}
                <div className="mt-2 text-center">
                    <div className="text-xs text-gray-600">System Process Health:</div>
                    <div className={`font-medium ${
                        scaling.process_health_ratio > 0.8 ? 'text-green-600' :
                            scaling.process_health_ratio > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                        {Math.round(scaling.process_health_ratio * 100)}%
                    </div>
                </div>
            </div>

            {/* Dynamic Thresholds */}
            <div className="border-t pt-2">
                <div className="text-xs text-gray-600 mb-1">Dynamic Thresholds (Based on Actual Processes):</div>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-2 bg-blue-50 rounded text-center">
                        <div className="text-xs text-gray-600">Anonymous Blocks At</div>
                        <div className="font-bold text-blue-700">{thresholds.anonymous_blocks_at}</div>
                        <div className="text-xs text-gray-500">{thresholds.anonymous_percentage}% of actual</div>
                    </div>
                    <div className="p-2 bg-orange-50 rounded text-center">
                        <div className="text-xs text-gray-600">Registered Blocks At</div>
                        <div className="font-bold text-orange-700">{thresholds.registered_blocks_at}</div>
                        <div className="text-xs text-gray-500">{thresholds.registered_percentage}% of actual</div>
                    </div>
                    <div className="p-2 bg-red-50 rounded text-center">
                        <div className="text-xs text-gray-600">Hard Limit At</div>
                        <div className="font-bold text-red-700">{thresholds.hard_limit_at}</div>
                        <div className="text-xs text-gray-500">{thresholds.hard_limit_percentage}% of actual</div>
                    </div>
                </div>
            </div>

            {/* Configuration vs Reality Summary */}
            {hasActualData && (
                <div className="border-t pt-2 mt-2">
                    <div className="text-xs text-gray-600 mb-1">Configuration vs Reality:</div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                        <div>
                            <div className="text-gray-600 mb-1">Configured Capacity:</div>
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span>Per Instance:</span>
                                    <span className="font-medium">{metrics.configuration.configured_processes_per_instance} × {metrics.configuration.configured_concurrent_per_process} = {metrics.configuration.configured_processes_per_instance * metrics.configuration.configured_concurrent_per_process}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>System Total:</span>
                                    <span className="font-medium">{scaling.total_configured_processes} × {metrics.configuration.configured_concurrent_per_process} = {scaling.total_configured_processes * metrics.configuration.configured_concurrent_per_process}</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="text-gray-600 mb-1">Actual Capacity:</div>
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span>Per Instance:</span>
                                    <span className={`font-medium ${hasActualData && healthMetrics.processes_vs_configured.healthy < metrics.configuration.configured_processes_per_instance ? 'text-red-600' : 'text-green-600'}`}>
                                        {hasActualData ? healthMetrics.processes_vs_configured.healthy : '?'} × {metrics.configuration.configured_concurrent_per_process} = {hasActualData ? metrics.actual_runtime.actual_concurrent_per_instance : '?'}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>System Total:</span>
                                    <span className={`font-medium ${scaling.total_healthy_processes < scaling.total_configured_processes ? 'text-red-600' : 'text-green-600'}`}>
                                        {scaling.total_healthy_processes} × {metrics.configuration.configured_concurrent_per_process} = {scaling.total_concurrent_capacity}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
