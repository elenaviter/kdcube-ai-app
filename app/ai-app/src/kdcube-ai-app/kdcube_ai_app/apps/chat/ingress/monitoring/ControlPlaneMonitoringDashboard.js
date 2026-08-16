"use strict";
// Control Plane Monitoring Dashboard (TypeScript)
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
var react_1 = require("react");
var client_1 = require("react-dom/client");
var isRawGatewayConfigPayload = function (payload) {
    if (!payload || typeof payload !== 'object')
        return false;
    if (payload.raw_config || payload.gateway_config || payload.config)
        return true;
    if (payload.profile)
        return true;
    var sections = ['service_capacity', 'backpressure', 'rate_limits', 'pools', 'limits'];
    for (var _i = 0, sections_1 = sections; _i < sections_1.length; _i++) {
        var key = sections_1[_i];
        var value = payload[key];
        if (value && typeof value === 'object') {
            if ('ingress' in value || 'proc' in value || 'processor' in value || 'worker' in value) {
                return true;
            }
        }
    }
    return false;
};
// =============================================================================
// Settings Manager (same pattern as other widgets)
// =============================================================================
var SettingsManager = /** @class */ (function () {
    function SettingsManager() {
        this.PLACEHOLDER_BASE_URL = '{{' + 'CHAT_BASE_URL' + '}}';
        this.PLACEHOLDER_ACCESS_TOKEN = '{{' + 'ACCESS_TOKEN' + '}}';
        this.PLACEHOLDER_ID_TOKEN = '{{' + 'ID_TOKEN' + '}}';
        this.PLACEHOLDER_ID_TOKEN_HEADER = '{{' + 'ID_TOKEN_HEADER' + '}}';
        this.PLACEHOLDER_TENANT = '{{' + 'DEFAULT_TENANT' + '}}';
        this.PLACEHOLDER_PROJECT = '{{' + 'DEFAULT_PROJECT' + '}}';
        this.PLACEHOLDER_BUNDLE_ID = '{{' + 'DEFAULT_APP_BUNDLE_ID' + '}}';
        this.settings = {
            baseUrl: '{{CHAT_BASE_URL}}',
            accessToken: '{{ACCESS_TOKEN}}',
            idToken: '{{ID_TOKEN}}',
            idTokenHeader: '{{ID_TOKEN_HEADER}}',
            defaultTenant: '{{DEFAULT_TENANT}}',
            defaultProject: '{{DEFAULT_PROJECT}}',
            defaultAppBundleId: '{{DEFAULT_APP_BUNDLE_ID}}'
        };
        this.configReceivedCallback = null;
    }
    SettingsManager.prototype.getBaseUrl = function () {
        if (this.settings.baseUrl === this.PLACEHOLDER_BASE_URL) {
            return window.location.origin;
        }
        try {
            var url = new URL(this.settings.baseUrl);
            if (url.port === 'None' || url.hostname.includes('None')) {
                return window.location.origin;
            }
            var trimmed = this.settings.baseUrl.replace(/\/+$/, '');
            return trimmed.endsWith('/api') ? trimmed.slice(0, -4) : trimmed;
        }
        catch (e) {
            return window.location.origin;
        }
    };
    SettingsManager.prototype.getAccessToken = function () {
        if (this.settings.accessToken === this.PLACEHOLDER_ACCESS_TOKEN || !this.settings.accessToken) {
            return null;
        }
        return this.settings.accessToken;
    };
    SettingsManager.prototype.getIdToken = function () {
        if (this.settings.idToken === this.PLACEHOLDER_ID_TOKEN || !this.settings.idToken) {
            return null;
        }
        return this.settings.idToken;
    };
    SettingsManager.prototype.getIdTokenHeader = function () {
        return this.settings.idTokenHeader === this.PLACEHOLDER_ID_TOKEN_HEADER
            ? 'X-ID-Token'
            : this.settings.idTokenHeader;
    };
    SettingsManager.prototype.getDefaultTenant = function () {
        return this.settings.defaultTenant === this.PLACEHOLDER_TENANT
            ? 'home'
            : this.settings.defaultTenant;
    };
    SettingsManager.prototype.getDefaultProject = function () {
        return this.settings.defaultProject === this.PLACEHOLDER_PROJECT
            ? 'demo'
            : this.settings.defaultProject;
    };
    SettingsManager.prototype.hasPlaceholderSettings = function () {
        return this.settings.baseUrl === this.PLACEHOLDER_BASE_URL;
    };
    SettingsManager.prototype.updateSettings = function (partial) {
        this.settings = __assign(__assign({}, this.settings), partial);
    };
    SettingsManager.prototype.onConfigReceived = function (callback) {
        this.configReceivedCallback = callback;
    };
    SettingsManager.prototype.applyRuntimeConfig = function (config, options) {
        var _a, _b;
        if (options === void 0) { options = {}; }
        var tenant = config.defaultTenant || config.tenant || config.tenant_id;
        var project = config.defaultProject || config.project || config.project_id;
        var idTokenHeader = config.idTokenHeader || config.idTokenHeaderName || ((_a = config.auth) === null || _a === void 0 ? void 0 : _a.idTokenHeaderName);
        var updates = {};
        if (config.baseUrl && typeof config.baseUrl === 'string')
            updates.baseUrl = config.baseUrl;
        if (config.accessToken !== undefined)
            updates.accessToken = config.accessToken;
        if (config.idToken !== undefined)
            updates.idToken = config.idToken;
        if (idTokenHeader)
            updates.idTokenHeader = idTokenHeader;
        if (tenant)
            updates.defaultTenant = tenant;
        if (project)
            updates.defaultProject = project;
        if (config.defaultAppBundleId)
            updates.defaultAppBundleId = config.defaultAppBundleId;
        if (Object.keys(updates).length === 0)
            return false;
        this.updateSettings(updates);
        if (options.notify !== false)
            (_b = this.configReceivedCallback) === null || _b === void 0 ? void 0 : _b.call(this);
        return true;
    };
    SettingsManager.prototype.loadFrontendConfig = function () {
        return __awaiter(this, void 0, void 0, function () {
            var controller, timeout, response, config, _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        controller = new AbortController();
                        timeout = window.setTimeout(function () { return controller.abort(); }, 1000);
                        _b.label = 1;
                    case 1:
                        _b.trys.push([1, 4, 5, 6]);
                        return [4 /*yield*/, fetch("".concat(this.getBaseUrl(), "/api/cp-frontend-config"), {
                                method: 'GET',
                                credentials: 'include',
                                cache: 'no-store',
                                headers: { Accept: 'application/json' },
                                signal: controller.signal,
                            })];
                    case 2:
                        response = _b.sent();
                        if (!response.ok)
                            return [2 /*return*/, false];
                        return [4 /*yield*/, response.json()];
                    case 3:
                        config = _b.sent();
                        if (!config || typeof config !== 'object')
                            return [2 /*return*/, false];
                        return [2 /*return*/, this.applyRuntimeConfig(config, { notify: false })];
                    case 4:
                        _a = _b.sent();
                        return [2 /*return*/, false];
                    case 5:
                        window.clearTimeout(timeout);
                        return [7 /*endfinally*/];
                    case 6: return [2 /*return*/];
                }
            });
        });
    };
    SettingsManager.prototype.setupParentListener = function () {
        var _this = this;
        var identity = 'CONTROL_PLANE_MONITORING';
        window.addEventListener('message', function (event) {
            if (event.data.type === 'CONN_RESPONSE' || event.data.type === 'CONFIG_RESPONSE') {
                var requestedIdentity = event.data.identity;
                if (requestedIdentity !== identity) {
                    return;
                }
                if (event.data.config)
                    _this.applyRuntimeConfig(event.data.config);
            }
        });
        if (this.hasPlaceholderSettings()) {
            return new Promise(function (resolve) {
                var resolved = false;
                var finish = function (ready) {
                    if (resolved)
                        return;
                    resolved = true;
                    resolve(ready);
                };
                var requestParentConfig = function () {
                    window.parent.postMessage({
                        type: 'CONFIG_REQUEST',
                        data: {
                            requestedFields: [
                                'baseUrl', 'accessToken', 'idToken', 'idTokenHeader',
                                'defaultTenant', 'defaultProject', 'defaultAppBundleId'
                            ],
                            identity: identity
                        }
                    }, '*');
                    var timeout = window.setTimeout(function () { return finish(false); }, 3000);
                    var originalCallback = _this.configReceivedCallback;
                    _this.onConfigReceived(function () {
                        window.clearTimeout(timeout);
                        if (originalCallback)
                            originalCallback();
                        finish(true);
                    });
                };
                void _this.loadFrontendConfig().then(function (loaded) {
                    if (loaded) {
                        finish(true);
                    }
                    else {
                        requestParentConfig();
                    }
                });
            });
        }
        return Promise.resolve(!this.hasPlaceholderSettings());
    };
    return SettingsManager;
}());
var settings = new SettingsManager();
function appendAuthHeaders(headers) {
    var accessToken = settings.getAccessToken();
    var idToken = settings.getIdToken();
    var idTokenHeader = settings.getIdTokenHeader();
    if (accessToken) {
        headers.set('Authorization', "Bearer ".concat(accessToken));
    }
    if (idToken) {
        headers.set(idTokenHeader, idToken);
    }
    return headers;
}
function makeAuthHeaders(base) {
    var headers = new Headers(base);
    return appendAuthHeaders(headers);
}
// =============================================================================
// API Client
// =============================================================================
var MonitoringAPI = /** @class */ (function () {
    function MonitoringAPI(basePath) {
        if (basePath === void 0) { basePath = ''; }
        this.basePath = basePath;
    }
    MonitoringAPI.prototype.url = function (path) {
        return "".concat(settings.getBaseUrl()).concat(this.basePath).concat(path);
    };
    //
    // private url(path: string): string {
    //     return `${this.baseUrl}${path}`;
    // }
    MonitoringAPI.prototype.getSystemStatus = function () {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/monitoring/system'), {
                            method: 'GET',
                            headers: makeAuthHeaders(),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Failed to load system status (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.getCircuitBreakers = function () {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/circuit-breakers'), {
                            method: 'GET',
                            headers: makeAuthHeaders(),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Failed to load circuit breakers (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.resetCircuitBreaker = function (name) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url("/admin/circuit-breakers/".concat(name, "/reset")), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Failed to reset circuit breaker (".concat(res.status, ")"));
                        return [2 /*return*/];
                }
            });
        });
    };
    MonitoringAPI.prototype.validateGatewayConfig = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/gateway/validate-config'), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(payload),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Validation failed (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.updateGatewayConfig = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/gateway/update-config'), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(payload),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Update failed (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.resetGatewayConfig = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/gateway/reset-config'), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(payload),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Reset failed (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.clearGatewayConfigCache = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/gateway/clear-cache'), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(payload),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Clear cache failed (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.resetThrottling = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/throttling/reset'), {
                            method: 'POST',
                            headers: makeAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(payload),
                        })];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Reset throttling failed (".concat(res.status, ")"));
                        return [2 /*return*/, res.json()];
                }
            });
        });
    };
    MonitoringAPI.prototype.getBurstUsers = function () {
        return __awaiter(this, void 0, void 0, function () {
            var res, data, _1, detail;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, fetch(this.url('/admin/burst/users'), {
                            method: 'GET',
                            headers: makeAuthHeaders(),
                        })];
                    case 1:
                        res = _a.sent();
                        data = null;
                        _a.label = 2;
                    case 2:
                        _a.trys.push([2, 4, , 5]);
                        return [4 /*yield*/, res.json()];
                    case 3:
                        data = _a.sent();
                        return [3 /*break*/, 5];
                    case 4:
                        _1 = _a.sent();
                        data = null;
                        return [3 /*break*/, 5];
                    case 5:
                        if (!res.ok) {
                            detail = (data === null || data === void 0 ? void 0 : data.detail) || (data === null || data === void 0 ? void 0 : data.message);
                            throw new Error(detail ? "Burst users: ".concat(detail) : "Failed to load burst users (".concat(res.status, ")"));
                        }
                        return [2 /*return*/, data];
                }
            });
        });
    };
    return MonitoringAPI;
}());
// =============================================================================
// UI Components (simple, neutral palette)
// =============================================================================
var Card = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"bg-white rounded-2xl shadow-sm border border-gray-200/70 ".concat(className)}>
        {children}
    </div>);
};
var CapacityPanel = function (_a) {
    var _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _2, _3, _4, _5, _6, _7;
    var capacity = _a.capacity, dbConnections = _a.dbConnections, capacitySource = _a.capacitySource, capacitySourceActual = _a.capacitySourceActual, capacitySourceHealthy = _a.capacitySourceHealthy;
    if (!capacity)
        return null;
    var metrics = capacity.capacity_metrics || {};
    var scaling = capacity.instance_scaling || {};
    var thresholds = capacity.threshold_breakdown || {};
    var warnings = capacity.capacity_warnings || [];
    var hasActual = metrics.actual_runtime && metrics.health_metrics;
    var health = metrics.health_metrics || {};
    var actualProcesses = (_c = capacitySourceActual !== null && capacitySourceActual !== void 0 ? capacitySourceActual : (_b = health.processes_vs_configured) === null || _b === void 0 ? void 0 : _b.actual) !== null && _c !== void 0 ? _c : 0;
    var configuredProcesses = (_e = (_d = health.processes_vs_configured) === null || _d === void 0 ? void 0 : _d.configured) !== null && _e !== void 0 ? _e : 0;
    var healthyProcesses = (_g = capacitySourceHealthy !== null && capacitySourceHealthy !== void 0 ? capacitySourceHealthy : (_f = health.processes_vs_configured) === null || _f === void 0 ? void 0 : _f.healthy) !== null && _g !== void 0 ? _g : 0;
    return (<Card>
            <CardHeader title="Capacity Transparency" subtitle={"Capacity source: ".concat(capacitySource || 'unknown', ". Actual runtime vs configured capacity.")}/>
            <CardBody className="space-y-4">
                <Legend>
                    Compares configured worker counts to live heartbeats from the capacity source component.
                </Legend>
                {(dbConnections === null || dbConnections === void 0 ? void 0 : dbConnections.warning) ? (<div className="p-3 rounded-xl border border-rose-200 bg-rose-50 text-rose-800 text-sm">
                        <div className="font-semibold">DB connection capacity warning</div>
                        <div>
                            {dbConnections.warning_reason || 'Estimated DB connections are close to max_connections.'}
                            {dbConnections.percent_of_max != null ? " (".concat(dbConnections.percent_of_max, "% of max)") : ''}
                        </div>
                        <div className="text-[11px] text-rose-700 mt-1">
                            estimated_total={(_h = dbConnections.estimated_total) !== null && _h !== void 0 ? _h : '—'} · max_connections={(_j = dbConnections.max_connections) !== null && _j !== void 0 ? _j : '—'} ·
                            pool_per_worker={(_k = dbConnections.pool_max_per_worker) !== null && _k !== void 0 ? _k : '—'} · processes_per_instance={(_l = dbConnections.processes_per_instance) !== null && _l !== void 0 ? _l : '—'}
                        </div>
                        <div className="text-[11px] text-rose-700">
                            source={dbConnections.source || 'unknown'}
                        </div>
                    </div>) : null}
                {actualProcesses === 0 ? (<div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm">
                        No capacity-source processes detected. Start the capacity source service (usually `proc`) or
                        align configured worker counts with the running service.
                    </div>) : warnings.length > 0 && (<div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-sm">
                        {warnings.map(function (w, i) { return (<div key={i}>• {w}</div>); })}
                    </div>)}

                {hasActual && actualProcesses > 0 && (<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Configured</div>
                            <div className="text-sm font-semibold">{configuredProcesses !== null && configuredProcesses !== void 0 ? configuredProcesses : '—'}</div>
                            <div className="text-xs text-gray-500">processes</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Actual</div>
                            <div className="text-sm font-semibold">{actualProcesses !== null && actualProcesses !== void 0 ? actualProcesses : '—'}</div>
                            <div className="text-xs text-gray-500">running</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Healthy</div>
                            <div className="text-sm font-semibold">{healthyProcesses !== null && healthyProcesses !== void 0 ? healthyProcesses : '—'}</div>
                            <div className="text-xs text-gray-500">{Math.round(((_m = health.process_health_ratio) !== null && _m !== void 0 ? _m : 0) * 100)}% health</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Process Deficit</div>
                            <div className="text-sm font-semibold">{(_p = (_o = health.processes_vs_configured) === null || _o === void 0 ? void 0 : _o.process_deficit) !== null && _p !== void 0 ? _p : 0}</div>
                            <div className="text-xs text-gray-500">missing</div>
                        </div>
                    </div>)}

                {metrics.actual_runtime && metrics.configuration && actualProcesses > 0 && (<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Per Process</div>
                            <div className="text-sm font-semibold">{(_q = metrics.configuration.configured_concurrent_per_process) !== null && _q !== void 0 ? _q : '—'}</div>
                            <div className="text-xs text-gray-500">{(_r = metrics.configuration.configured_avg_processing_time_seconds) !== null && _r !== void 0 ? _r : '—'}s avg</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Actual Concurrent</div>
                            <div className="text-sm font-semibold">{(_s = metrics.actual_runtime.actual_concurrent_per_instance) !== null && _s !== void 0 ? _s : '—'}</div>
                            <div className="text-xs text-gray-500">per instance</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Effective</div>
                            <div className="text-sm font-semibold">{(_t = metrics.actual_runtime.actual_effective_concurrent_per_instance) !== null && _t !== void 0 ? _t : '—'}</div>
                            <div className="text-xs text-gray-500">after buffer</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Total Capacity</div>
                            <div className="text-sm font-semibold">{(_u = metrics.actual_runtime.actual_total_capacity_per_instance) !== null && _u !== void 0 ? _u : '—'}</div>
                            <div className="text-xs text-gray-500">per instance</div>
                        </div>
                    </div>)}

                {scaling && actualProcesses > 0 && (<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Instances</div>
                            <div className="text-sm font-semibold">{(_v = scaling.detected_instances) !== null && _v !== void 0 ? _v : '—'}</div>
                            <div className="text-xs text-gray-500">detected</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">System Concurrent</div>
                            <div className="text-sm font-semibold">{(_w = scaling.total_concurrent_capacity) !== null && _w !== void 0 ? _w : '—'}</div>
                            <div className="text-xs text-gray-500">total</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">System Total</div>
                            <div className="text-sm font-semibold">{(_x = scaling.total_system_capacity) !== null && _x !== void 0 ? _x : '—'}</div>
                            <div className="text-xs text-gray-500">capacity</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Health Ratio</div>
                            <div className="text-sm font-semibold">{Math.round(((_y = scaling.process_health_ratio) !== null && _y !== void 0 ? _y : 0) * 100)}%</div>
                            <div className="text-xs text-gray-500">system</div>
                        </div>
                    </div>)}

                {thresholds && (<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Anonymous Blocks At</div>
                            <div className="text-sm font-semibold">{(_z = thresholds.anonymous_blocks_at) !== null && _z !== void 0 ? _z : '—'}</div>
                            <div className="text-xs text-gray-500">{(_0 = thresholds.anonymous_percentage) !== null && _0 !== void 0 ? _0 : '—'}%</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Registered Blocks At</div>
                            <div className="text-sm font-semibold">{(_2 = thresholds.registered_blocks_at) !== null && _2 !== void 0 ? _2 : '—'}</div>
                            <div className="text-xs text-gray-500">{(_3 = thresholds.registered_percentage) !== null && _3 !== void 0 ? _3 : '—'}%</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Paid Blocks At</div>
                            <div className="text-sm font-semibold">{(_4 = thresholds.paid_blocks_at) !== null && _4 !== void 0 ? _4 : '—'}</div>
                            <div className="text-xs text-gray-500">{(_5 = thresholds.paid_percentage) !== null && _5 !== void 0 ? _5 : '—'}%</div>
                        </div>
                        <div className="p-3 rounded-xl bg-gray-100">
                            <div className="text-xs text-gray-600">Hard Limit At</div>
                            <div className="text-sm font-semibold">{(_6 = thresholds.hard_limit_at) !== null && _6 !== void 0 ? _6 : '—'}</div>
                            <div className="text-xs text-gray-500">{(_7 = thresholds.hard_limit_percentage) !== null && _7 !== void 0 ? _7 : '—'}%</div>
                        </div>
                    </div>)}
            </CardBody>
        </Card>);
};
var LatencyTable = function (_a) {
    var _b, _c;
    var title = _a.title, data = _a.data, _d = _a.compact, compact = _d === void 0 ? false : _d, _e = _a.showMax, showMax = _e === void 0 ? true : _e, _f = _a.className, className = _f === void 0 ? '' : _f;
    var windows = ["1m", "15m", "1h"];
    var padding = compact ? 'p-3' : 'p-4';
    var titleClass = compact ? 'text-xs font-semibold mb-2' : 'text-sm font-semibold mb-2';
    if (!data) {
        return (<div className={"".concat(padding, " rounded-xl bg-gray-100 ").concat(className)}>
                <div className={titleClass}>{title}</div>
                <div className="text-xs text-gray-500">No samples yet.</div>
            </div>);
    }
    return (<div className={"".concat(padding, " rounded-xl bg-gray-100 ").concat(className)}>
            <div className={titleClass}>{title}</div>
            <div className="grid grid-cols-4 gap-2 text-[11px] text-gray-600">
                <div className="font-semibold">Window</div>
                <div className="font-semibold">p50</div>
                <div className="font-semibold">p95</div>
                <div className="font-semibold">p99</div>
                {windows.map(function (w) {
            var _a, _b, _c, _d, _e, _f;
            return (<react_1.default.Fragment key={w}>
                        <div>{w}</div>
                        <div>{(_b = (_a = data === null || data === void 0 ? void 0 : data[w]) === null || _a === void 0 ? void 0 : _a.p50) !== null && _b !== void 0 ? _b : '—'}</div>
                        <div>{(_d = (_c = data === null || data === void 0 ? void 0 : data[w]) === null || _c === void 0 ? void 0 : _c.p95) !== null && _d !== void 0 ? _d : '—'}</div>
                        <div>{(_f = (_e = data === null || data === void 0 ? void 0 : data[w]) === null || _e === void 0 ? void 0 : _e.p99) !== null && _f !== void 0 ? _f : '—'}</div>
                    </react_1.default.Fragment>);
        })}
            </div>
            {showMax && (<div className="text-[11px] text-gray-500 mt-2">max (1h): {(_c = (_b = data === null || data === void 0 ? void 0 : data["1h"]) === null || _b === void 0 ? void 0 : _b.max) !== null && _c !== void 0 ? _c : '—'} ms</div>)}
        </div>);
};
var CardHeader = function (_a) {
    var title = _a.title, subtitle = _a.subtitle, action = _a.action;
    return (<div className="px-4 py-3 border-b border-gray-200/70">
        <div className="flex items-start justify-between gap-4">
            <div>
                <h2 className="text-base font-semibold text-gray-900">{title}</h2>
                {subtitle && <p className="mt-1 text-xs text-gray-600 leading-relaxed">{subtitle}</p>}
            </div>
            {action && <div className="pt-1">{action}</div>}
        </div>
    </div>);
};
var CardBody = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"px-4 py-3 ".concat(className)}>
        {children}
    </div>);
};
var Button = function (_a) {
    var children = _a.children, onClick = _a.onClick, _b = _a.type, type = _b === void 0 ? 'button' : _b, _c = _a.variant, variant = _c === void 0 ? 'primary' : _c, _d = _a.disabled, disabled = _d === void 0 ? false : _d, _e = _a.className, className = _e === void 0 ? '' : _e;
    var variants = {
        primary: 'bg-gray-900 hover:bg-gray-800 text-white',
        secondary: 'bg-white hover:bg-gray-50 text-gray-900 border border-gray-200/80',
        danger: 'bg-rose-600 hover:bg-rose-700 text-white',
    };
    return (<button type={type} onClick={onClick} disabled={disabled} className={"px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ".concat(variants[variant], " ").concat(className)}>
            {children}
        </button>);
};
var Input = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, placeholder = _a.placeholder, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={className}>
        {label && <label className="block text-xs font-medium text-gray-800 mb-1.5">{label}</label>}
        <input type="text" value={value} onChange={onChange} placeholder={placeholder} className="w-full px-3 py-1.5 border border-gray-200/80 rounded-lg bg-white text-xs focus:ring-2 focus:ring-gray-900/10 focus:border-gray-300 transition-colors placeholder:text-gray-400"/>
    </div>);
};
var TextArea = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={className}>
        {label && <label className="block text-xs font-medium text-gray-800 mb-1.5">{label}</label>}
        <textarea value={value} onChange={onChange} rows={10} className="w-full px-3 py-2 border border-gray-200/80 rounded-lg bg-white font-mono text-xs leading-relaxed focus:ring-2 focus:ring-gray-900/10 focus:border-gray-300"/>
    </div>);
};
var Pill = function (_a) {
    var _b = _a.tone, tone = _b === void 0 ? 'neutral' : _b, children = _a.children;
    var tones = {
        neutral: 'bg-gray-100 text-gray-700',
        success: 'bg-emerald-100 text-emerald-700',
        warning: 'bg-amber-100 text-amber-700',
        danger: 'bg-rose-100 text-rose-700',
    };
    return <span className={"px-2 py-0.5 rounded-full text-[10px] font-semibold ".concat(tones[tone])}>{children}</span>;
};
var Legend = function (_a) {
    var children = _a.children;
    return (<div className="text-[11px] text-gray-500 mb-3">Legend: {children}</div>);
};
var MonitoringDashboard = function () {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24;
    var api = (0, react_1.useMemo)(function () { return new MonitoringAPI(); }, []);
    var _25 = (0, react_1.useState)(null), system = _25[0], setSystem = _25[1];
    var _26 = (0, react_1.useState)({}), circuitBreakers = _26[0], setCircuitBreakers = _26[1];
    var _27 = (0, react_1.useState)(null), circuitSummary = _27[0], setCircuitSummary = _27[1];
    var _28 = (0, react_1.useState)(false), loading = _28[0], setLoading = _28[1];
    var _29 = (0, react_1.useState)(null), error = _29[0], setError = _29[1];
    var _30 = (0, react_1.useState)(true), autoRefresh = _30[0], setAutoRefresh = _30[1];
    var _31 = (0, react_1.useState)(null), lastUpdate = _31[0], setLastUpdate = _31[1];
    var _32 = (0, react_1.useState)(settings.getDefaultTenant()), tenant = _32[0], setTenant = _32[1];
    var _33 = (0, react_1.useState)(settings.getDefaultProject()), project = _33[0], setProject = _33[1];
    var _34 = (0, react_1.useState)(false), dryRun = _34[0], setDryRun = _34[1];
    var _35 = (0, react_1.useState)('ingress'), selectedComponent = _35[0], setSelectedComponent = _35[1];
    var _36 = (0, react_1.useState)(''), configJson = _36[0], setConfigJson = _36[1];
    var _37 = (0, react_1.useState)(null), validationResult = _37[0], setValidationResult = _37[1];
    var _38 = (0, react_1.useState)(null), actionMessage = _38[0], setActionMessage = _38[1];
    var _39 = (0, react_1.useState)(''), resetSessionId = _39[0], setResetSessionId = _39[1];
    var _40 = (0, react_1.useState)(false), resetAllSessions = _40[0], setResetAllSessions = _40[1];
    var _41 = (0, react_1.useState)(true), resetRateLimits = _41[0], setResetRateLimits = _41[1];
    var _42 = (0, react_1.useState)(true), resetBackpressure = _42[0], setResetBackpressure = _42[1];
    var _43 = (0, react_1.useState)(false), resetThrottlingStats = _43[0], setResetThrottlingStats = _43[1];
    var _44 = (0, react_1.useState)(false), purgeChatQueues = _44[0], setPurgeChatQueues = _44[1];
    var _45 = (0, react_1.useState)(false), resettingThrottling = _45[0], setResettingThrottling = _45[1];
    var _46 = (0, react_1.useState)(null), resetThrottlingMessage = _46[0], setResetThrottlingMessage = _46[1];
    var _47 = (0, react_1.useState)(null), clearCacheMessage = _47[0], setClearCacheMessage = _47[1];
    var _48 = (0, react_1.useState)(null), burstUsers = _48[0], setBurstUsers = _48[1];
    var _49 = (0, react_1.useState)(null), burstError = _49[0], setBurstError = _49[1];
    var _50 = (0, react_1.useState)(null), burstStatus = _50[0], setBurstStatus = _50[1];
    var _51 = (0, react_1.useState)('10'), burstAdminCount = _51[0], setBurstAdminCount = _51[1];
    var _52 = (0, react_1.useState)('10'), burstRegisteredCount = _52[0], setBurstRegisteredCount = _52[1];
    var _53 = (0, react_1.useState)('1'), burstMessagesPerUser = _53[0], setBurstMessagesPerUser = _53[1];
    var _54 = (0, react_1.useState)('10'), burstConcurrency = _54[0], setBurstConcurrency = _54[1];
    var _55 = (0, react_1.useState)('ping'), burstMessage = _55[0], setBurstMessage = _55[1];
    var _56 = (0, react_1.useState)(''), burstBundleId = _56[0], setBurstBundleId = _56[1];
    var _57 = (0, react_1.useState)(0), burstOpenCount = _57[0], setBurstOpenCount = _57[1];
    var _58 = (0, react_1.useState)(false), burstRunning = _58[0], setBurstRunning = _58[1];
    var burstSessionsRef = (0, react_1.useRef)([]);
    var _59 = (0, react_1.useState)('10'), plannerAdmins = _59[0], setPlannerAdmins = _59[1];
    var _60 = (0, react_1.useState)('15'), plannerRegistered = _60[0], setPlannerRegistered = _60[1];
    var _61 = (0, react_1.useState)('15'), plannerPaid = _61[0], setPlannerPaid = _61[1];
    var _62 = (0, react_1.useState)('12'), plannerPageLoad = _62[0], setPlannerPageLoad = _62[1];
    var _63 = (0, react_1.useState)('10'), plannerTabs = _63[0], setPlannerTabs = _63[1];
    var _64 = (0, react_1.useState)('10'), plannerPageWindow = _64[0], setPlannerPageWindow = _64[1];
    var _65 = (0, react_1.useState)('1.5'), plannerSafety = _65[0], setPlannerSafety = _65[1];
    var _66 = (0, react_1.useState)('5'), plannerConcurrentPerProcess = _66[0], setPlannerConcurrentPerProcess = _66[1];
    var _67 = (0, react_1.useState)('1'), plannerProcessesPerInstance = _67[0], setPlannerProcessesPerInstance = _67[1];
    var _68 = (0, react_1.useState)('25'), plannerAvgProcessing = _68[0], setPlannerAvgProcessing = _68[1];
    var _69 = (0, react_1.useState)('1'), plannerInstances = _69[0], setPlannerInstances = _69[1];
    var plannerInitializedRef = (0, react_1.useRef)(false);
    var gatewayCacheKeyPattern = "".concat(tenant || '<tenant>', ":").concat(project || '<project>', ":kdcube:config:gateway:current");
    var refreshAll = (0, react_1.useCallback)(function () { return __awaiter(void 0, void 0, void 0, function () {
        var _a, sys, cb, e_1;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    setLoading(true);
                    setError(null);
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, Promise.all([
                            api.getSystemStatus(),
                            api.getCircuitBreakers(),
                        ])];
                case 2:
                    _a = _b.sent(), sys = _a[0], cb = _a[1];
                    setSystem(sys);
                    setCircuitBreakers(cb.circuits || {});
                    setCircuitSummary(cb.summary || null);
                    setLastUpdate(new Date().toLocaleTimeString());
                    return [3 /*break*/, 5];
                case 3:
                    e_1 = _b.sent();
                    setError((e_1 === null || e_1 === void 0 ? void 0 : e_1.message) || 'Failed to load monitoring data');
                    return [3 /*break*/, 5];
                case 4:
                    setLoading(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); }, [api]);
    var loadBurstUsers = (0, react_1.useCallback)(function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, e_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    return [4 /*yield*/, api.getBurstUsers()];
                case 1:
                    res = _a.sent();
                    setBurstUsers(res);
                    setBurstError(null);
                    return [3 /*break*/, 3];
                case 2:
                    e_2 = _a.sent();
                    setBurstUsers(null);
                    setBurstError((e_2 === null || e_2 === void 0 ? void 0 : e_2.message) || 'Failed to load burst users');
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); }, [api]);
    (0, react_1.useEffect)(function () {
        var mounted = true;
        settings.setupParentListener().then(function () {
            if (mounted) {
                refreshAll();
                loadBurstUsers();
            }
        });
        return function () { mounted = false; };
    }, [refreshAll, loadBurstUsers]);
    (0, react_1.useEffect)(function () {
        if (!autoRefresh)
            return;
        var t = setInterval(function () { return refreshAll(); }, 5000);
        return function () { return clearInterval(t); };
    }, [autoRefresh, refreshAll]);
    var queue = system === null || system === void 0 ? void 0 : system.queue_stats;
    var capacityCtx = ((_a = system === null || system === void 0 ? void 0 : system.queue_stats) === null || _a === void 0 ? void 0 : _a.capacity_context) || {};
    var queueAnalytics = system === null || system === void 0 ? void 0 : system.queue_analytics;
    var queueUtilization = system === null || system === void 0 ? void 0 : system.queue_utilization;
    var throttling = system === null || system === void 0 ? void 0 : system.throttling_stats;
    var events = (system === null || system === void 0 ? void 0 : system.recent_throttling_events) || [];
    var lastThrottle = events.length ? events[0] : null;
    var gateway = system === null || system === void 0 ? void 0 : system.gateway_configuration;
    var throttlingByPeriod = (system === null || system === void 0 ? void 0 : system.throttling_by_period) || {};
    var throttlingWindows = (system === null || system === void 0 ? void 0 : system.throttling_windows) || {};
    var sseStats = system === null || system === void 0 ? void 0 : system.sse_connections;
    var components = (system === null || system === void 0 ? void 0 : system.components) || {};
    var autoscaler = (system === null || system === void 0 ? void 0 : system.autoscaler) || {};
    var configSource = (system === null || system === void 0 ? void 0 : system.gateway_config_source) || 'unknown';
    var configRaw = system === null || system === void 0 ? void 0 : system.gateway_config_raw;
    var configComponents = (system === null || system === void 0 ? void 0 : system.gateway_config_components) || {};
    var capacitySource = ((_b = configRaw === null || configRaw === void 0 ? void 0 : configRaw.backpressure) === null || _b === void 0 ? void 0 : _b.capacity_source_component)
        || ((_d = (_c = configComponents === null || configComponents === void 0 ? void 0 : configComponents.ingress) === null || _c === void 0 ? void 0 : _c.backpressure) === null || _d === void 0 ? void 0 : _d.capacity_source_component)
        || ((_f = (_e = configComponents === null || configComponents === void 0 ? void 0 : configComponents.proc) === null || _e === void 0 ? void 0 : _e.backpressure) === null || _f === void 0 ? void 0 : _f.capacity_source_component)
        || (configRaw === null || configRaw === void 0 ? void 0 : configRaw.capacity_source_component);
    var capacitySourceKey = (0, react_1.useMemo)(function () {
        var raw = (capacitySource || '').toLowerCase();
        if (raw.includes('proc'))
            return 'proc';
        if (raw.includes('rest') || raw.includes('ingress'))
            return 'ingress';
        if (raw.startsWith('chat:proc'))
            return 'proc';
        if (raw.startsWith('chat:rest'))
            return 'ingress';
        return raw || 'proc';
    }, [capacitySource]);
    var capacitySourceActual = (_g = components === null || components === void 0 ? void 0 : components[capacitySourceKey]) === null || _g === void 0 ? void 0 : _g.actual_processes;
    var capacitySourceHealthy = (_h = components === null || components === void 0 ? void 0 : components[capacitySourceKey]) === null || _h === void 0 ? void 0 : _h.healthy_processes;
    var plannerComponentKey = capacitySourceKey || 'proc';
    var poolAggregateEntries = (0, react_1.useMemo)(function () {
        return Object.entries(components)
            .map(function (_a) {
            var _b, _c, _d, _e, _f;
            var name = _a[0], data = _a[1];
            var poolsAgg = data === null || data === void 0 ? void 0 : data.pools_aggregate;
            var pgUtil = (_c = (_b = poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.postgres) === null || _b === void 0 ? void 0 : _b.utilization_percent) !== null && _c !== void 0 ? _c : 0;
            var redisUtil = (_f = (_e = (_d = poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.redis) === null || _d === void 0 ? void 0 : _d.async) === null || _e === void 0 ? void 0 : _e.utilization_percent) !== null && _f !== void 0 ? _f : 0;
            var sortKey = Math.max(pgUtil, redisUtil);
            return { name: name, data: data, poolsAgg: poolsAgg, sortKey: sortKey };
        })
            .sort(function (a, b) { var _a, _b; return ((_a = b.sortKey) !== null && _a !== void 0 ? _a : -1) - ((_b = a.sortKey) !== null && _b !== void 0 ? _b : -1); });
    }, [components]);
    (0, react_1.useEffect)(function () {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
        if (!(system === null || system === void 0 ? void 0 : system.gateway_configuration))
            return;
        if (configRaw) {
            setConfigJson(JSON.stringify(configRaw, null, 2));
            return;
        }
        var compCfg = (configComponents && configComponents[selectedComponent]) || system.gateway_configuration;
        var sc = compCfg.service_capacity || {};
        var bp = compCfg.backpressure || compCfg.backpressure_settings || {};
        var payload = {
            tenant: tenant,
            project: project,
            component: selectedComponent,
            guarded_rest_patterns: compCfg.guarded_rest_patterns || [],
            bypass_throttling_patterns: compCfg.bypass_throttling_patterns || [],
            service_capacity: {
                concurrent_requests_per_process: (_b = (_a = sc.concurrent_requests_per_process) !== null && _a !== void 0 ? _a : sc.concurrent_requests_per_instance) !== null && _b !== void 0 ? _b : 5,
                processes_per_instance: (_c = sc.processes_per_instance) !== null && _c !== void 0 ? _c : 1,
                avg_processing_time_seconds: (_d = sc.avg_processing_time_seconds) !== null && _d !== void 0 ? _d : 25,
            },
            backpressure: {
                capacity_buffer: (_e = bp.capacity_buffer) !== null && _e !== void 0 ? _e : 0.2,
                queue_depth_multiplier: (_f = bp.queue_depth_multiplier) !== null && _f !== void 0 ? _f : 2.0,
                anonymous_pressure_threshold: (_g = bp.anonymous_pressure_threshold) !== null && _g !== void 0 ? _g : 0.6,
                registered_pressure_threshold: (_h = bp.registered_pressure_threshold) !== null && _h !== void 0 ? _h : 0.8,
                paid_pressure_threshold: (_j = bp.paid_pressure_threshold) !== null && _j !== void 0 ? _j : 0.8,
                hard_limit_threshold: (_k = bp.hard_limit_threshold) !== null && _k !== void 0 ? _k : 0.95,
            },
            rate_limits: compCfg.rate_limits || {},
        };
        setConfigJson(JSON.stringify(payload, null, 2));
    }, [system, tenant, project, selectedComponent, configComponents, configRaw]);
    (0, react_1.useEffect)(function () {
        plannerInitializedRef.current = false;
    }, [selectedComponent, system === null || system === void 0 ? void 0 : system.gateway_configuration]);
    (0, react_1.useEffect)(function () {
        var _a, _b, _c, _d, _e, _f, _g, _h;
        if (plannerInitializedRef.current)
            return;
        if (!system)
            return;
        var compCfg = (configComponents && configComponents[plannerComponentKey]) || system.gateway_configuration;
        var sc = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.service_capacity) || {};
        var instanceCount = (_e = (_b = (_a = components === null || components === void 0 ? void 0 : components[plannerComponentKey]) === null || _a === void 0 ? void 0 : _a.instance_count) !== null && _b !== void 0 ? _b : (_d = (_c = system.queue_stats) === null || _c === void 0 ? void 0 : _c.capacity_context) === null || _d === void 0 ? void 0 : _d.instance_count) !== null && _e !== void 0 ? _e : 1;
        setPlannerConcurrentPerProcess(String((_f = sc.concurrent_requests_per_process) !== null && _f !== void 0 ? _f : 5));
        setPlannerProcessesPerInstance(String((_g = sc.processes_per_instance) !== null && _g !== void 0 ? _g : 1));
        setPlannerAvgProcessing(String((_h = sc.avg_processing_time_seconds) !== null && _h !== void 0 ? _h : 25));
        setPlannerInstances(String(instanceCount));
        plannerInitializedRef.current = true;
    }, [system, selectedComponent, configComponents, components, plannerComponentKey]);
    var planner = (0, react_1.useMemo)(function () {
        var toNum = function (value, fallback) {
            var n = Number(value);
            return Number.isFinite(n) ? n : fallback;
        };
        var admins = toNum(plannerAdmins, 0);
        var registered = toNum(plannerRegistered, 0);
        var paid = toNum(plannerPaid, 0);
        var totalUsers = admins + registered + paid;
        var pageLoad = toNum(plannerPageLoad, 0);
        var maxTabs = Math.max(1, toNum(plannerTabs, 1));
        var windowSeconds = Math.max(1, toNum(plannerPageWindow, 10));
        var safety = Math.max(1.0, toNum(plannerSafety, 1.2));
        var concurrentPerProcess = Math.max(1, toNum(plannerConcurrentPerProcess, 1));
        var processesPerInstance = Math.max(1, toNum(plannerProcessesPerInstance, 1));
        var instances = Math.max(1, toNum(plannerInstances, 1));
        var avgSeconds = Math.max(1, toNum(plannerAvgProcessing, 25));
        var burstPerSession = pageLoad * maxTabs;
        var suggestedBurst = Math.ceil(burstPerSession * safety);
        var peakRps = windowSeconds > 0 ? (pageLoad * totalUsers) / windowSeconds : 0;
        var totalConcurrent = concurrentPerProcess * processesPerInstance * instances;
        var maxRps = avgSeconds > 0 ? totalConcurrent / avgSeconds : 0;
        var peakUtilization = maxRps > 0 ? peakRps / maxRps : 0;
        return {
            totalUsers: totalUsers,
            burstPerSession: burstPerSession,
            suggestedBurst: suggestedBurst,
            peakRps: peakRps,
            maxRps: maxRps,
            peakUtilization: peakUtilization,
            totalConcurrent: totalConcurrent,
            windowSeconds: windowSeconds,
            concurrentPerProcess: concurrentPerProcess,
            processesPerInstance: processesPerInstance,
            avgSeconds: avgSeconds,
            safety: safety,
        };
    }, [
        plannerAdmins,
        plannerRegistered,
        plannerPaid,
        plannerPageLoad,
        plannerTabs,
        plannerPageWindow,
        plannerSafety,
        plannerConcurrentPerProcess,
        plannerProcessesPerInstance,
        plannerAvgProcessing,
        plannerInstances,
    ]);
    var recommendedConfigJson = (0, react_1.useMemo)(function () {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _2, _3, _4, _5, _6;
        var compCfg = (configComponents && configComponents[selectedComponent]) || gateway;
        var roleLimits = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.rate_limits) || {};
        var recommendedBurst = Math.max(1, planner.suggestedBurst || 1);
        var windowSeconds = Math.max(1, Math.round(planner.windowSeconds || 60));
        var baseBackpressure = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.backpressure) || (compCfg === null || compCfg === void 0 ? void 0 : compCfg.backpressure_settings) || {};
        var poolsCfg = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.pools) || {};
        var limitsCfg = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.limits) || {};
        var currentServiceCapacity = (compCfg === null || compCfg === void 0 ? void 0 : compCfg.service_capacity) || {};
        var usePlannerCapacity = selectedComponent === plannerComponentKey;
        var serviceCapacityPayload = usePlannerCapacity
            ? {
                concurrent_requests_per_process: Math.max(1, Math.round(planner.concurrentPerProcess || 1)),
                processes_per_instance: Math.max(1, Math.round(planner.processesPerInstance || 1)),
                avg_processing_time_seconds: Math.max(1, Math.round(planner.avgSeconds || 25)),
            }
            : {
                concurrent_requests_per_process: (_b = (_a = currentServiceCapacity.concurrent_requests_per_process) !== null && _a !== void 0 ? _a : currentServiceCapacity.concurrent_requests_per_instance) !== null && _b !== void 0 ? _b : 5,
                processes_per_instance: (_c = currentServiceCapacity.processes_per_instance) !== null && _c !== void 0 ? _c : 1,
                avg_processing_time_seconds: (_d = currentServiceCapacity.avg_processing_time_seconds) !== null && _d !== void 0 ? _d : 25,
            };
        var suggestedPgPoolMax = selectedComponent === 'proc'
            ? Math.max(1, Math.round(planner.concurrentPerProcess || 1))
            : ((_e = poolsCfg === null || poolsCfg === void 0 ? void 0 : poolsCfg.pg_pool_max_size) !== null && _e !== void 0 ? _e : 4);
        var suggestedRedisMax = selectedComponent === 'proc'
            ? Math.max(20, Math.round((planner.concurrentPerProcess || 1) * 4))
            : ((_f = poolsCfg === null || poolsCfg === void 0 ? void 0 : poolsCfg.redis_max_connections) !== null && _f !== void 0 ? _f : 20);
        var suggested = {
            tenant: tenant,
            project: project,
            component: selectedComponent,
            service_capacity: __assign({}, serviceCapacityPayload),
            backpressure: {
                capacity_buffer: (_g = baseBackpressure.capacity_buffer) !== null && _g !== void 0 ? _g : 0.2,
                queue_depth_multiplier: (_h = baseBackpressure.queue_depth_multiplier) !== null && _h !== void 0 ? _h : 2.0,
                anonymous_pressure_threshold: (_j = baseBackpressure.anonymous_pressure_threshold) !== null && _j !== void 0 ? _j : 0.6,
                registered_pressure_threshold: (_k = baseBackpressure.registered_pressure_threshold) !== null && _k !== void 0 ? _k : 0.8,
                paid_pressure_threshold: (_l = baseBackpressure.paid_pressure_threshold) !== null && _l !== void 0 ? _l : 0.8,
                hard_limit_threshold: (_m = baseBackpressure.hard_limit_threshold) !== null && _m !== void 0 ? _m : 0.95,
            },
            rate_limits: {
                roles: {
                    anonymous: {
                        hourly: (_p = (_o = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.anonymous) === null || _o === void 0 ? void 0 : _o.hourly) !== null && _p !== void 0 ? _p : 120,
                        burst: (_r = (_q = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.anonymous) === null || _q === void 0 ? void 0 : _q.burst) !== null && _r !== void 0 ? _r : 10,
                        burst_window: (_t = (_s = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.anonymous) === null || _s === void 0 ? void 0 : _s.burst_window) !== null && _t !== void 0 ? _t : windowSeconds,
                    },
                    registered: {
                        hourly: (_v = (_u = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.registered) === null || _u === void 0 ? void 0 : _u.hourly) !== null && _v !== void 0 ? _v : 600,
                        burst: recommendedBurst,
                        burst_window: windowSeconds,
                    },
                    paid: {
                        hourly: (_x = (_w = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.paid) === null || _w === void 0 ? void 0 : _w.hourly) !== null && _x !== void 0 ? _x : 2000,
                        burst: recommendedBurst,
                        burst_window: windowSeconds,
                    },
                    privileged: {
                        hourly: (_z = (_y = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.privileged) === null || _y === void 0 ? void 0 : _y.hourly) !== null && _z !== void 0 ? _z : -1,
                        burst: Math.max(recommendedBurst, (_2 = (_0 = roleLimits === null || roleLimits === void 0 ? void 0 : roleLimits.privileged) === null || _0 === void 0 ? void 0 : _0.burst) !== null && _2 !== void 0 ? _2 : 200),
                        burst_window: windowSeconds,
                    },
                }
            },
            pools: {
                pg_pool_min_size: (_3 = poolsCfg === null || poolsCfg === void 0 ? void 0 : poolsCfg.pg_pool_min_size) !== null && _3 !== void 0 ? _3 : 0,
                pg_pool_max_size: suggestedPgPoolMax,
                redis_max_connections: suggestedRedisMax,
            },
            limits: selectedComponent === 'ingress'
                ? { max_sse_connections_per_instance: (_4 = limitsCfg === null || limitsCfg === void 0 ? void 0 : limitsCfg.max_sse_connections_per_instance) !== null && _4 !== void 0 ? _4 : 200 }
                : {
                    max_integrations_ops_concurrency: (_5 = limitsCfg === null || limitsCfg === void 0 ? void 0 : limitsCfg.max_integrations_ops_concurrency) !== null && _5 !== void 0 ? _5 : 200,
                    max_queue_size: (_6 = limitsCfg === null || limitsCfg === void 0 ? void 0 : limitsCfg.max_queue_size) !== null && _6 !== void 0 ? _6 : 0,
                },
        };
        return JSON.stringify(suggested, null, 2);
    }, [gateway, planner, tenant, project, selectedComponent, configComponents, plannerComponentKey]);
    var handleValidate = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, res, e_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    payload = JSON.parse(configJson);
                    if (!isRawGatewayConfigPayload(payload) && !payload.component) {
                        payload.component = selectedComponent;
                    }
                    return [4 /*yield*/, api.validateGatewayConfig(payload)];
                case 1:
                    res = _a.sent();
                    setValidationResult(res);
                    setActionMessage('Validation completed');
                    return [3 /*break*/, 3];
                case 2:
                    e_3 = _a.sent();
                    setActionMessage((e_3 === null || e_3 === void 0 ? void 0 : e_3.message) || 'Validation failed');
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); };
    var handleUpdate = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, e_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    payload = JSON.parse(configJson);
                    if (!isRawGatewayConfigPayload(payload) && !payload.component) {
                        payload.component = selectedComponent;
                    }
                    return [4 /*yield*/, api.updateGatewayConfig(payload)];
                case 1:
                    _a.sent();
                    setActionMessage('Config updated');
                    return [4 /*yield*/, refreshAll()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 4];
                case 3:
                    e_4 = _a.sent();
                    setActionMessage((e_4 === null || e_4 === void 0 ? void 0 : e_4.message) || 'Update failed');
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleReset = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, e_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    payload = { tenant: tenant, project: project, dry_run: dryRun };
                    return [4 /*yield*/, api.resetGatewayConfig(payload)];
                case 1:
                    _a.sent();
                    setActionMessage(dryRun ? 'Dry run completed' : 'Config reset to env');
                    return [4 /*yield*/, refreshAll()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 4];
                case 3:
                    e_5 = _a.sent();
                    setActionMessage((e_5 === null || e_5 === void 0 ? void 0 : e_5.message) || 'Reset failed');
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleClearCache = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, res, key, deleted, e_6;
        var _a, _b, _c;
        return __generator(this, function (_d) {
            switch (_d.label) {
                case 0:
                    _d.trys.push([0, 2, , 3]);
                    payload = { tenant: tenant, project: project };
                    return [4 /*yield*/, api.clearGatewayConfigCache(payload)];
                case 1:
                    res = _d.sent();
                    key = (_a = res === null || res === void 0 ? void 0 : res.result) === null || _a === void 0 ? void 0 : _a.key;
                    deleted = (_c = (_b = res === null || res === void 0 ? void 0 : res.result) === null || _b === void 0 ? void 0 : _b.deleted) !== null && _c !== void 0 ? _c : 0;
                    setClearCacheMessage("Cleared cache key ".concat(key || '(unknown)', " (deleted=").concat(deleted, "). Restart to re-apply env/GATEWAY_CONFIG_JSON."));
                    return [3 /*break*/, 3];
                case 2:
                    e_6 = _d.sent();
                    setClearCacheMessage((e_6 === null || e_6 === void 0 ? void 0 : e_6.message) || 'Clear cache failed');
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); };
    var resetCircuit = function (name) { return __awaiter(void 0, void 0, void 0, function () {
        var e_7;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    return [4 /*yield*/, api.resetCircuitBreaker(name)];
                case 1:
                    _a.sent();
                    return [4 /*yield*/, refreshAll()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 4];
                case 3:
                    e_7 = _a.sent();
                    setActionMessage((e_7 === null || e_7 === void 0 ? void 0 : e_7.message) || 'Failed to reset circuit breaker');
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleResetThrottling = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, res, e_8;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!resetRateLimits && !resetBackpressure && !resetThrottlingStats && !purgeChatQueues) {
                        setResetThrottlingMessage('Select at least one reset option');
                        return [2 /*return*/];
                    }
                    setResettingThrottling(true);
                    setResetThrottlingMessage(null);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    payload = {
                        tenant: tenant,
                        project: project,
                        reset_rate_limits: resetRateLimits,
                        reset_backpressure: resetBackpressure,
                        reset_throttling_stats: resetThrottlingStats,
                        purge_chat_queues: purgeChatQueues,
                        all_sessions: resetAllSessions,
                    };
                    if (resetSessionId.trim()) {
                        payload.session_id = resetSessionId.trim();
                    }
                    return [4 /*yield*/, api.resetThrottling(payload)];
                case 2:
                    res = _a.sent();
                    setResetThrottlingMessage((res === null || res === void 0 ? void 0 : res.message) || 'Throttling reset');
                    return [4 /*yield*/, refreshAll()];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 6];
                case 4:
                    e_8 = _a.sent();
                    setResetThrottlingMessage((e_8 === null || e_8 === void 0 ? void 0 : e_8.message) || 'Failed to reset throttling');
                    return [3 /*break*/, 6];
                case 5:
                    setResettingThrottling(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var closeBurstStreams = (0, react_1.useCallback)(function () {
        var sessions = burstSessionsRef.current || [];
        sessions.forEach(function (s) {
            try {
                s.es.close();
            }
            catch (_) { /* noop */ }
        });
        burstSessionsRef.current = [];
        setBurstOpenCount(0);
    }, []);
    var openBurstStreams = (0, react_1.useCallback)(function () { return __awaiter(void 0, void 0, void 0, function () {
        var adminCount, regCount, admins, regs, selected, baseUrl, sessions;
        return __generator(this, function (_a) {
            if (!(burstUsers === null || burstUsers === void 0 ? void 0 : burstUsers.users)) {
                setBurstStatus('Burst users not loaded');
                return [2 /*return*/];
            }
            closeBurstStreams();
            adminCount = Math.max(0, parseInt(burstAdminCount, 10) || 0);
            regCount = Math.max(0, parseInt(burstRegisteredCount, 10) || 0);
            admins = burstUsers.users.admin.slice(0, adminCount);
            regs = burstUsers.users.registered.slice(0, regCount);
            selected = __spreadArray(__spreadArray([], admins.map(function (u) { return ({ user: u, role: 'admin' }); }), true), regs.map(function (u) { return ({ user: u, role: 'registered' }); }), true);
            if (!selected.length) {
                setBurstStatus('No users selected for SSE streams');
                return [2 /*return*/];
            }
            baseUrl = settings.getBaseUrl().replace(/\/$/, '');
            sessions = [];
            selected.forEach(function (entry, idx) {
                var streamId = "burst-".concat(entry.role, "-").concat(idx, "-").concat(Date.now(), "-").concat(Math.random().toString(16).slice(2, 8));
                var url = new URL("".concat(baseUrl, "/sse/stream"));
                url.searchParams.set('stream_id', streamId);
                url.searchParams.set('bearer_token', entry.user.token);
                if (tenant)
                    url.searchParams.set('tenant', tenant);
                if (project)
                    url.searchParams.set('project', project);
                var es = new EventSource(url.toString());
                es.addEventListener('error', function () {
                    // keep simple: errors are visible in devtools
                });
                sessions.push({ token: entry.user.token, streamId: streamId, role: entry.role, es: es });
            });
            burstSessionsRef.current = sessions;
            setBurstOpenCount(sessions.length);
            setBurstStatus("Opened ".concat(sessions.length, " SSE streams"));
            return [2 /*return*/];
        });
    }); }, [burstUsers, burstAdminCount, burstRegisteredCount, closeBurstStreams, tenant, project]);
    var runWithConcurrency = function (tasks, limit) { return __awaiter(void 0, void 0, void 0, function () {
        var idx, safeLimit, workers;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    idx = 0;
                    safeLimit = Math.max(1, Math.min(limit, tasks.length || 1));
                    workers = new Array(safeLimit).fill(null).map(function () { return __awaiter(void 0, void 0, void 0, function () {
                        var current;
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0:
                                    if (!(idx < tasks.length)) return [3 /*break*/, 2];
                                    current = idx++;
                                    return [4 /*yield*/, tasks[current]()];
                                case 1:
                                    _a.sent();
                                    return [3 /*break*/, 0];
                                case 2: return [2 /*return*/];
                            }
                        });
                    }); });
                    return [4 /*yield*/, Promise.all(workers)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); };
    var sendBurstMessages = (0, react_1.useCallback)(function () { return __awaiter(void 0, void 0, void 0, function () {
        var sessions, perUser, concurrency, baseUrl, payloadBase, tasks, startedAt, elapsed, e_9;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    sessions = burstSessionsRef.current || [];
                    if (!sessions.length) {
                        setBurstStatus('No active SSE streams. Open streams first.');
                        return [2 /*return*/];
                    }
                    perUser = Math.max(1, parseInt(burstMessagesPerUser, 10) || 1);
                    concurrency = Math.max(1, parseInt(burstConcurrency, 10) || 10);
                    baseUrl = settings.getBaseUrl().replace(/\/$/, '');
                    payloadBase = {
                        external_events: [
                            {
                                type: 'event.user.prompt',
                                event_source_id: 'event.user.prompt',
                                reactive: true,
                                payload: {
                                    mime: 'text/plain',
                                    event: { text: burstMessage || 'ping' },
                                },
                            },
                        ],
                    };
                    if (burstBundleId)
                        payloadBase.bundle_id = burstBundleId;
                    tasks = [];
                    sessions.forEach(function (s) {
                        var _loop_1 = function (i) {
                            tasks.push(function () { return __awaiter(void 0, void 0, void 0, function () {
                                var convId, payload, res;
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0:
                                            convId = "burst-".concat(s.streamId, "-").concat(i);
                                            payload = __assign(__assign({}, payloadBase), { conversation_id: convId });
                                            return [4 /*yield*/, fetch("".concat(baseUrl, "/sse/chat?stream_id=").concat(encodeURIComponent(s.streamId)), {
                                                    method: 'POST',
                                                    headers: new Headers({
                                                        'Content-Type': 'application/json',
                                                        'Authorization': "Bearer ".concat(s.token),
                                                    }),
                                                    body: JSON.stringify(payload),
                                                })];
                                        case 1:
                                            res = _a.sent();
                                            if (!res.ok) {
                                                throw new Error("chat ".concat(res.status));
                                            }
                                            return [2 /*return*/];
                                    }
                                });
                            }); });
                        };
                        for (var i = 0; i < perUser; i++) {
                            _loop_1(i);
                        }
                    });
                    startedAt = performance.now();
                    setBurstRunning(true);
                    setBurstStatus("Sending ".concat(tasks.length, " messages\u2026"));
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, runWithConcurrency(tasks, concurrency)];
                case 2:
                    _a.sent();
                    elapsed = Math.round(performance.now() - startedAt);
                    setBurstStatus("Burst complete: ".concat(tasks.length, " messages in ").concat(elapsed, "ms"));
                    return [3 /*break*/, 5];
                case 3:
                    e_9 = _a.sent();
                    setBurstStatus("Burst error: ".concat((e_9 === null || e_9 === void 0 ? void 0 : e_9.message) || 'unknown error'));
                    return [3 /*break*/, 5];
                case 4:
                    setBurstRunning(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); }, [burstMessagesPerUser, burstConcurrency, burstMessage, burstBundleId]);
    return (<div className="min-h-screen bg-gray-50 text-gray-900">
            <div className="max-w-6xl mx-auto px-4 py-4 space-y-4">
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <h1 className="text-lg font-semibold">Gateway Monitoring</h1>
                        <p className="text-xs text-gray-600">System health, queues, throttling, and config management.</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <label className="text-[11px] text-gray-600 flex items-center gap-2">
                            <input type="checkbox" checked={autoRefresh} onChange={function (e) { return setAutoRefresh(e.target.checked); }}/>
                            Auto refresh
                        </label>
                        <Button variant="secondary" onClick={refreshAll} disabled={loading}>
                            {loading ? 'Refreshing…' : 'Refresh'}
                        </Button>
                    </div>
                </div>

                {error && (<Card>
                        <CardBody>
                            <div className="text-xs text-rose-700">{error}</div>
                        </CardBody>
                    </Card>)}

                <Card>
                    <CardHeader title="Tenant Summary" subtitle={"Last update: ".concat(lastUpdate || '—')} action={gateway ? <Pill tone="success">{gateway.current_profile}</Pill> : null}/>
                    <CardBody>
                        <Legend>
                            Proc queue = backpressure queue depth; SSE = active ingress streams; Instances = heartbeat counts; throttled (1h) = 429/503 totals.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Tenant / Project</div>
                                <div className="text-sm font-semibold">{(configRaw === null || configRaw === void 0 ? void 0 : configRaw.tenant) || (gateway === null || gateway === void 0 ? void 0 : gateway.tenant_id) || '—'}</div>
                                <div className="text-xs text-gray-500">{(configRaw === null || configRaw === void 0 ? void 0 : configRaw.project) || (gateway === null || gateway === void 0 ? void 0 : gateway.display_name) || '—'}</div>
                                <div className="text-[11px] text-gray-500">Config source: {configSource}</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Proc Queue</div>
                                <div className="text-sm font-semibold">{(_j = queue === null || queue === void 0 ? void 0 : queue.total) !== null && _j !== void 0 ? _j : 0}</div>
                                <div className="text-xs text-gray-500">{Math.round((capacityCtx.pressure_ratio || 0) * 100)}% pressure</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Ingress SSE</div>
                                <div className="text-sm font-semibold">
                                    {(_l = (_k = sseStats === null || sseStats === void 0 ? void 0 : sseStats.global_total_connections) !== null && _k !== void 0 ? _k : sseStats === null || sseStats === void 0 ? void 0 : sseStats.total_connections) !== null && _l !== void 0 ? _l : 0}
                                    {typeof ((_m = sseStats === null || sseStats === void 0 ? void 0 : sseStats.global_max_connections) !== null && _m !== void 0 ? _m : sseStats === null || sseStats === void 0 ? void 0 : sseStats.max_connections) === 'number'
            && ((_o = sseStats === null || sseStats === void 0 ? void 0 : sseStats.global_max_connections) !== null && _o !== void 0 ? _o : sseStats === null || sseStats === void 0 ? void 0 : sseStats.max_connections) > 0
            ? " / ".concat(((_p = sseStats === null || sseStats === void 0 ? void 0 : sseStats.global_max_connections) !== null && _p !== void 0 ? _p : sseStats === null || sseStats === void 0 ? void 0 : sseStats.max_connections))
            : ''}
                                </div>
                                <div className="text-xs text-gray-500">sessions {(_r = (_q = sseStats === null || sseStats === void 0 ? void 0 : sseStats.global_sessions) !== null && _q !== void 0 ? _q : sseStats === null || sseStats === void 0 ? void 0 : sseStats.sessions) !== null && _r !== void 0 ? _r : 0}</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Instances</div>
                                <div className="text-sm font-semibold">
                                    ingress {(_t = (_s = components === null || components === void 0 ? void 0 : components.ingress) === null || _s === void 0 ? void 0 : _s.instance_count) !== null && _t !== void 0 ? _t : 0} · proc {(_v = (_u = components === null || components === void 0 ? void 0 : components.proc) === null || _u === void 0 ? void 0 : _u.instance_count) !== null && _v !== void 0 ? _v : 0}
                                </div>
                                <div className="text-xs text-gray-500">
                                    throttled (1h) {(_w = throttling === null || throttling === void 0 ? void 0 : throttling.total_throttled) !== null && _w !== void 0 ? _w : 0} · {((_x = throttling === null || throttling === void 0 ? void 0 : throttling.throttle_rate) !== null && _x !== void 0 ? _x : 0).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Components & Autoscaler" subtitle="Ingress/proc health, capacity, and scaling signals."/>
                    <CardBody>
                        <Legend>
                            Utilization = current / max; decision is autoscaler suggestion; windows are rolling 1m/15m/1h.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {["ingress", "proc"].map(function (comp) {
            var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _2, _3, _4, _5, _6, _7;
            var data = components === null || components === void 0 ? void 0 : components[comp];
            var auto = autoscaler === null || autoscaler === void 0 ? void 0 : autoscaler[comp];
            var decision = (auto === null || auto === void 0 ? void 0 : auto.decision) || 'hold';
            var tone = decision === 'scale_up' ? 'danger' : decision === 'scale_down' ? 'warning' : 'success';
            return (<div key={comp} className="p-4 rounded-xl bg-gray-100">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="text-sm font-semibold">{comp}</div>
                                            <Pill tone={tone}>{decision}</Pill>
                                        </div>
                                        {data ? (<div className="space-y-1 text-xs text-gray-600">
                                                <div>Instances: {(_a = data.instance_count) !== null && _a !== void 0 ? _a : 0}</div>
                                                <div>
                                                    Processes: {(_b = data.healthy_processes) !== null && _b !== void 0 ? _b : 0}/{(_c = data.actual_processes) !== null && _c !== void 0 ? _c : 0}
                                                    {typeof data.expected_processes === 'number' ? " (expected ".concat(data.expected_processes, ")") : ''}
                                                </div>
                                                <div>Utilization: {(_d = data.utilization_percent) !== null && _d !== void 0 ? _d : 0}%</div>
                                                {comp === 'ingress' && data.sse && (<div>
                                                        SSE: {(_e = data.sse.total_connections) !== null && _e !== void 0 ? _e : 0}
                                                        {data.sse.max_connections ? " / ".concat(data.sse.max_connections) : ''}
                                                        {data.sse.utilization_percent ? " (".concat(data.sse.utilization_percent, "%)") : ''}
                                                        {data.sse.windows && (<div className="text-[11px] text-gray-500">
                                                                windows: 1m {(_f = data.sse.windows["1m"]) !== null && _f !== void 0 ? _f : '—'} · 15m {(_g = data.sse.windows["15m"]) !== null && _g !== void 0 ? _g : '—'} · 1h {(_h = data.sse.windows["1h"]) !== null && _h !== void 0 ? _h : '—'} · max {(_j = data.sse.windows["max"]) !== null && _j !== void 0 ? _j : '—'}
                                                            </div>)}
                                                    </div>)}
                                                {comp === 'ingress' && (<LatencyTable title="Ingress REST latency (ms)" data={(_k = data.latency) === null || _k === void 0 ? void 0 : _k.rest_ms} compact className="mt-2"/>)}
                                                {comp === 'proc' && data.queue && (<div>
                                                        Queue: {(_l = data.queue.total) !== null && _l !== void 0 ? _l : 0} · pressure {((_m = data.queue.pressure_ratio) !== null && _m !== void 0 ? _m : 0).toFixed(2)}
                                                        {data.queue.windows && (<div className="text-[11px] text-gray-500">
                                                                depth windows: 1m {(_p = (_o = data.queue.windows.depth) === null || _o === void 0 ? void 0 : _o["1m"]) !== null && _p !== void 0 ? _p : '—'} · 15m {(_r = (_q = data.queue.windows.depth) === null || _q === void 0 ? void 0 : _q["15m"]) !== null && _r !== void 0 ? _r : '—'} · 1h {(_t = (_s = data.queue.windows.depth) === null || _s === void 0 ? void 0 : _s["1h"]) !== null && _t !== void 0 ? _t : '—'} · max {(_v = (_u = data.queue.windows.depth) === null || _u === void 0 ? void 0 : _u["max"]) !== null && _v !== void 0 ? _v : '—'}
                                                                <br />
                                                                pressure windows: 1m {(_x = (_w = data.queue.windows.pressure_ratio) === null || _w === void 0 ? void 0 : _w["1m"]) !== null && _x !== void 0 ? _x : '—'} · 15m {(_z = (_y = data.queue.windows.pressure_ratio) === null || _y === void 0 ? void 0 : _y["15m"]) !== null && _z !== void 0 ? _z : '—'} · 1h {(_2 = (_0 = data.queue.windows.pressure_ratio) === null || _0 === void 0 ? void 0 : _0["1h"]) !== null && _2 !== void 0 ? _2 : '—'} · max {(_4 = (_3 = data.queue.windows.pressure_ratio) === null || _3 === void 0 ? void 0 : _3["max"]) !== null && _4 !== void 0 ? _4 : '—'}
                                                            </div>)}
                                                        {data.latency && (<div className="text-[11px] text-gray-500 mt-1">
                                                                Latency: see Latency card.
                                                            </div>)}
                                                    </div>)}
                                                {data.pools && (<div className="text-[11px] text-gray-500">
                                                        Pools: pg_max={(_5 = data.pools.pg_pool_max_size) !== null && _5 !== void 0 ? _5 : '—'} · redis_max={(_6 = data.pools.redis_max_connections) !== null && _6 !== void 0 ? _6 : '—'}
                                                        {data.pools.estimated_pg_total ? " \u00B7 est_pg_total=".concat(data.pools.estimated_pg_total) : ''}
                                                    </div>)}
                                                {((_7 = auto === null || auto === void 0 ? void 0 : auto.reasons) === null || _7 === void 0 ? void 0 : _7.length) ? (<div className="text-[11px] text-gray-500">Reasons: {auto.reasons.join('; ')}</div>) : (<div className="text-[11px] text-gray-500">Reasons: none</div>)}
                                                {Array.isArray(data.instances) && data.instances.length > 0 && (<div className="text-[11px] text-gray-500 mt-1">
                                                        <div className="mb-1">Instances:</div>
                                                        <div className="flex flex-wrap gap-2">
                                                            {data.instances.map(function (i) {
                            var _a, _b;
                            var unhealthy = ((_a = i.healthy_processes) !== null && _a !== void 0 ? _a : 0) < ((_b = i.processes) !== null && _b !== void 0 ? _b : 0);
                            return (<span key={i.instance_id} className="flex items-center gap-1">
                                                                        <span>{i.instance_id}</span>
                                                                        {i.draining && <Pill tone="warning">draining</Pill>}
                                                                        {!i.draining && unhealthy && <Pill tone="danger">unhealthy</Pill>}
                                                                    </span>);
                        })}
                                                        </div>
                                                    </div>)}
                                            </div>) : (<div className="text-xs text-gray-500">No heartbeat data.</div>)}
                                    </div>);
        })}
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Latency (Rolling Windows)" subtitle="P50/P95/P99 in ms over 1m, 15m, 1h windows."/>
                    <CardBody>
                        <Legend>
                            Windows are rolling; max = 1h high-water mark.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <LatencyTable title="Ingress REST" data={(_z = (_y = components === null || components === void 0 ? void 0 : components.ingress) === null || _y === void 0 ? void 0 : _y.latency) === null || _z === void 0 ? void 0 : _z.rest_ms}/>
                            <LatencyTable title="Proc Queue Wait" data={(_2 = (_0 = components === null || components === void 0 ? void 0 : components.proc) === null || _0 === void 0 ? void 0 : _0.latency) === null || _2 === void 0 ? void 0 : _2.queue_wait_ms}/>
                            <LatencyTable title="Proc Execution" data={(_4 = (_3 = components === null || components === void 0 ? void 0 : components.proc) === null || _3 === void 0 ? void 0 : _3.latency) === null || _4 === void 0 ? void 0 : _4.exec_ms}/>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Pools (Aggregated)" subtitle="Totals across all workers, sorted by utilization."/>
                    <CardBody>
                        <Legend>
                            Reported = number of workers reporting; max in-use = 1h high-water mark; totals are aggregated across the component.
                        </Legend>
                        {poolAggregateEntries.length ? (<div className="space-y-3">
                                {poolAggregateEntries.map(function (_a) {
                var _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m;
                var name = _a.name, poolsAgg = _a.poolsAgg;
                var pg = (poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.postgres) || {};
                var rAsync = ((_b = poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.redis) === null || _b === void 0 ? void 0 : _b.async) || {};
                var rAsyncDecode = ((_c = poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.redis) === null || _c === void 0 ? void 0 : _c.async_decode) || {};
                var rSync = ((_d = poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.redis) === null || _d === void 0 ? void 0 : _d.sync) || {};
                var fmt = function (val) { return (val === null || val === undefined ? '—' : val); };
                var fmtMaybeZero = function (val, fallbackZero) {
                    if (val === null || val === undefined) {
                        return fallbackZero ? 0 : '—';
                    }
                    return val;
                };
                var pgReported = (_e = pg.reported_processes) !== null && _e !== void 0 ? _e : 0;
                var raReported = (_f = rAsync.reported_processes) !== null && _f !== void 0 ? _f : 0;
                var radReported = (_g = rAsyncDecode.reported_processes) !== null && _g !== void 0 ? _g : 0;
                var rsReported = (_h = rSync.reported_processes) !== null && _h !== void 0 ? _h : 0;
                var windows = (poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.utilization_windows) || {};
                var inUseWindows = (poolsAgg === null || poolsAgg === void 0 ? void 0 : poolsAgg.in_use_windows) || {};
                var fmtWindow = function (w) {
                    var _a, _b, _c, _d;
                    if (!w)
                        return '—';
                    var w1m = (_a = w["1m"]) !== null && _a !== void 0 ? _a : '—';
                    var w15 = (_b = w["15m"]) !== null && _b !== void 0 ? _b : '—';
                    var w1h = (_c = w["1h"]) !== null && _c !== void 0 ? _c : '—';
                    var wMax = (_d = w["max"]) !== null && _d !== void 0 ? _d : '—';
                    return "1m ".concat(w1m, "% \u00B7 15m ").concat(w15, "% \u00B7 1h ").concat(w1h, "% \u00B7 max ").concat(wMax, "%");
                };
                var fmtInUseMax = function (w) {
                    var _a;
                    if (!w)
                        return '—';
                    return (_a = w["max"]) !== null && _a !== void 0 ? _a : '—';
                };
                return (<div key={name} className="p-4 rounded-xl bg-gray-100">
                                            <div className="text-sm font-semibold mb-2">{name}</div>
                                            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs text-gray-600">
                                                <div>
                                                    <div className="text-[11px] text-gray-500">PG</div>
                                                    <div className="text-sm font-semibold">
                                                        {pgReported ? "".concat(fmtMaybeZero(pg.in_use_total, true), "/").concat(fmt((_j = pg.max_total) !== null && _j !== void 0 ? _j : pg.size_total)) : '—'}
                                                        {pgReported && pg.utilization_percent != null ? " (".concat(pg.utilization_percent, "%)") : ''}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        reported {pgReported}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        {fmtWindow(windows.postgres)}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        max in-use (1h): {fmtInUseMax(inUseWindows.postgres)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-[11px] text-gray-500">Redis (async)</div>
                                                    <div className="text-sm font-semibold">
                                                        {raReported ? "".concat(fmt(rAsync.in_use_total), "/").concat(fmt((_k = rAsync.max_total) !== null && _k !== void 0 ? _k : rAsync.total_total)) : '—'}
                                                        {raReported && rAsync.utilization_percent != null ? " (".concat(rAsync.utilization_percent, "%)") : ''}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        reported {raReported}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        {fmtWindow(windows.redis_async)}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        max in-use (1h): {fmtInUseMax(inUseWindows.redis_async)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-[11px] text-gray-500">Redis (async decode)</div>
                                                    <div className="text-sm font-semibold">
                                                        {radReported ? "".concat(fmt(rAsyncDecode.in_use_total), "/").concat(fmt((_l = rAsyncDecode.max_total) !== null && _l !== void 0 ? _l : rAsyncDecode.total_total)) : '—'}
                                                        {radReported && rAsyncDecode.utilization_percent != null ? " (".concat(rAsyncDecode.utilization_percent, "%)") : ''}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        reported {radReported}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        {fmtWindow(windows.redis_async_decode)}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        max in-use (1h): {fmtInUseMax(inUseWindows.redis_async_decode)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-[11px] text-gray-500">Redis (sync)</div>
                                                    <div className="text-sm font-semibold">
                                                        {rsReported ? "".concat(fmt(rSync.in_use_total), "/").concat(fmt((_m = rSync.max_total) !== null && _m !== void 0 ? _m : rSync.total_total)) : '—'}
                                                        {rsReported && rSync.utilization_percent != null ? " (".concat(rSync.utilization_percent, "%)") : ''}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        reported {rsReported}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        {fmtWindow(windows.redis_sync)}
                                                    </div>
                                                    <div className="text-[11px] text-gray-500">
                                                        max in-use (1h): {fmtInUseMax(inUseWindows.redis_sync)}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>);
            })}
                            </div>) : (<div className="text-xs text-gray-500">No pool data reported yet.</div>)}
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Traffic (Requests)" subtitle="Totals and average per minute by period."/>
                    <CardBody>
                        <Legend>
                            Periods are rolling windows; values show totals and averages per minute.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {["1h", "3h", "24h"].map(function (key) {
            var _a;
            var period = throttlingByPeriod[key] || {};
            var total = (_a = period.total_requests) !== null && _a !== void 0 ? _a : 0;
            var hours = parseInt(key.replace("h", ""), 10) || 1;
            var perHour = hours ? total / hours : 0;
            var perMin = perHour / 60;
            return (<div key={key} className="p-4 rounded-xl bg-gray-100">
                                        <div className="text-xs text-gray-600">{key} total</div>
                                        <div className="text-sm font-semibold">{Math.round(total)}</div>
                                        <div className="text-xs text-gray-500">
                                            ~{Math.round(perMin)} / min · ~{Math.round(perHour)} / hour
                                        </div>
                                    </div>);
        })}
                        </div>
                        {Object.keys(throttlingWindows).length > 0 && (<div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                                {["1m", "15m", "1h"].map(function (key) {
                var _a, _b, _c;
                var win = throttlingWindows[key] || {};
                return (<div key={key} className="p-4 rounded-xl bg-gray-100">
                                            <div className="text-xs text-gray-600">{key} throttling</div>
                                            <div className="text-sm font-semibold">{(_a = win.total_throttled) !== null && _a !== void 0 ? _a : 0}</div>
                                            <div className="text-xs text-gray-500">
                                                429 {(_b = win.rate_limit_429) !== null && _b !== void 0 ? _b : 0} · 503 {(_c = win.backpressure_503) !== null && _c !== void 0 ? _c : 0}
                                            </div>
                                            <div className="text-xs text-gray-500">
                                                {win.events_per_min != null ? "".concat(win.events_per_min, " / min") : '—'}
                                            </div>
                                        </div>);
            })}
                            </div>)}
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Queues" subtitle="Current queue sizes and admission state."/>
                    <CardBody>
                        <Legend>
                            Queue sizes are current backpressure queues; “accepting/blocked” is per-role admission status.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Anonymous</div>
                                <div className="text-sm font-semibold">{(_5 = queue === null || queue === void 0 ? void 0 : queue.anonymous) !== null && _5 !== void 0 ? _5 : 0}</div>
                                <div className="text-xs text-gray-500">
                                    {capacityCtx.accepting_anonymous ? 'accepting' : 'blocked'}
                                </div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Registered</div>
                                <div className="text-sm font-semibold">{(_6 = queue === null || queue === void 0 ? void 0 : queue.registered) !== null && _6 !== void 0 ? _6 : 0}</div>
                                <div className="text-xs text-gray-500">
                                    {capacityCtx.accepting_registered ? 'accepting' : 'blocked'}
                                </div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Paid</div>
                                <div className="text-sm font-semibold">{(_7 = queue === null || queue === void 0 ? void 0 : queue.paid) !== null && _7 !== void 0 ? _7 : 0}</div>
                                <div className="text-xs text-gray-500">
                                    {((_8 = capacityCtx.accepting_paid) !== null && _8 !== void 0 ? _8 : true) ? 'accepting' : 'blocked'}
                                </div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Privileged</div>
                                <div className="text-sm font-semibold">{(_9 = queue === null || queue === void 0 ? void 0 : queue.privileged) !== null && _9 !== void 0 ? _9 : 0}</div>
                                <div className="text-xs text-gray-500">
                                    {capacityCtx.accepting_privileged ? 'accepting' : 'blocked'}
                                </div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Hard Limit</div>
                                <div className="text-sm font-semibold">{(_11 = (_10 = capacityCtx.thresholds) === null || _10 === void 0 ? void 0 : _10.hard_limit_threshold) !== null && _11 !== void 0 ? _11 : 0}</div>
                                <div className="text-xs text-gray-500">items</div>
                            </div>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Queue Analytics" subtitle="Average wait time and throughput (last hour)."/>
                    <CardBody>
                        <Legend>
                            Analytics are rolling (last hour) across proc workers.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            {["anonymous", "registered", "paid", "privileged"].map(function (key) {
            var _a, _b, _c, _d;
            var q = ((_a = queueAnalytics === null || queueAnalytics === void 0 ? void 0 : queueAnalytics.individual_queues) === null || _a === void 0 ? void 0 : _a[key]) || {};
            var wait = (_b = q.avg_wait) !== null && _b !== void 0 ? _b : 0;
            var throughput = (_c = q.throughput) !== null && _c !== void 0 ? _c : 0;
            return (<div key={key} className="p-4 rounded-xl bg-gray-100">
                                        <div className="text-xs text-gray-600">{key}</div>
                                        <div className="text-sm font-semibold">{(_d = q.size) !== null && _d !== void 0 ? _d : 0} queued</div>
                                        <div className="text-xs text-gray-500">avg wait {wait.toFixed(2)}s</div>
                                        <div className="text-xs text-gray-500">throughput {throughput}/hr</div>
                                        <div className="text-xs text-gray-500">{q.blocked ? 'blocked' : 'accepting'}</div>
                                    </div>);
        })}
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Utilization</div>
                                <div className="text-sm font-semibold">
                                    {typeof queueUtilization === 'number' ? "".concat(queueUtilization.toFixed(1), "%") : '—'}
                                </div>
                                <div className="text-xs text-gray-500">queue / weighted capacity</div>
                            </div>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Burst Simulator" subtitle="Dev-only load generator using SimpleIDP tokens."/>
                    <CardBody className="space-y-4">
                        <Legend>
                            Uses SimpleIDP tokens to open SSE streams and send synthetic chat bursts.
                        </Legend>
                        {burstError && (<div className="text-xs text-rose-700">{burstError}</div>)}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <Input label="Admin streams" value={burstAdminCount} onChange={function (e) { return setBurstAdminCount(e.target.value); }}/>
                            <Input label="Registered streams" value={burstRegisteredCount} onChange={function (e) { return setBurstRegisteredCount(e.target.value); }}/>
                            <Input label="Messages / user" value={burstMessagesPerUser} onChange={function (e) { return setBurstMessagesPerUser(e.target.value); }}/>
                            <Input label="Concurrency" value={burstConcurrency} onChange={function (e) { return setBurstConcurrency(e.target.value); }}/>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <Input label="Message text" value={burstMessage} onChange={function (e) { return setBurstMessage(e.target.value); }}/>
                            <Input label="Bundle ID (optional)" value={burstBundleId} onChange={function (e) { return setBurstBundleId(e.target.value); }}/>
                        </div>
                        <div className="flex flex-wrap items-center gap-3">
                            <Button variant="secondary" onClick={loadBurstUsers}>Load tokens</Button>
                            <Button variant="secondary" onClick={openBurstStreams}>Open SSE</Button>
                            <Button variant="secondary" onClick={closeBurstStreams}>Close SSE</Button>
                            <Button onClick={sendBurstMessages} disabled={burstRunning}>Send chat burst</Button>
                            <span className="text-xs text-gray-600">
                                Open streams: {burstOpenCount}
                            </span>
                        </div>
                        {burstUsers ? (<div className="text-xs text-gray-500">
                                Available tokens: admin {(_13 = (_12 = burstUsers.counts) === null || _12 === void 0 ? void 0 : _12.admin) !== null && _13 !== void 0 ? _13 : 0}, registered {(_15 = (_14 = burstUsers.counts) === null || _14 === void 0 ? void 0 : _14.registered) !== null && _15 !== void 0 ? _15 : 0}, paid {(_17 = (_16 = burstUsers.counts) === null || _16 === void 0 ? void 0 : _16.paid) !== null && _17 !== void 0 ? _17 : 0}
                            </div>) : (<div className="text-xs text-gray-500">
                                Enable with `MONITORING_BURST_ENABLE=1` and `AUTH_PROVIDER=simple`.
                            </div>)}
                        {burstStatus && (<div className="text-xs text-gray-600">{burstStatus}</div>)}
                    </CardBody>
                </Card>

                <CapacityPanel capacity={system === null || system === void 0 ? void 0 : system.capacity_transparency} dbConnections={system === null || system === void 0 ? void 0 : system.db_connections} capacitySource={capacitySource} capacitySourceActual={capacitySourceActual} capacitySourceHealthy={capacitySourceHealthy}/>

                <Card>
                    <CardHeader title="Capacity Planner (Rough)" subtitle={"Estimate burst limits and compare expected peak traffic to capacity. Uses service_capacity for capacity source: ".concat(plannerComponentKey, ".")}/>
                    <CardBody className="space-y-4">
                        <Legend>
                            Rough sizing only; validate with real traffic and latency.
                        </Legend>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-600">
                            <label className="flex items-center gap-2">
                                Draft target
                                <select className="border border-gray-200 rounded px-2 py-1 text-xs" value={selectedComponent} onChange={function (e) { return setSelectedComponent(e.target.value); }}>
                                    <option value="ingress">ingress</option>
                                    <option value="proc">proc</option>
                                </select>
                            </label>
                            <span>Config source: {configSource}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                            {"Source: GATEWAY_CONFIG_JSON.service_capacity.".concat(plannerComponentKey, " (or admin update). Assumes all instances in the selected tenant/project share the same config.")}
                        </div>
                        {selectedComponent !== plannerComponentKey && (<div className="text-xs text-amber-700">
                                Planner is anchored to capacity source <span className="font-semibold">{plannerComponentKey}</span>. Updating
                                <span className="font-semibold"> {selectedComponent}</span> will keep its current service_capacity and only
                                apply rate limits/limits for that component.
                            </div>)}
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                            <Input label="Admins" value={plannerAdmins} onChange={function (e) { return setPlannerAdmins(e.target.value); }}/>
                            <Input label="Registered" value={plannerRegistered} onChange={function (e) { return setPlannerRegistered(e.target.value); }}/>
                            <Input label="Paid" value={plannerPaid} onChange={function (e) { return setPlannerPaid(e.target.value); }}/>
                            <Input label="Page-load requests" value={plannerPageLoad} onChange={function (e) { return setPlannerPageLoad(e.target.value); }}/>
                            <Input label="Max tabs / session" value={plannerTabs} onChange={function (e) { return setPlannerTabs(e.target.value); }}/>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                            <Input label="Page-load window (s)" value={plannerPageWindow} onChange={function (e) { return setPlannerPageWindow(e.target.value); }}/>
                            <Input label="Safety factor" value={plannerSafety} onChange={function (e) { return setPlannerSafety(e.target.value); }}/>
                            <Input label={"Concurrent / processor (".concat(plannerComponentKey, ".service_capacity.concurrent_requests_per_process)")} value={plannerConcurrentPerProcess} onChange={function (e) { return setPlannerConcurrentPerProcess(e.target.value); }}/>
                            <Input label={"Workers / instance (".concat(plannerComponentKey, ".service_capacity.processes_per_instance)")} value={plannerProcessesPerInstance} onChange={function (e) { return setPlannerProcessesPerInstance(e.target.value); }}/>
                            <Input label="Instances" value={plannerInstances} onChange={function (e) { return setPlannerInstances(e.target.value); }}/>
                            <Input label="Avg processing (s)" value={plannerAvgProcessing} onChange={function (e) { return setPlannerAvgProcessing(e.target.value); }}/>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Total users</div>
                                <div className="text-sm font-semibold">{planner.totalUsers}</div>
                                <div className="text-xs text-gray-500">admins + registered + paid</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Burst / session (min)</div>
                                <div className="text-sm font-semibold">{planner.burstPerSession}</div>
                                <div className="text-xs text-gray-500">page-load × tabs</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Suggested burst</div>
                                <div className="text-sm font-semibold">{planner.suggestedBurst}</div>
                                <div className="text-xs text-gray-500">with safety factor</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Peak RPS</div>
                                <div className="text-sm font-semibold">{planner.peakRps.toFixed(1)}</div>
                                <div className="text-xs text-gray-500">page-load surge</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Max RPS</div>
                                <div className="text-sm font-semibold">{planner.maxRps.toFixed(1)}</div>
                                <div className="text-xs text-gray-500">capacity estimate</div>
                            </div>
                            <div className="p-4 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Peak utilization</div>
                                <div className="text-sm font-semibold">
                                    {(planner.peakUtilization * 100).toFixed(1)}%
                                </div>
                                <div className="text-xs text-gray-500">
                                    {planner.peakUtilization > 1 ? 'over capacity' : 'ok'}
                                </div>
                            </div>
                        </div>
                        <div className="text-[11px] text-gray-500">
                            Suggested burst is a per-session value. Set it per role in the config JSON under `rate_limits`.
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title={"Recommended Config Draft (".concat(selectedComponent, ")")} subtitle="Computed from the planner inputs. Copy into Gateway Configuration if desired."/>
                    <CardBody className="space-y-3">
                        <Legend>
                            Draft is component-scoped and preserves current hourly limits.
                        </Legend>
                        {selectedComponent !== plannerComponentKey && (<div className="text-xs text-amber-700">
                                Service capacity stays anchored to <span className="font-semibold">{plannerComponentKey}</span>. This draft
                                only changes rate limits/limits for <span className="font-semibold">{selectedComponent}</span>.
                            </div>)}
                        <TextArea value={recommendedConfigJson} onChange={function () { }}/>
                        <div className="text-[11px] text-gray-500">
                            This draft keeps current hourly limits, updates burst/burst_window, and mirrors the planner’s service capacity values.
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Circuit Breakers" subtitle="Live circuit states and resets."/>
                    <CardBody>
                        <Legend>
                            States and counters are aggregated per circuit; reset clears the circuit’s rolling failure window.
                        </Legend>
                        <div className="flex items-center gap-3 mb-4">
                            <Pill tone={(circuitSummary === null || circuitSummary === void 0 ? void 0 : circuitSummary.open_circuits) ? 'danger' : 'success'}>
                                Open: {(_18 = circuitSummary === null || circuitSummary === void 0 ? void 0 : circuitSummary.open_circuits) !== null && _18 !== void 0 ? _18 : 0}
                            </Pill>
                            <Pill tone="neutral">Half-open: {(_19 = circuitSummary === null || circuitSummary === void 0 ? void 0 : circuitSummary.half_open_circuits) !== null && _19 !== void 0 ? _19 : 0}</Pill>
                            <Pill tone="neutral">Closed: {(_20 = circuitSummary === null || circuitSummary === void 0 ? void 0 : circuitSummary.closed_circuits) !== null && _20 !== void 0 ? _20 : 0}</Pill>
                        </div>
                        <div className="space-y-3">
                            {Object.entries(circuitBreakers).map(function (_a) {
            var name = _a[0], cb = _a[1];
            return (<div key={name} className="flex items-center justify-between p-3 rounded-xl bg-gray-100">
                                    <div className="text-sm">
                                        <div className="font-semibold">{name}</div>
                                        <div className="text-xs text-gray-600">
                                            state: {cb.state} • failures: {cb.current_window_failures}/{cb.failure_count}
                                        </div>
                                    </div>
                                    <Button variant="secondary" onClick={function () { return resetCircuit(name); }}>
                                        Reset
                                    </Button>
                                </div>);
        })}
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Reset Throttling / Backpressure" subtitle="Clear rate-limit counters and backpressure slots."/>
                    <CardBody className="space-y-3">
                        <Legend>
                            Actions apply to the selected tenant/project. “All sessions” clears all rate-limit keys.
                        </Legend>
                        <div className="text-xs text-gray-600">
                            Active scope: <span className="font-semibold">{tenant || '—'}</span> / <span className="font-semibold">{project || '—'}</span>
                        </div>
                        <div className="text-[11px] text-gray-500">
                            Affected keys:
                            <div className="font-mono break-all">
                                {tenant && project ? "".concat(tenant, ":").concat(project, ":kdcube:system:ratelimit:<session_id>") : '<tenant>:<project>:kdcube:system:ratelimit:<session_id>'}
                            </div>
                            <div className="font-mono break-all">
                                {tenant && project ? "".concat(tenant, ":").concat(project, ":kdcube:system:capacity:counter") : '<tenant>:<project>:kdcube:system:capacity:counter'}
                            </div>
                            <div className="font-mono break-all">
                                {tenant && project ? "".concat(tenant, ":").concat(project, ":kdcube:throttling:*") : '<tenant>:<project>:kdcube:throttling:*'}
                            </div>
                            <div className="font-mono break-all">
                                {tenant && project ? "".concat(tenant, ":").concat(project, ":kdcube:chat:prompt:queue:*") : '<tenant>:<project>:kdcube:chat:prompt:queue:*'}
                            </div>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <Input label="Session ID (optional)" value={resetSessionId} onChange={function (e) { return setResetSessionId(e.target.value); }} placeholder="defaults to current session"/>
                            <div className="flex items-end">
                                <label className="text-xs text-gray-600 flex items-center gap-2">
                                    <input type="checkbox" checked={resetAllSessions} onChange={function (e) { return setResetAllSessions(e.target.checked); }}/>
                                    All sessions (danger)
                                </label>
                            </div>
                        </div>
                        <div className="flex flex-wrap items-center gap-4">
                            <label className="text-xs text-gray-600 flex items-center gap-2">
                                <input type="checkbox" checked={resetRateLimits} onChange={function (e) { return setResetRateLimits(e.target.checked); }}/>
                                Reset rate limits
                            </label>
                            <label className="text-xs text-gray-600 flex items-center gap-2">
                                <input type="checkbox" checked={resetBackpressure} onChange={function (e) { return setResetBackpressure(e.target.checked); }}/>
                                Reset backpressure counters
                            </label>
                            <label className="text-xs text-gray-600 flex items-center gap-2">
                                <input type="checkbox" checked={resetThrottlingStats} onChange={function (e) { return setResetThrottlingStats(e.target.checked); }}/>
                                Clear throttling stats
                            </label>
                            <label className="text-xs text-gray-600 flex items-center gap-2">
                                <input type="checkbox" checked={purgeChatQueues} onChange={function (e) { return setPurgeChatQueues(e.target.checked); }}/>
                                Purge chat queues (drops pending tasks)
                            </label>
                        </div>
                        {(resetAllSessions || purgeChatQueues) && (<div className="text-xs text-rose-700">
                                {resetAllSessions ? 'Warning: clears rate limits for all sessions in this tenant/project.' : ''}
                                {resetAllSessions && purgeChatQueues ? ' ' : ''}
                                {purgeChatQueues ? 'Warning: purging queues drops pending event payloads.' : ''}
                            </div>)}
                        <div className="flex flex-wrap items-center gap-3">
                            <Button variant="danger" onClick={handleResetThrottling} disabled={resettingThrottling}>
                                Reset
                            </Button>
                            {resetThrottlingMessage && (<span className="text-xs text-gray-600">{resetThrottlingMessage}</span>)}
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Gateway Configuration" subtitle="View, validate, update, or reset config."/>
                    <CardBody className="space-y-4">
                        <Legend>
                            Updates are stored in Redis cache and broadcast to live replicas for this tenant/project.
                        </Legend>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <Input label="Tenant" value={tenant} onChange={function (e) { return setTenant(e.target.value); }}/>
                            <Input label="Project" value={project} onChange={function (e) { return setProject(e.target.value); }}/>
                            <div className="flex items-end gap-3">
                                <label className="text-xs text-gray-600 flex items-center gap-2">
                                    <input type="checkbox" checked={dryRun} onChange={function (e) { return setDryRun(e.target.checked); }}/>
                                    Dry run reset
                                </label>
                            </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-600">
                            <span>Config source: {configSource}</span>
                        </div>

                        {configRaw && (<TextArea label="Current Gateway Config (read-only)" value={JSON.stringify(configRaw, null, 2)} onChange={function () { }}/>)}

                        <TextArea label="Gateway Config JSON (editable)" value={configJson} onChange={function (e) { return setConfigJson(e.target.value); }}/>

                        <div className="flex flex-wrap gap-3">
                            <Button variant="secondary" onClick={handleValidate}>Validate</Button>
                            <Button onClick={handleUpdate}>Update</Button>
                            <Button variant="danger" onClick={handleReset}>Reset to Env</Button>
                            {actionMessage && <span className="text-sm text-gray-600">{actionMessage}</span>}
                        </div>

                        <div className="text-xs text-amber-700">
                            Note: changing `service_capacity.processes_per_instance` requires a service restart to affect worker count.
                        </div>
                        <div className="text-xs text-gray-600">
                            Updates are persisted in the tenant/project cache and broadcast to running replicas.
                            Paste the full `GATEWAY_CONFIG_JSON` to replace the cached config.
                        </div>

                        <div className="flex flex-wrap items-center gap-3">
                            <Button variant="secondary" onClick={handleClearCache}>Clear Cached Config</Button>
                            <span className="text-xs text-gray-500">Cache key: {gatewayCacheKeyPattern}</span>
                            {clearCacheMessage && <span className="text-xs text-gray-600">{clearCacheMessage}</span>}
                        </div>

                        {validationResult && (<div className="mt-4 p-3 rounded-xl bg-gray-100 text-xs font-mono whitespace-pre-wrap">
                                {JSON.stringify(validationResult, null, 2)}
                            </div>)}
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Throttling (Recent)" subtitle="Last hour summary and recent events."/>
                    <CardBody>
                        <Legend>
                            Counts are for the last hour. Events list the most recent throttles (429/503).
                        </Legend>
                        {lastThrottle && (<div className="mb-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs">
                                <div className="font-semibold">Latest throttle</div>
                                <div>reason: {lastThrottle.reason}</div>
                                <div>endpoint: {lastThrottle.endpoint || '—'}</div>
                                <div>user_type: {lastThrottle.user_type || '—'} · status: {lastThrottle.http_status || '—'}</div>
                                {lastThrottle.retry_after ? (<div>retry_after: {lastThrottle.retry_after}s</div>) : null}
                            </div>)}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                            <div className="p-3 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Total</div>
                                <div className="text-sm font-semibold">{(_21 = throttling === null || throttling === void 0 ? void 0 : throttling.total_requests) !== null && _21 !== void 0 ? _21 : 0}</div>
                            </div>
                            <div className="p-3 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">Throttled</div>
                                <div className="text-sm font-semibold">{(_22 = throttling === null || throttling === void 0 ? void 0 : throttling.total_throttled) !== null && _22 !== void 0 ? _22 : 0}</div>
                            </div>
                            <div className="p-3 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">429</div>
                                <div className="text-sm font-semibold">{(_23 = throttling === null || throttling === void 0 ? void 0 : throttling.rate_limit_429) !== null && _23 !== void 0 ? _23 : 0}</div>
                            </div>
                            <div className="p-3 rounded-xl bg-gray-100">
                                <div className="text-xs text-gray-600">503</div>
                                <div className="text-sm font-semibold">{(_24 = throttling === null || throttling === void 0 ? void 0 : throttling.backpressure_503) !== null && _24 !== void 0 ? _24 : 0}</div>
                            </div>
                        </div>

                        <div className="space-y-2">
                            {events.slice(0, 10).map(function (e, idx) { return (<div key={e.event_id || idx} className="text-xs flex items-center justify-between bg-white border border-gray-200/70 rounded-xl px-3 py-2">
                                    <div className="text-gray-700">{e.reason}</div>
                                    <div className="text-gray-500">{e.endpoint || '—'}</div>
                                    <div className="text-gray-500">{e.user_type}</div>
                                    <div className="text-gray-500">{e.http_status}</div>
                                </div>); })}
                            {events.length === 0 && <div className="text-sm text-gray-500">No recent events.</div>}
                        </div>
                    </CardBody>
                </Card>
            </div>
        </div>);
};
// Render
var rootElement = document.getElementById('root');
if (rootElement) {
    var root = client_1.default.createRoot(rootElement);
    root.render(<MonitoringDashboard />);
}
