"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
var react_1 = require("react");
var client_1 = require("react-dom/client");
// =============================================================================
// Settings Manager
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
        this.PLACEHOLDER_HOST_BUNDLES_PATH = '{{' + 'HOST_BUNDLES_PATH' + '}}';
        this.PLACEHOLDER_AGENTIC_BUNDLES_ROOT = '{{' + 'AGENTIC_BUNDLES_ROOT' + '}}';
        this.settings = {
            baseUrl: '{{CHAT_BASE_URL}}',
            accessToken: '{{ACCESS_TOKEN}}',
            idToken: '{{ID_TOKEN}}',
            idTokenHeader: '{{ID_TOKEN_HEADER}}',
            defaultTenant: '{{DEFAULT_TENANT}}',
            defaultProject: '{{DEFAULT_PROJECT}}',
            defaultAppBundleId: '{{DEFAULT_APP_BUNDLE_ID}}',
            hostBundlesPath: '{{HOST_BUNDLES_PATH}}',
            agenticBundlesRoot: '{{AGENTIC_BUNDLES_ROOT}}'
        };
        this.configReceivedCallback = null;
    }
    SettingsManager.prototype.getBaseUrl = function () {
        if (this.settings.baseUrl === this.PLACEHOLDER_BASE_URL) {
            return 'http://localhost:8010';
        }
        try {
            var url = new URL(this.settings.baseUrl);
            if (url.port === 'None' || url.hostname.includes('None')) {
                return 'http://localhost:8010';
            }
            var trimmed = this.settings.baseUrl.replace(/\/+$/, '');
            return trimmed.endsWith('/api') ? trimmed.slice(0, -4) : trimmed;
        }
        catch (_a) {
            return 'http://localhost:8010';
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
    SettingsManager.prototype.getHostBundlesPath = function () {
        return this.settings.hostBundlesPath === this.PLACEHOLDER_HOST_BUNDLES_PATH
            ? ''
            : this.settings.hostBundlesPath;
    };
    SettingsManager.prototype.getAgenticBundlesRoot = function () {
        return this.settings.agenticBundlesRoot === this.PLACEHOLDER_AGENTIC_BUNDLES_ROOT
            ? ''
            : this.settings.agenticBundlesRoot;
    };
    SettingsManager.prototype.updateSettings = function (partial) {
        this.settings = __assign(__assign({}, this.settings), partial);
    };
    SettingsManager.prototype.hasPlaceholderSettings = function () {
        return this.settings.baseUrl === this.PLACEHOLDER_BASE_URL;
    };
    SettingsManager.prototype.onConfigReceived = function (callback) {
        this.configReceivedCallback = callback;
    };
    SettingsManager.prototype.setupParentListener = function () {
        var _this = this;
        var identity = "INTEGRATIONS_BUNDLES_ADMIN";
        window.addEventListener('message', function (event) {
            if (event.data.type === 'CONN_RESPONSE' || event.data.type === 'CONFIG_RESPONSE') {
                var requestedIdentity = event.data.identity;
                if (requestedIdentity !== identity) {
                    return;
                }
                if (event.data.config) {
                    var config = event.data.config;
                    var updates = {};
                    if (config.baseUrl && typeof config.baseUrl === 'string') {
                        updates.baseUrl = config.baseUrl;
                    }
                    if (config.accessToken !== undefined) {
                        updates.accessToken = config.accessToken;
                    }
                    if (config.idToken !== undefined) {
                        updates.idToken = config.idToken;
                    }
                    if (config.idTokenHeader) {
                        updates.idTokenHeader = config.idTokenHeader;
                    }
                    if (config.defaultTenant) {
                        updates.defaultTenant = config.defaultTenant;
                    }
                    if (config.defaultProject) {
                        updates.defaultProject = config.defaultProject;
                    }
                    if (config.defaultAppBundleId) {
                        updates.defaultAppBundleId = config.defaultAppBundleId;
                    }
                    if (config.hostBundlesPath) {
                        updates.hostBundlesPath = config.hostBundlesPath;
                    }
                    if (config.agenticBundlesRoot) {
                        updates.agenticBundlesRoot = config.agenticBundlesRoot;
                    }
                    if (Object.keys(updates).length > 0) {
                        _this.updateSettings(updates);
                        if (_this.configReceivedCallback) {
                            _this.configReceivedCallback();
                        }
                    }
                }
            }
        });
        if (this.hasPlaceholderSettings()) {
            window.parent.postMessage({
                type: 'CONFIG_REQUEST',
                data: {
                    requestedFields: [
                        'baseUrl', 'accessToken', 'idToken', 'idTokenHeader',
                        'defaultTenant', 'defaultProject', 'defaultAppBundleId',
                        'hostBundlesPath', 'agenticBundlesRoot'
                    ],
                    identity: identity
                }
            }, '*');
            return new Promise(function (resolve) {
                var timeout = setTimeout(function () {
                    resolve(false);
                }, 3000);
                var originalCallback = _this.configReceivedCallback;
                _this.onConfigReceived(function () {
                    clearTimeout(timeout);
                    if (originalCallback)
                        originalCallback();
                    resolve(true);
                });
            });
        }
        return Promise.resolve(!this.hasPlaceholderSettings());
    };
    return SettingsManager;
}());
var settings = new SettingsManager();
// =============================================================================
// Auth Helpers
// =============================================================================
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
function normalizeScope(tenant, project) {
    var t = (tenant || '').trim();
    var p = (project || '').trim();
    return {
        tenant: t || undefined,
        project: p || undefined
    };
}
function formatScopeLabel(tenant, project) {
    var t = (tenant || '').trim();
    var p = (project || '').trim();
    if (t && p)
        return "".concat(t, " / ").concat(p);
    if (t)
        return t;
    if (p)
        return p;
    return '';
}
function parseScopeValue(value) {
    var _a, _b;
    var raw = (value || '').trim();
    if (!raw)
        return {};
    var tenant = raw;
    var project = '';
    if (raw.includes('::')) {
        _a = raw.split('::', 2), tenant = _a[0], project = _a[1];
    }
    else if (raw.includes('/')) {
        _b = raw.split('/', 2), tenant = _b[0], project = _b[1];
    }
    return normalizeScope((tenant || '').trim(), (project || '').trim());
}
function buildScopeParams(scope) {
    if (!scope)
        return '';
    var params = new URLSearchParams();
    if (scope.tenant)
        params.set('tenant', scope.tenant);
    if (scope.project)
        params.set('project', scope.project);
    var query = params.toString();
    return query ? "?".concat(query) : '';
}
function withScope(payload, scope) {
    var out = __assign({}, payload);
    if ((scope === null || scope === void 0 ? void 0 : scope.tenant) && out.tenant === undefined) {
        out.tenant = scope.tenant;
    }
    if ((scope === null || scope === void 0 ? void 0 : scope.project) && out.project === undefined) {
        out.project = scope.project;
    }
    return out;
}
// =============================================================================
// Integrations API Client
// =============================================================================
var IntegrationsAPI = /** @class */ (function () {
    function IntegrationsAPI(basePath) {
        if (basePath === void 0) { basePath = '/admin/integrations'; }
        this.basePath = basePath;
    }
    IntegrationsAPI.prototype.buildUrl = function (path) {
        return "".concat(settings.getBaseUrl()).concat(this.basePath).concat(path);
    };
    IntegrationsAPI.prototype.fetchWithAuth = function (url_1) {
        return __awaiter(this, arguments, void 0, function (url, options) {
            var headers, response, errorText;
            if (options === void 0) { options = {}; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        headers = makeAuthHeaders(options.headers);
                        return [4 /*yield*/, fetch(url, __assign(__assign({}, options), { headers: headers }))];
                    case 1:
                        response = _a.sent();
                        if (!!response.ok) return [3 /*break*/, 3];
                        return [4 /*yield*/, response.text().catch(function () { return response.statusText; })];
                    case 2:
                        errorText = _a.sent();
                        throw new Error("API request failed: ".concat(response.status, " - ").concat(errorText));
                    case 3: return [2 /*return*/, response];
                }
            });
        });
    };
    IntegrationsAPI.prototype.listTenantProjects = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth("".concat(settings.getBaseUrl(), "/api/admin/control-plane/conversations/tenant-projects"))];
                    case 1:
                        response = _a.sent();
                        return [4 /*yield*/, response.json()];
                    case 2:
                        data = _a.sent();
                        return [2 /*return*/, data.items || []];
                }
            });
        });
    };
    IntegrationsAPI.prototype.listBundles = function (scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles".concat(buildScopeParams(scope))))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.updateBundles = function (payload, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl('/bundles'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope(payload, scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.reloadFromAuthority = function (scope, bundleId) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl('/bundles/reload-authority'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope(__assign({}, (bundleId ? { bundle_id: bundleId } : {})), scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.cleanupBundles = function (payload, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl('/bundles/cleanup'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope(payload, scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.getBundleProps = function (bundleId, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles/".concat(encodeURIComponent(bundleId), "/props").concat(buildScopeParams(scope))))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.setBundleProps = function (bundleId, payload, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles/".concat(encodeURIComponent(bundleId), "/props")), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope(payload, scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.resetBundlePropsFromCode = function (bundleId, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles/".concat(encodeURIComponent(bundleId), "/props/reset-code")), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope({}, scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.setBundleSecrets = function (bundleId, payload, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles/".concat(encodeURIComponent(bundleId), "/secrets")), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(withScope(payload, scope))
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    IntegrationsAPI.prototype.getBundleSecrets = function (bundleId, scope) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.buildUrl("/bundles/".concat(encodeURIComponent(bundleId), "/secrets").concat(buildScopeParams(scope))))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    return IntegrationsAPI;
}());
var api = new IntegrationsAPI();
// =============================================================================
// UI Components
// =============================================================================
var Card = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"bg-white rounded-2xl shadow-sm border border-gray-200/70 ".concat(className)}>{children}</div>);
};
var CardHeader = function (_a) {
    var title = _a.title, subtitle = _a.subtitle, action = _a.action;
    return (<div className="px-6 py-5 border-b border-gray-200/70">
        <div className="flex items-start justify-between gap-4">
            <div>
                <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
                {subtitle && <p className="mt-1 text-sm text-gray-600 leading-relaxed">{subtitle}</p>}
            </div>
            {action && <div className="pt-1">{action}</div>}
        </div>
    </div>);
};
var CardBody = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"px-6 py-5 ".concat(className)}>{children}</div>);
};
var Button = function (_a) {
    var children = _a.children, onClick = _a.onClick, _b = _a.type, type = _b === void 0 ? 'button' : _b, _c = _a.variant, variant = _c === void 0 ? 'primary' : _c, _d = _a.disabled, disabled = _d === void 0 ? false : _d;
    var variants = {
        primary: 'bg-gray-900 text-white hover:bg-gray-800',
        secondary: 'bg-gray-100 text-gray-800 hover:bg-gray-200',
        danger: 'bg-red-600 text-white hover:bg-red-500'
    };
    return (<button type={type} onClick={onClick} disabled={disabled} className={"px-4 py-2.5 rounded-xl text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ".concat(variants[variant])}>
            {children}
        </button>);
};
var InputField = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, placeholder = _a.placeholder, listId = _a.listId;
    return (<div>
        <label className="block text-sm font-medium text-gray-800 mb-2">{label}</label>
        <input className="w-full px-4 py-2.5 border border-gray-200/80 rounded-xl bg-white text-sm focus:outline-none focus:ring-2 focus:ring-gray-900/10" value={value} onChange={function (e) { return onChange(e.target.value); }} placeholder={placeholder} list={listId}/>
    </div>);
};
var isRecord = function (value) { return (typeof value === 'object' && value !== null && !Array.isArray(value)); };
var normalizeDotPath = function (raw) { return (raw
    .split('.')
    .map(function (part) { return part.trim(); })
    .filter(Boolean)); };
var setNestedValue = function (target, path, value) {
    var next = __assign({}, (isRecord(target) ? target : {}));
    var cursor = next;
    path.forEach(function (part, idx) {
        if (idx === path.length - 1) {
            cursor[part] = value;
            return;
        }
        var existing = cursor[part];
        if (!isRecord(existing)) {
            cursor[part] = {};
        }
        else {
            cursor[part] = __assign({}, existing);
        }
        cursor = cursor[part];
    });
    return next;
};
var deleteNestedValue = function (target, path) {
    var next = __assign({}, (isRecord(target) ? target : {}));
    var cursor = next;
    path.forEach(function (part, idx) {
        if (idx === path.length - 1) {
            delete cursor[part];
            return;
        }
        var existing = cursor[part];
        if (!isRecord(existing)) {
            cursor[part] = {};
        }
        else {
            cursor[part] = __assign({}, existing);
        }
        cursor = cursor[part];
    });
    return next;
};
var buildNestedObject = function (path, value) {
    return path.reduceRight(function (acc, key) {
        var _a;
        return (_a = {}, _a[key] = acc, _a);
    }, value);
};
var parseJsonValue = function (raw) {
    var trimmed = raw.trim();
    if (!trimmed) {
        return { ok: false, error: 'Value is required.' };
    }
    try {
        return { ok: true, value: JSON.parse(trimmed) };
    }
    catch (_a) {
        return { ok: true, value: trimmed };
    }
};
var extractDotKeys = function (node, out, prefix) {
    if (prefix === void 0) { prefix = ''; }
    if (node === null || node === undefined) {
        return;
    }
    if (Array.isArray(node)) {
        node.forEach(function (value, idx) {
            var nextPrefix = prefix ? "".concat(prefix, ".").concat(idx) : "".concat(idx);
            extractDotKeys(value, out, nextPrefix);
        });
        return;
    }
    if (isRecord(node)) {
        Object.entries(node).forEach(function (_a) {
            var key = _a[0], value = _a[1];
            var nextPrefix = prefix ? "".concat(prefix, ".").concat(key) : key;
            extractDotKeys(value, out, nextPrefix);
        });
        return;
    }
    if (prefix) {
        out.push(prefix);
    }
};
var deepMergeObjects = function (base, patch) {
    var merged = __assign({}, (base || {}));
    Object.entries(patch || {}).forEach(function (_a) {
        var key = _a[0], value = _a[1];
        var baseValue = merged[key];
        if (isRecord(baseValue) && isRecord(value)) {
            merged[key] = deepMergeObjects(baseValue, value);
        }
        else {
            merged[key] = value;
        }
    });
    return merged;
};
// =============================================================================
// Main Component
// =============================================================================
var AIBundleDashboard = function () {
    var _a = (0, react_1.useState)(true), loading = _a[0], setLoading = _a[1];
    var _b = (0, react_1.useState)(false), configReady = _b[0], setConfigReady = _b[1];
    var _c = (0, react_1.useState)(null), error = _c[0], setError = _c[1];
    var _d = (0, react_1.useState)({}), bundles = _d[0], setBundles = _d[1];
    var _e = (0, react_1.useState)(''), defaultBundleId = _e[0], setDefaultBundleId = _e[1];
    var _f = (0, react_1.useState)(null), bundleAuthority = _f[0], setBundleAuthority = _f[1];
    var _g = (0, react_1.useState)(null), editingId = _g[0], setEditingId = _g[1];
    var _h = (0, react_1.useState)(null), reloadingBundleId = _h[0], setReloadingBundleId = _h[1];
    var _j = (0, react_1.useState)(settings.getDefaultTenant()), scopeTenant = _j[0], setScopeTenant = _j[1];
    var _k = (0, react_1.useState)(settings.getDefaultProject()), scopeProject = _k[0], setScopeProject = _k[1];
    var _l = (0, react_1.useState)(formatScopeLabel(settings.getDefaultTenant(), settings.getDefaultProject())), scopeInput = _l[0], setScopeInput = _l[1];
    var _m = (0, react_1.useState)([]), tenantProjects = _m[0], setTenantProjects = _m[1];
    var _o = (0, react_1.useState)(false), tenantProjectsLoading = _o[0], setTenantProjectsLoading = _o[1];
    var _p = (0, react_1.useState)(null), tenantProjectsError = _p[0], setTenantProjectsError = _p[1];
    var _q = (0, react_1.useState)(''), propsBundleId = _q[0], setPropsBundleId = _q[1];
    var _r = (0, react_1.useState)('{}'), propsJson = _r[0], setPropsJson = _r[1];
    var _s = (0, react_1.useState)('{}'), propsDefaultsJson = _s[0], setPropsDefaultsJson = _s[1];
    var _t = (0, react_1.useState)(false), propsLoading = _t[0], setPropsLoading = _t[1];
    var _u = (0, react_1.useState)(''), propsKeyPath = _u[0], setPropsKeyPath = _u[1];
    var _v = (0, react_1.useState)(''), propsValue = _v[0], setPropsValue = _v[1];
    var _w = (0, react_1.useState)(''), secretsBundleId = _w[0], setSecretsBundleId = _w[1];
    var _x = (0, react_1.useState)('{}'), secretsJson = _x[0], setSecretsJson = _x[1];
    var _y = (0, react_1.useState)(false), secretsSaving = _y[0], setSecretsSaving = _y[1];
    var _z = (0, react_1.useState)(null), secretsStatus = _z[0], setSecretsStatus = _z[1];
    var _0 = (0, react_1.useState)([]), secretsKeys = _0[0], setSecretsKeys = _0[1];
    var _1 = (0, react_1.useState)(false), secretsLoading = _1[0], setSecretsLoading = _1[1];
    var _2 = (0, react_1.useState)(''), secretsKeyPath = _2[0], setSecretsKeyPath = _2[1];
    var _3 = (0, react_1.useState)(''), secretsValue = _3[0], setSecretsValue = _3[1];
    var registryScope = (0, react_1.useMemo)(function () { return normalizeScope(scopeTenant, scopeProject); }, [scopeTenant, scopeProject]);
    var propsScope = (0, react_1.useMemo)(function () { return normalizeScope(scopeTenant, scopeProject); }, [scopeTenant, scopeProject]);
    var draftScope = (0, react_1.useMemo)(function () { return parseScopeValue(scopeInput); }, [scopeInput]);
    var scopeDirty = (0, react_1.useMemo)(function () {
        var applied = normalizeScope(scopeTenant, scopeProject);
        return applied.tenant !== draftScope.tenant || applied.project !== draftScope.project;
    }, [scopeTenant, scopeProject, draftScope]);
    var bundleVersion = (0, react_1.useMemo)(function () {
        try {
            var parsed = JSON.parse(propsDefaultsJson || '{}');
            return typeof (parsed === null || parsed === void 0 ? void 0 : parsed.bundle_version) === 'string' ? parsed.bundle_version : '';
        }
        catch (_a) {
            return '';
        }
    }, [propsDefaultsJson]);
    var authorityLabel = (0, react_1.useMemo)(function () {
        var label = ((bundleAuthority === null || bundleAuthority === void 0 ? void 0 : bundleAuthority.label) || '').trim();
        return label || 'configured bundle authority';
    }, [bundleAuthority]);
    var authorityDescription = (0, react_1.useMemo)(function () {
        var description = ((bundleAuthority === null || bundleAuthority === void 0 ? void 0 : bundleAuthority.description) || '').trim();
        return description || "Reload from ".concat(authorityLabel, ".");
    }, [bundleAuthority, authorityLabel]);
    var authorityDetail = (0, react_1.useMemo)(function () {
        var detail = ((bundleAuthority === null || bundleAuthority === void 0 ? void 0 : bundleAuthority.detail) || '').trim();
        return detail;
    }, [bundleAuthority]);
    var reloadAuthorityLabel = (0, react_1.useMemo)(function () { return "Reload from ".concat(authorityLabel); }, [authorityLabel]);
    var propsResolutionLabel = (0, react_1.useMemo)(function () { return authorityLabel; }, [authorityLabel]);
    var bundleSnapshotPath = (0, react_1.useMemo)(function () {
        if (!bundleVersion || !propsBundleId || !scopeTenant || !scopeProject)
            return '';
        return "cb/tenants/".concat(scopeTenant, "/projects/").concat(scopeProject, "/ai-bundle-snapshots/").concat(propsBundleId, ".").concat(bundleVersion, ".zip");
    }, [bundleVersion, propsBundleId, scopeTenant, scopeProject]);
    var copyText = function (value) { return __awaiter(void 0, void 0, void 0, function () {
        var _a, el;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    if (!value)
                        return [2 /*return*/];
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, navigator.clipboard.writeText(value)];
                case 2:
                    _b.sent();
                    return [3 /*break*/, 4];
                case 3:
                    _a = _b.sent();
                    try {
                        el = document.createElement('textarea');
                        el.value = value;
                        el.style.position = 'fixed';
                        el.style.opacity = '0';
                        document.body.appendChild(el);
                        el.select();
                        document.execCommand('copy');
                        document.body.removeChild(el);
                    }
                    catch (_c) {
                        // no-op
                    }
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var _4 = (0, react_1.useState)({
        id: '',
        name: '',
        path: '',
        module: '',
        singleton: false,
        description: '',
        repo: '',
        ref: '',
        subdir: ''
    }), form = _4[0], setForm = _4[1];
    var formRef = (0, react_1.useRef)(null);
    var bundleList = (0, react_1.useMemo)(function () { return Object.values(bundles).sort(function (a, b) { return a.id.localeCompare(b.id); }); }, [bundles]);
    var deriveRepoName = function (repoUrl) {
        var trimmed = (repoUrl || '').trim().replace(/\/+$/, '');
        if (!trimmed)
            return '';
        var last = trimmed.split('/').pop() || '';
        return last.endsWith('.git') ? last.slice(0, -4) : last;
    };
    var derivedGitPath = (0, react_1.useMemo)(function () {
        if (!form.repo)
            return '';
        var id = form.id || '<bundle_id>';
        var ref = (form.ref || '').trim();
        var subdir = (form.subdir || '').trim();
        var repo = deriveRepoName(form.repo) || '<repo>';
        var base = "<bundles_root>/".concat(repo, "__").concat(id).concat(ref ? "__".concat(ref) : '');
        return subdir ? "".concat(base, "/").concat(subdir) : base;
    }, [form.repo, form.ref, form.subdir, form.id]);
    var derivedHostPath = (0, react_1.useMemo)(function () {
        if (!form.repo)
            return '';
        var root = settings.getHostBundlesPath() || '<HOST_BUNDLES_PATH>';
        var id = form.id || '<bundle_id>';
        var ref = (form.ref || '').trim();
        var subdir = (form.subdir || '').trim();
        var repo = deriveRepoName(form.repo) || '<repo>';
        var base = "".concat(root.replace(/\/+$/, ''), "/").concat(repo, "__").concat(id).concat(ref ? "__".concat(ref) : '');
        return subdir ? "".concat(base, "/").concat(subdir) : base;
    }, [form.repo, form.ref, form.subdir, form.id]);
    var derivedAgenticPath = (0, react_1.useMemo)(function () {
        if (!form.repo)
            return '';
        var root = settings.getAgenticBundlesRoot() || '<AGENTIC_BUNDLES_ROOT>';
        var id = form.id || '<bundle_id>';
        var ref = (form.ref || '').trim();
        var subdir = (form.subdir || '').trim();
        var repo = deriveRepoName(form.repo) || '<repo>';
        var base = "".concat(root.replace(/\/+$/, ''), "/").concat(repo, "__").concat(id).concat(ref ? "__".concat(ref) : '');
        return subdir ? "".concat(base, "/").concat(subdir) : base;
    }, [form.repo, form.ref, form.subdir, form.id]);
    var loadBundles = function (scopeOverride) { return __awaiter(void 0, void 0, void 0, function () {
        var data, e_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, 3, 4]);
                    setLoading(true);
                    return [4 /*yield*/, api.listBundles(scopeOverride !== null && scopeOverride !== void 0 ? scopeOverride : registryScope)];
                case 1:
                    data = _a.sent();
                    setBundles(data.available_bundles || {});
                    setDefaultBundleId(data.default_bundle_id || '');
                    setBundleAuthority(data.authority || null);
                    if (!propsBundleId || !(propsBundleId in (data.available_bundles || {}))) {
                        setPropsBundleId(data.default_bundle_id || '');
                    }
                    if (!secretsBundleId || !(secretsBundleId in (data.available_bundles || {}))) {
                        setSecretsBundleId(data.default_bundle_id || '');
                    }
                    setError(null);
                    return [3 /*break*/, 4];
                case 2:
                    e_1 = _a.sent();
                    setError(e_1.message || 'Failed to load bundles');
                    setBundleAuthority(null);
                    return [3 /*break*/, 4];
                case 3:
                    setLoading(false);
                    return [7 /*endfinally*/];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var loadProps = function () { return __awaiter(void 0, void 0, void 0, function () {
        var data, props, defaults, merged, e_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!propsBundleId)
                        return [2 /*return*/];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    setPropsLoading(true);
                    return [4 /*yield*/, api.getBundleProps(propsBundleId, propsScope)];
                case 2:
                    data = _a.sent();
                    props = data.props || {};
                    defaults = data.defaults || {};
                    merged = deepMergeObjects(defaults, props);
                    setPropsJson(JSON.stringify(merged, null, 2));
                    setPropsDefaultsJson(JSON.stringify(defaults, null, 2));
                    return [3 /*break*/, 5];
                case 3:
                    e_2 = _a.sent();
                    setError(e_2.message || 'Failed to load bundle props');
                    return [3 /*break*/, 5];
                case 4:
                    setPropsLoading(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var parseJsonObject = function (raw, label) {
        var trimmed = raw.trim();
        if (!trimmed) {
            return {};
        }
        try {
            var parsed = JSON.parse(trimmed);
            if (!isRecord(parsed)) {
                throw new Error("".concat(label, " must be a JSON object."));
            }
            return parsed;
        }
        catch (err) {
            var message = (err === null || err === void 0 ? void 0 : err.message) ? String(err.message) : '';
            throw new Error(message || "Invalid ".concat(label, " JSON."));
        }
    };
    var collectSecretKeys = function (payload) {
        var keys = [];
        extractDotKeys(payload, keys);
        return keys.sort();
    };
    var applyPropsDotPath = function (mode) {
        var path = normalizeDotPath(propsKeyPath);
        if (!path.length) {
            setError('Enter a dot-path for props.');
            return;
        }
        try {
            var parsed = parseJsonObject(propsJson, 'Props');
            var updated = parsed;
            if (mode === 'set') {
                var parsedValue = parseJsonValue(propsValue);
                if (!parsedValue.ok) {
                    setError(parsedValue.error);
                    return;
                }
                updated = setNestedValue(parsed, path, parsedValue.value);
            }
            else {
                updated = deleteNestedValue(parsed, path);
            }
            setPropsJson(JSON.stringify(updated, null, 2));
            setError(null);
        }
        catch (e) {
            setError(e.message || 'Failed to update props.');
        }
    };
    var submitSecretDotPath = function (mode) { return __awaiter(void 0, void 0, void 0, function () {
        var path, value, parsedValue, payload, response, e_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!secretsBundleId) {
                        setError('Select a bundle to update secrets.');
                        return [2 /*return*/];
                    }
                    path = normalizeDotPath(secretsKeyPath);
                    if (!path.length) {
                        setError('Enter a dot-path for secrets.');
                        return [2 /*return*/];
                    }
                    value = true;
                    if (mode === 'set') {
                        parsedValue = parseJsonValue(secretsValue);
                        if (!parsedValue.ok) {
                            setError(parsedValue.error);
                            return [2 /*return*/];
                        }
                        value = parsedValue.value;
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    setSecretsSaving(true);
                    payload = buildNestedObject(path, value);
                    return [4 /*yield*/, api.setBundleSecrets(secretsBundleId, { secrets: payload, mode: mode }, propsScope)];
                case 2:
                    response = _a.sent();
                    setSecretsStatus({ mode: mode, keys: response.keys || [] });
                    if (response.stored_keys) {
                        setSecretsKeys(response.stored_keys);
                    }
                    else if (response.keys) {
                        setSecretsKeys(response.keys);
                    }
                    setError(null);
                    return [3 /*break*/, 5];
                case 3:
                    e_3 = _a.sent();
                    setError(e_3.message || 'Failed to update secrets');
                    return [3 /*break*/, 5];
                case 4:
                    setSecretsSaving(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    (0, react_1.useEffect)(function () {
        var applyDefaults = function () {
            var nextTenant = settings.getDefaultTenant();
            var nextProject = settings.getDefaultProject();
            setScopeTenant(nextTenant);
            setScopeProject(nextProject);
            setScopeInput(formatScopeLabel(nextTenant, nextProject));
        };
        settings.setupParentListener()
            .then(function () {
            applyDefaults();
            setConfigReady(true);
        })
            .catch(function () {
            applyDefaults();
            setConfigReady(true);
        });
    }, []);
    (0, react_1.useEffect)(function () {
        if (!configReady)
            return;
        loadBundles();
    }, [configReady]);
    (0, react_1.useEffect)(function () {
        if (!configReady)
            return;
        setTenantProjectsLoading(true);
        setTenantProjectsError(null);
        api.listTenantProjects()
            .then(setTenantProjects)
            .catch(function (err) { return setTenantProjectsError(err.message || 'Failed to load tenant/projects'); })
            .finally(function () { return setTenantProjectsLoading(false); });
    }, [configReady]);
    (0, react_1.useEffect)(function () {
        if (!propsBundleId)
            return;
        loadProps();
    }, [propsBundleId, scopeTenant, scopeProject]);
    var loadSecrets = function () { return __awaiter(void 0, void 0, void 0, function () {
        var data, e_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!secretsBundleId)
                        return [2 /*return*/];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    setSecretsLoading(true);
                    return [4 /*yield*/, api.getBundleSecrets(secretsBundleId, propsScope)];
                case 2:
                    data = _a.sent();
                    setSecretsKeys(data.keys || []);
                    return [3 /*break*/, 5];
                case 3:
                    e_4 = _a.sent();
                    setError(e_4.message || 'Failed to load bundle secrets');
                    return [3 /*break*/, 5];
                case 4:
                    setSecretsLoading(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var reloadBundleFromAuthority = function (bundleId) { return __awaiter(void 0, void 0, void 0, function () {
        var e_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!bundleId)
                        return [2 /*return*/];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 8, 9, 10]);
                    setReloadingBundleId(bundleId);
                    return [4 /*yield*/, api.reloadFromAuthority(registryScope, bundleId)];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, loadBundles()];
                case 3:
                    _a.sent();
                    if (!(propsBundleId === bundleId)) return [3 /*break*/, 5];
                    return [4 /*yield*/, loadProps()];
                case 4:
                    _a.sent();
                    _a.label = 5;
                case 5:
                    if (!(secretsBundleId === bundleId)) return [3 /*break*/, 7];
                    return [4 /*yield*/, loadSecrets()];
                case 6:
                    _a.sent();
                    _a.label = 7;
                case 7:
                    setError(null);
                    return [3 /*break*/, 10];
                case 8:
                    e_5 = _a.sent();
                    setError(e_5.message || "Failed to reload bundle ".concat(bundleId, " from ").concat(authorityLabel));
                    return [3 /*break*/, 10];
                case 9:
                    setReloadingBundleId(null);
                    return [7 /*endfinally*/];
                case 10: return [2 /*return*/];
            }
        });
    }); };
    (0, react_1.useEffect)(function () {
        if (!secretsBundleId)
            return;
        loadSecrets();
    }, [secretsBundleId, scopeTenant, scopeProject]);
    var resetForm = function () {
        setEditingId(null);
        setForm({ id: '', name: '', path: '', module: '', singleton: false, description: '', repo: '', ref: '', subdir: '' });
    };
    var saveBundle = function () { return __awaiter(void 0, void 0, void 0, function () {
        var payload, e_6;
        var _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    if (!form.id || (!form.path && !form.repo)) {
                        setError('Bundle id is required. Provide either a path or a repo.');
                        return [2 /*return*/];
                    }
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 4, , 5]);
                    payload = __assign(__assign({}, form), { path: form.repo ? '' : form.path, git_commit: undefined, singleton: !!form.singleton });
                    return [4 /*yield*/, api.updateBundles({
                            op: 'merge',
                            bundles: (_a = {}, _a[payload.id] = payload, _a),
                            default_bundle_id: defaultBundleId || undefined
                        }, registryScope)];
                case 2:
                    _b.sent();
                    resetForm();
                    return [4 /*yield*/, loadBundles()];
                case 3:
                    _b.sent();
                    return [3 /*break*/, 5];
                case 4:
                    e_6 = _b.sent();
                    setError(e_6.message || 'Failed to save bundle');
                    return [3 /*break*/, 5];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var deleteBundle = function (id) { return __awaiter(void 0, void 0, void 0, function () {
        var next, nextDefault, e_7;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    next = __assign({}, bundles);
                    delete next[id];
                    nextDefault = defaultBundleId === id ? (Object.keys(next)[0] || '') : defaultBundleId;
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    return [4 /*yield*/, api.updateBundles({
                            op: 'replace',
                            bundles: next,
                            default_bundle_id: nextDefault || undefined
                        }, registryScope)];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, loadBundles()];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 5];
                case 4:
                    e_7 = _a.sent();
                    setError(e_7.message || 'Failed to delete bundle');
                    return [3 /*break*/, 5];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var editBundle = function (entry) {
        setEditingId(entry.id);
        setForm({
            id: entry.id,
            name: entry.name || '',
            path: entry.path || '',
            module: entry.module || '',
            singleton: !!entry.singleton,
            description: entry.description || '',
            repo: entry.repo || '',
            ref: entry.ref || '',
            subdir: entry.subdir || ''
        });
        setTimeout(function () { var _a; return (_a = formRef.current) === null || _a === void 0 ? void 0 : _a.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 0);
    };
    var updateDefault = function () { return __awaiter(void 0, void 0, void 0, function () {
        var e_8;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    return [4 /*yield*/, api.updateBundles({
                            op: 'merge',
                            bundles: {},
                            default_bundle_id: defaultBundleId || undefined
                        }, registryScope)];
                case 1:
                    _a.sent();
                    return [4 /*yield*/, loadBundles()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 4];
                case 3:
                    e_8 = _a.sent();
                    setError(e_8.message || 'Failed to update default bundle');
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var reloadFromAuthority = function () { return __awaiter(void 0, void 0, void 0, function () {
        var e_9;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    return [4 /*yield*/, api.reloadFromAuthority(registryScope)];
                case 1:
                    _a.sent();
                    return [4 /*yield*/, loadBundles()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 4];
                case 3:
                    e_9 = _a.sent();
                    setError(e_9.message || "Failed to reload from ".concat(authorityLabel));
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var cleanupBundles = function () { return __awaiter(void 0, void 0, void 0, function () {
        var e_10;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    return [4 /*yield*/, api.cleanupBundles({ drop_sys_modules: true }, registryScope)];
                case 1:
                    _a.sent();
                    return [3 /*break*/, 3];
                case 2:
                    e_10 = _a.sent();
                    setError(e_10.message || 'Failed to cleanup bundles');
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); };
    var saveProps = function (op) { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, e_11;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!propsBundleId) {
                        setError('Select a bundle to update props.');
                        return [2 /*return*/];
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    parsed = parseJsonObject(propsJson, 'Props');
                    return [4 /*yield*/, api.setBundleProps(propsBundleId, {
                            op: op,
                            props: parsed
                        }, propsScope)];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, loadProps()];
                case 3:
                    _a.sent();
                    setError(null);
                    return [3 /*break*/, 5];
                case 4:
                    e_11 = _a.sent();
                    setError(e_11.message || 'Failed to update props');
                    return [3 /*break*/, 5];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var resetPropsFromCode = function () { return __awaiter(void 0, void 0, void 0, function () {
        var e_12;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!propsBundleId) {
                        setError('Select a bundle to reset props.');
                        return [2 /*return*/];
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    return [4 /*yield*/, api.resetBundlePropsFromCode(propsBundleId, propsScope)];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, loadProps()];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 5];
                case 4:
                    e_12 = _a.sent();
                    setError(e_12.message || 'Failed to reset props from code');
                    return [3 /*break*/, 5];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var saveSecrets = function () { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, keys, response, e_13;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!secretsBundleId) {
                        setError('Select a bundle to update secrets.');
                        return [2 /*return*/];
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    setSecretsSaving(true);
                    parsed = parseJsonObject(secretsJson, 'Secrets');
                    keys = collectSecretKeys(parsed);
                    if (!keys.length) {
                        setError('Provide at least one secret key to save.');
                        return [2 /*return*/];
                    }
                    return [4 /*yield*/, api.setBundleSecrets(secretsBundleId, { secrets: parsed, mode: 'set' }, propsScope)];
                case 2:
                    response = _a.sent();
                    setSecretsStatus({ mode: 'set', keys: response.keys || [] });
                    if (response.stored_keys) {
                        setSecretsKeys(response.stored_keys);
                    }
                    else if (response.keys) {
                        setSecretsKeys(response.keys);
                    }
                    setError(null);
                    return [3 /*break*/, 5];
                case 3:
                    e_13 = _a.sent();
                    setError(e_13.message || 'Failed to update secrets');
                    return [3 /*break*/, 5];
                case 4:
                    setSecretsSaving(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var clearSecrets = function () { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, keys, confirmed, response, e_14;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!secretsBundleId) {
                        setError('Select a bundle to clear secrets.');
                        return [2 /*return*/];
                    }
                    parsed = {};
                    try {
                        parsed = parseJsonObject(secretsJson, 'Secrets');
                    }
                    catch (e) {
                        setError(e.message || 'Invalid secrets JSON.');
                        return [2 /*return*/];
                    }
                    keys = collectSecretKeys(parsed);
                    if (!keys.length) {
                        setError('Provide at least one secret key to clear.');
                        return [2 /*return*/];
                    }
                    confirmed = window.confirm("Clear these secrets for this bundle?\\n- ".concat(keys.join('\\n- '), "\\nThis cannot be undone."));
                    if (!confirmed)
                        return [2 /*return*/];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    setSecretsSaving(true);
                    return [4 /*yield*/, api.setBundleSecrets(secretsBundleId, { secrets: parsed, mode: 'clear' }, propsScope)];
                case 2:
                    response = _a.sent();
                    setSecretsStatus({ mode: 'clear', keys: response.keys || [] });
                    if (response.stored_keys) {
                        setSecretsKeys(response.stored_keys);
                    }
                    else if (response.keys) {
                        setSecretsKeys(response.keys);
                    }
                    setError(null);
                    return [3 /*break*/, 5];
                case 3:
                    e_14 = _a.sent();
                    setError(e_14.message || 'Failed to clear secrets');
                    return [3 /*break*/, 5];
                case 4:
                    setSecretsSaving(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var applyScope = function () { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, nextTenant, nextProject;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    parsed = parseScopeValue(scopeInput);
                    nextTenant = parsed.tenant || '';
                    nextProject = parsed.project || '';
                    setScopeTenant(nextTenant);
                    setScopeProject(nextProject);
                    setScopeInput(formatScopeLabel(nextTenant, nextProject));
                    return [4 /*yield*/, loadBundles(parsed)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); };
    if (loading) {
        return (<div className="min-h-screen bg-white flex items-center justify-center p-8">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-200 border-t-gray-900"></div>
                    <p className="mt-4 text-gray-600">Loading AI bundle registry…</p>
                </div>
            </div>);
    }
    return (<div className="min-h-screen bg-white">
            <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">
                <div className="text-center">
                    <h1 className="text-4xl md:text-5xl font-semibold text-gray-900 tracking-tight">AI Bundles</h1>
                    <div className="mt-3 flex justify-center">
                        <div className="h-1 w-24 bg-gray-900 rounded-full opacity-80"></div>
                    </div>
                    <p className="mt-4 text-gray-600 text-base md:text-lg leading-relaxed">
                        Manage dynamic bundles (plugins) and set the default bundle for the tenant/project.
                    </p>
                </div>

                <Card>
                    <CardHeader title="Tenant / Project" subtitle="All registry and bundle props operations use this scope."/>
                    <CardBody>
                        <InputField label="Tenant / Project" value={scopeInput} onChange={function (v) { return setScopeInput(v); }} placeholder={formatScopeLabel(settings.getDefaultTenant(), settings.getDefaultProject())} listId="tenant-project-options"/>
                        <datalist id="tenant-project-options">
                            {tenantProjects.map(function (tp) {
            var value = formatScopeLabel(tp.tenant, tp.project);
            return (<option key={"".concat(tp.tenant, "::").concat(tp.project)} value={value} label={value}/>);
        })}
                        </datalist>
                        <div className="mt-4 flex items-center gap-3">
                            <Button variant="primary" onClick={applyScope} disabled={!scopeDirty}>
                                Apply scope
                            </Button>
                            {!scopeDirty ? (<span className="text-xs text-gray-500">Scope is up to date.</span>) : null}
                            {tenantProjectsLoading ? (<span className="text-xs text-gray-500">Loading tenant/projects…</span>) : null}
                            {!tenantProjectsLoading && tenantProjectsError ? (<span className="text-xs text-red-600">{tenantProjectsError}</span>) : null}
                        </div>
                    </CardBody>
                </Card>

                {error && (<div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                        {error}
                    </div>)}

                <Card>
                    <CardHeader title="Registry" subtitle={"Current bundles stored in the registry. ".concat(authorityDescription, " This replaces the runtime registry and descriptor-backed bundle props from that source.")} action={<div className="flex gap-2">
                                <Button variant="secondary" onClick={loadBundles}>Refresh</Button>
                                <Button variant="secondary" onClick={reloadFromAuthority}>{reloadAuthorityLabel}</Button>
                                <Button variant="secondary" onClick={cleanupBundles}>Cleanup old versions</Button>
                            </div>}/>
                    <CardBody className="space-y-4">
                        <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs text-gray-600">
                            <div>
                                <strong className="text-gray-800">Current reload source:</strong> {authorityLabel}
                            </div>
                            {authorityDetail ? (<div className="mt-1 break-all">
                                    <strong className="text-gray-800">Location:</strong> {authorityDetail}
                                </div>) : null}
                        </div>
                        <div className="flex items-center gap-3">
                            <label className="text-sm font-medium text-gray-800">Default bundle</label>
                            <select className="px-3 py-2 border border-gray-200 rounded-lg text-sm" value={defaultBundleId} onChange={function (e) { return setDefaultBundleId(e.target.value); }}>
                                <option value="">—</option>
                                {bundleList.map(function (b) { return (<option key={b.id} value={b.id}>{b.id}</option>); })}
                            </select>
                            <Button variant="primary" onClick={updateDefault}>Save default</Button>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead className="bg-gray-50 border-b border-gray-200/70">
                                    <tr className="text-gray-600">
                                        <th className="px-4 py-3 text-left font-semibold">ID</th>
                                        <th className="px-4 py-3 text-left font-semibold">Name</th>
                                        <th className="px-4 py-3 text-left font-semibold">Path</th>
                                        <th className="px-4 py-3 text-left font-semibold">Module</th>
                                        <th className="px-4 py-3 text-left font-semibold">Singleton</th>
                                        <th className="px-4 py-3 text-left font-semibold">Description</th>
                                        <th className="px-4 py-3 text-left font-semibold">Version</th>
                                        <th className="px-4 py-3 text-left font-semibold">Git</th>
                                        <th className="px-4 py-3 text-right font-semibold">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-200/70">
                                    {bundleList.map(function (b) {
            var isAdminBundle = b.id === 'kdcube.admin';
            return (<tr key={b.id} className="hover:bg-gray-50/70 transition-colors">
                                            <td className="px-4 py-3 font-semibold text-gray-900">{b.id}</td>
                                            <td className="px-4 py-3 text-gray-700">{b.name || '—'}</td>
                                            <td className="px-4 py-3 text-gray-700">{b.path}</td>
                                            <td className="px-4 py-3 text-gray-700">{b.module || '—'}</td>
                                            <td className="px-4 py-3 text-gray-700">{b.singleton ? 'true' : 'false'}</td>
                                            <td className="px-4 py-3 text-gray-600">{b.description || '—'}</td>
                                            <td className="px-4 py-3 text-gray-600">{b.version || '—'}</td>
                                            <td className="px-4 py-3 text-gray-600">
                                                {b.repo ? (<div className="space-y-1">
                                                        <div className="truncate max-w-[220px]" title={b.repo || ''}>{b.repo}</div>
                                                        {b.ref && <div>ref: {b.ref}</div>}
                                                        {b.git_commit && <div className="text-xs text-gray-500">commit: {b.git_commit.slice(0, 12)}</div>}
                                                    </div>) : '—'}
                                            </td>
                                            <td className="px-4 py-3 text-right">
                                                <div className="flex justify-end gap-2">
                                                    <Button variant="secondary" onClick={function () { return reloadBundleFromAuthority(b.id); }} disabled={reloadingBundleId === b.id} title={"Reload ".concat(b.id, " from ").concat(authorityLabel)}>
                                                        {reloadingBundleId === b.id ? 'Reloading…' : 'Reload'}
                                                    </Button>
                                                    <Button variant="secondary" onClick={function () { return editBundle(b); }} disabled={isAdminBundle} title={isAdminBundle ? 'Admin bundle is protected' : undefined}>
                                                        Edit
                                                    </Button>
                                                    <Button variant="danger" onClick={function () { return deleteBundle(b.id); }} disabled={isAdminBundle} title={isAdminBundle ? 'Admin bundle is protected' : undefined}>
                                                        Delete
                                                    </Button>
                                                </div>
                                            </td>
                                        </tr>);
        })}
                                    {bundleList.length === 0 && (<tr>
                                            <td colSpan={9} className="px-4 py-6 text-center text-gray-500">
                                                No bundles configured.
                                            </td>
                                        </tr>)}
                                </tbody>
                            </table>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title={<div className="flex items-center gap-3">
                                <span>Bundle props</span>
                                {bundleVersion ? (<div className="flex items-center gap-2">
                                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-900 text-white">
                                            v{bundleVersion}
                                        </span>
                                        <Button variant="secondary" onClick={function () { return copyText(bundleVersion); }}>Copy</Button>
                                    </div>) : null}
                            </div>} subtitle={"Override bundle props per tenant/project. ".concat(reloadAuthorityLabel, " re-applies props from that source; reset from code restores bundle code defaults only.")} action={<div className="flex gap-2">
                                <Button variant="secondary" onClick={loadProps} disabled={!propsBundleId || propsLoading}>
                                    {propsLoading ? 'Loading…' : 'Refresh'}
                                </Button>
                                <Button variant="secondary" onClick={resetPropsFromCode} disabled={!propsBundleId}>
                                    Reset from code
                                </Button>
                            </div>}/>
                    <CardBody className="space-y-5">
                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">Bundle ID</label>
                            <select className="w-full px-4 py-2.5 border border-gray-200/80 rounded-xl bg-white text-sm" value={propsBundleId} onChange={function (e) { return setPropsBundleId(e.target.value); }}>
                                <option value="">—</option>
                                {bundleList.map(function (b) { return (<option key={b.id} value={b.id}>{b.id}</option>); })}
                            </select>
                        </div>

                        <div className="text-xs text-gray-600">
                            Props resolution order: <strong>code defaults → {propsResolutionLabel} → runtime overrides</strong>.
                            The editor shows the full effective props; <strong>Save props</strong> stores exactly what you see.
                            Use dot-path updates for precise changes. <strong>{reloadAuthorityLabel}</strong> rebuilds this Redis props layer from the
                            current source, removes keys no longer present there, and discards runtime overrides.
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField label="Dot-path (props)" value={propsKeyPath} onChange={function (v) { return setPropsKeyPath(v); }} placeholder="role_models.solver.react.v2.decision.v2.strong.model"/>
                            <InputField label="Value (JSON or string)" value={propsValue} onChange={function (v) { return setPropsValue(v); }} placeholder={'"claude-sonnet-4-6"'}/>
                        </div>

                        <div className="flex flex-wrap gap-3">
                            <Button variant="secondary" onClick={function () { return applyPropsDotPath('set'); }}>
                                Apply dot-path to editor
                            </Button>
                            <Button variant="secondary" onClick={function () { return applyPropsDotPath('delete'); }}>
                                Remove key from editor
                            </Button>
                        </div>

                        {bundleSnapshotPath ? (<div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
                                <span className="font-semibold">Snapshot path:</span>
                                <code className="px-2 py-1 rounded bg-gray-100 border border-gray-200">{bundleSnapshotPath}</code>
                                <Button variant="secondary" onClick={function () { return copyText(bundleSnapshotPath); }}>Copy path</Button>
                            </div>) : null}

                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">Props JSON</label>
                            <textarea className="w-full min-h-[220px] px-4 py-3 border border-gray-200/80 rounded-xl bg-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-gray-900/10" value={propsJson} onChange={function (e) { return setPropsJson(e.target.value); }} placeholder={"{\n  \"key\": \"value\"\n}"}/>
                        </div>

                        <div className="flex flex-wrap gap-3">
                            <Button variant="primary" onClick={function () { return saveProps('replace'); }}>Save props</Button>
                            <Button variant="secondary" onClick={loadProps} disabled={!propsBundleId || propsLoading}>
                                {propsLoading ? 'Loading…' : 'Reset editor'}
                            </Button>
                        </div>
                        <div className="text-xs text-gray-500">
                            The JSON editor shows the <strong>full effective props</strong> (defaults + overrides).<br />
                            <strong>Save props</strong> stores exactly what you see in the editor.
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">Code defaults (read-only)</label>
                            <textarea className="w-full min-h-[180px] px-4 py-3 border border-gray-200/70 rounded-xl bg-gray-50 text-sm font-mono text-gray-600" value={propsDefaultsJson} readOnly/>
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title="Bundle secrets" subtitle="Write-only secrets for bundles. Use dot-path for single keys or JSON for bulk updates."/>
                    <CardBody className="space-y-5">
                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">Bundle ID</label>
                            <select className="w-full px-4 py-2.5 border border-gray-200/80 rounded-xl bg-white text-sm" value={secretsBundleId} onChange={function (e) { return setSecretsBundleId(e.target.value); }}>
                                <option value="">—</option>
                                {bundleList.map(function (b) { return (<option key={b.id} value={b.id}>{b.id}</option>); })}
                            </select>
                        </div>

                        <div className="text-xs text-gray-600">
                            {secretsLoading ? 'Loading keys…' : (<>
                                    Known keys:{' '}
                                    <code>{(secretsKeys || []).join(', ') || 'none'}</code>
                                </>)}
                        </div>

                        <div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <InputField label="Dot-path (secrets)" value={secretsKeyPath} onChange={function (v) { return setSecretsKeyPath(v); }} placeholder="openai.api_key"/>
                            <InputField label="Value (JSON or string)" value={secretsValue} onChange={function (v) { return setSecretsValue(v); }} placeholder={'"sk-..."'}/>
                            </div>
                            <div className="mt-3 flex flex-wrap gap-3">
                                <Button variant="primary" onClick={function () { return submitSecretDotPath('set'); }} disabled={secretsSaving}>
                                    {secretsSaving ? 'Saving…' : 'Set key'}
                                </Button>
                                <Button variant="secondary" onClick={function () { return submitSecretDotPath('clear'); }} disabled={secretsSaving}>
                                    Clear key
                                </Button>
                            </div>
                            <div className="mt-2 text-xs text-gray-500">
                                Dot-path writes a single key. Values accept JSON (objects/arrays) or raw strings.
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-800 mb-2">Bulk secrets JSON (optional)</label>
                            <textarea className="w-full min-h-[180px] px-4 py-3 border border-gray-200/80 rounded-xl bg-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-gray-900/10" value={secretsJson} onChange={function (e) { return setSecretsJson(e.target.value); }} placeholder={"{\n  \"openai\": { \"api_key\": \"...\" },\n  \"stripe\": { \"secret_key\": \"...\" }\n}"}/>
                        </div>

                        <div className="flex flex-wrap gap-3">
                            <Button variant="primary" onClick={saveSecrets} disabled={secretsSaving}>
                                {secretsSaving ? 'Saving…' : 'Set secrets (JSON)'}
                            </Button>
                            <Button variant="secondary" onClick={clearSecrets} disabled={secretsSaving}>
                                Clear keys (JSON)
                            </Button>
                        </div>
                        {secretsStatus ? (<div className="text-xs text-gray-600">
                                {secretsStatus.mode === 'set' ? 'Saved' : 'Cleared'} keys:{' '}
                                <code>{(secretsStatus.keys || []).join(', ') || 'none'}</code>
                            </div>) : null}
                        <div className="text-xs text-gray-500">
                            Secrets are stored under <code>bundles.&lt;bundle_id&gt;.secrets.*</code> and are write-only.
                        </div>
                    </CardBody>
                </Card>

                <Card>
                    <CardHeader title={editingId ? "Edit bundle: ".concat(editingId) : 'Add bundle'} subtitle="Provide id and either path or repo; module is optional unless using zip/whl." action={editingId ? <Button variant="secondary" onClick={resetForm}>Cancel edit</Button> : undefined}/>
                    <CardBody className="space-y-5">
                        <div ref={formRef}/>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField label="Bundle ID" value={form.id} onChange={function (v) { return setForm(__assign(__assign({}, form), { id: v })); }} placeholder="demo.react@1.0.0"/>
                            <InputField label="Name" value={form.name || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { name: v })); }} placeholder="Demo bundle"/>
                            <InputField label="Path" value={form.path} onChange={function (v) { return setForm(__assign(__assign({}, form), { path: v })); }} placeholder="/bundles"/>
                            <InputField label="Module" value={form.module || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { module: v })); }} placeholder="demo.react@1.0.0.entrypoint"/>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField label="Repo" value={form.repo || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { repo: v })); }} placeholder="git@github.com:org/repo.git"/>
                            <InputField label="Ref" value={form.ref || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { ref: v })); }} placeholder="main | v1.2.3 | <commit>"/>
                            <InputField label="Subdir" value={form.subdir || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { subdir: v })); }} placeholder="path/to/bundles"/>
                        </div>
                        <div className="rounded-xl border border-slate-200/70 bg-slate-50 px-4 py-3 text-xs text-slate-700">
                            <div className="font-semibold mb-1">Resolved path preview</div>
                            <div className="space-y-1">
                                <div>
                                    <span className="font-medium">HOST_BUNDLES_PATH:</span>{' '}
                                    <code className="px-1 py-0.5 rounded bg-white border border-slate-200">
                                        {settings.getHostBundlesPath() || '—'}
                                    </code>
                                </div>
                                <div>
                                    <span className="font-medium">AGENTIC_BUNDLES_ROOT:</span>{' '}
                                    <code className="px-1 py-0.5 rounded bg-white border border-slate-200">
                                        {settings.getAgenticBundlesRoot() || '—'}
                                    </code>
                                </div>
                                <div>
                                    <span className="font-medium">Current path:</span>{' '}
                                    <code className="px-1 py-0.5 rounded bg-white border border-slate-200">{form.path || '—'}</code>
                                </div>
                                {derivedGitPath ? (<div>
                                        <span className="font-medium">Derived path (repo/ref template):</span>{' '}
                                        <code className="px-1 py-0.5 rounded bg-white border border-slate-200">{derivedGitPath}</code>
                                    </div>) : null}
                                {derivedHostPath ? (<div>
                                        <span className="font-medium">Derived path (HOST_BUNDLES_PATH):</span>{' '}
                                        <code className="px-1 py-0.5 rounded bg-white border border-slate-200">{derivedHostPath}</code>
                                    </div>) : null}
                                {derivedAgenticPath ? (<div>
                                        <span className="font-medium">Derived path (AGENTIC_BUNDLES_ROOT):</span>{' '}
                                        <code className="px-1 py-0.5 rounded bg-white border border-slate-200">{derivedAgenticPath}</code>
                                    </div>) : null}
                            </div>
                            <div className="mt-2 text-[11px] text-slate-600">
                                Updates take effect when the bundle path changes. For repo bundles, use a new <code>ref</code>.
                                For local bundles, deploy to a new path and update <code>path</code>.
                            </div>
                        </div>
                        <div className="rounded-xl border border-amber-200/60 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                            <div className="font-semibold mb-1">Private Git repos</div>
                            <div>Set one of:</div>
                            <ul className="list-disc pl-5 space-y-1">
                                <li><code>GIT_SSH_KEY_PATH</code> (+ optional <code>GIT_SSH_KNOWN_HOSTS</code>, <code>GIT_SSH_STRICT_HOST_KEY_CHECKING</code>)</li>
                                <li>or embed a token in the URL: <code>https://&lt;token&gt;@github.com/org/repo.git</code></li>
                            </ul>
                        </div>
                        <InputField label="Description" value={form.description || ''} onChange={function (v) { return setForm(__assign(__assign({}, form), { description: v })); }} placeholder="Optional description"/>

                        <div className="flex items-center gap-2">
                            <input type="checkbox" checked={!!form.singleton} onChange={function (e) { return setForm(__assign(__assign({}, form), { singleton: e.target.checked })); }} className="h-4 w-4"/>
                            <span className="text-sm text-gray-700">Singleton (reuse workflow instance)</span>
                        </div>

                        <div className="flex gap-3">
                            <Button variant="primary" onClick={saveBundle}>
                                {editingId ? 'Save changes' : 'Add bundle'}
                            </Button>
                            <Button variant="secondary" onClick={resetForm}>Clear</Button>
                        </div>
                    </CardBody>
                </Card>
            </div>
        </div>);
};
var rootEl = document.getElementById('root');
if (rootEl) {
    var root = client_1.default.createRoot(rootEl);
    root.render(<AIBundleDashboard />);
}
