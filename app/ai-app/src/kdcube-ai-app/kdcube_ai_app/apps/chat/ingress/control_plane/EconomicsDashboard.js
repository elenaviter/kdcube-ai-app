"use strict";
// Economics Admin React App (TypeScript)
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
// Accounting identities that are platform/app principals, not people. These are
// the literal user_id values as they appear in RECORDED events (e.g. scheduled
// app runs historically account with user_id="bundle"), so the set must match
// the data on disk — extend it if an emitter starts writing a new placeholder.
// In the user grouping such rows are listed separately so real users are never
// mixed with app-run identities; run-as attribution removes them going forward.
var SYSTEM_PRINCIPAL_IDS = new Set(['bundle', 'system', 'anonymous']);
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
        this.PLACEHOLDER_STRIPE_DASHBOARD = '{{' + 'STRIPE_DASHBOARD_BASE_URL' + '}}';
        this.settings = {
            baseUrl: '{{CHAT_BASE_URL}}',
            accessToken: '{{ACCESS_TOKEN}}',
            idToken: '{{ID_TOKEN}}',
            idTokenHeader: '{{ID_TOKEN_HEADER}}',
            defaultTenant: '{{DEFAULT_TENANT}}',
            defaultProject: '{{DEFAULT_PROJECT}}',
            defaultAppBundleId: '{{DEFAULT_APP_BUNDLE_ID}}',
            stripeDashboardBaseUrl: '{{STRIPE_DASHBOARD_BASE_URL}}'
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
                console.warn('[SettingsManager] Invalid baseUrl detected, using fallback');
                return window.location.origin;
            }
            var trimmed = this.settings.baseUrl.replace(/\/+$/, '');
            return trimmed.endsWith('/api') ? trimmed.slice(0, -4) : trimmed;
        }
        catch (e) {
            console.warn('[SettingsManager] Invalid baseUrl, using fallback:', this.settings.baseUrl);
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
    SettingsManager.prototype.getDefaultAppBundleId = function () {
        return this.settings.defaultAppBundleId === this.PLACEHOLDER_BUNDLE_ID
            ? 'kdcube.codegen.orchestrator'
            : this.settings.defaultAppBundleId;
    };
    SettingsManager.prototype.getStripeDashboardBaseUrl = function () {
        if (!this.settings.stripeDashboardBaseUrl || this.settings.stripeDashboardBaseUrl === this.PLACEHOLDER_STRIPE_DASHBOARD) {
            return 'https://dashboard.stripe.com';
        }
        return this.settings.stripeDashboardBaseUrl;
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
        if (config.stripeDashboardBaseUrl)
            updates.stripeDashboardBaseUrl = config.stripeDashboardBaseUrl;
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
        console.log('[SettingsManager] Setting up parent listener');
        var identity = "CONTROL_PLANE_ADMIN";
        window.addEventListener('message', function (event) {
            if (event.data.type === 'CONN_RESPONSE' || event.data.type === 'CONFIG_RESPONSE') {
                var requestedIdentity = event.data.identity;
                if (requestedIdentity !== identity) {
                    console.warn("[SettingsManager] Ignoring response for identity ".concat(requestedIdentity));
                    return;
                }
                console.log('[SettingsManager] Received config from parent', event.data.config);
                if (event.data.config && _this.applyRuntimeConfig(event.data.config)) {
                    console.log('[SettingsManager] Settings updated from parent');
                }
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
                    console.log('[SettingsManager] Requesting config from parent');
                    window.parent.postMessage({
                        type: 'CONFIG_REQUEST',
                        data: {
                            requestedFields: [
                                'baseUrl', 'accessToken', 'idToken', 'idTokenHeader',
                                'defaultTenant', 'defaultProject', 'defaultAppBundleId', 'stripeDashboardBaseUrl'
                            ],
                            identity: identity
                        }
                    }, '*');
                    var timeout = window.setTimeout(function () {
                        console.log('[SettingsManager] Config request timeout');
                        finish(false);
                    }, 3000);
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
        else {
            console.log('[SettingsManager] Using existing settings');
            return Promise.resolve(!this.hasPlaceholderSettings());
        }
    };
    return SettingsManager;
}());
var settings = new SettingsManager();
// =============================================================================
// Auth Header Helper
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
// =============================================================================
// Economics API Client
// =============================================================================
var EconomicsAPI = /** @class */ (function () {
    function EconomicsAPI(basePath) {
        if (basePath === void 0) { basePath = '/api/admin/control-plane'; }
        this.basePath = basePath;
    }
    EconomicsAPI.prototype.getFullUrl = function (path) {
        var baseUrl = settings.getBaseUrl();
        return "".concat(baseUrl).concat(this.basePath).concat(path);
    };
    EconomicsAPI.prototype.getStripeUrl = function (path) {
        var baseUrl = settings.getBaseUrl();
        return "".concat(baseUrl, "/api/economics").concat(path);
    };
    EconomicsAPI.prototype.fetchWithAuth = function (url_1) {
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
    EconomicsAPI.prototype.grantTrial = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/plan-override/grant-trial'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: payload.userId,
                                days: payload.days,
                                requests_per_day: payload.requestsPerDay,
                                tokens_per_hour: payload.tokensPerHour,
                                tokens_per_day: payload.tokensPerDay,
                                tokens_per_month: payload.tokensPerMonth,
                                usd_per_hour: payload.usdPerHour,
                                usd_per_day: payload.usdPerDay,
                                usd_per_month: payload.usdPerMonth,
                                notes: payload.notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.updatePlanOverride = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/plan-override/update'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: payload.userId,
                                requests_per_day: payload.requestsPerDay,
                                requests_per_month: payload.requestsPerMonth,
                                tokens_per_hour: payload.tokensPerHour,
                                tokens_per_day: payload.tokensPerDay,
                                tokens_per_month: payload.tokensPerMonth,
                                usd_per_hour: payload.usdPerHour,
                                usd_per_day: payload.usdPerDay,
                                usd_per_month: payload.usdPerMonth,
                                max_concurrent: payload.maxConcurrent,
                                expires_in_days: payload.expiresInDays,
                                notes: payload.notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getPlanBalance = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, includeExpired) {
            var queryParams, response;
            if (includeExpired === void 0) { includeExpired = false; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        queryParams = new URLSearchParams({
                            include_expired: includeExpired.toString()
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl("/plan-override/user/".concat(userId)), "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.deactivatePlanBalance = function (userId) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl("/plan-override/user/".concat(userId)), { method: 'DELETE' })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.addLifetimeCredits = function (userId, usdAmount, notes) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/plan-override/add-lifetime-credits'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: userId,
                                usd_amount: usdAmount,
                                notes: notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getLifetimeBalance = function (userId) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl("/plan-override/lifetime-balance/".concat(userId)))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listQuotaPolicies = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/policies/quota'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getReservation = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/economics/reservation'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.setReservation = function (floor, amount) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/economics/reservation'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ floor: floor, amount: amount }),
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.deleteReservation = function (floor) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl("/economics/reservation/".concat(encodeURIComponent(floor))), { method: 'DELETE' })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.setQuotaPolicy = function (policy) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/policies/quota'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                plan_id: policy.planId,
                                max_concurrent: policy.maxConcurrent,
                                requests_per_day: policy.requestsPerDay,
                                requests_per_month: policy.requestsPerMonth,
                                total_requests: policy.totalRequests,
                                tokens_per_hour: policy.tokensPerHour,
                                tokens_per_day: policy.tokensPerDay,
                                tokens_per_month: policy.tokensPerMonth,
                                usd_per_hour: policy.usdPerHour,
                                usd_per_day: policy.usdPerDay,
                                usd_per_month: policy.usdPerMonth,
                                notes: policy.notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listBudgetPolicies = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/policies/budget'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.setBudgetPolicy = function (policy) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/policies/budget'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                provider: policy.provider,
                                usd_per_hour: policy.usdPerHour,
                                usd_per_day: policy.usdPerDay,
                                usd_per_month: policy.usdPerMonth,
                                notes: policy.notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    // async getUserQuotaBreakdown(userId: string, userType: string): Promise<{ status: string; } & QuotaBreakdown> {
    //     const queryParams = new URLSearchParams({
    //         user_type: userType
    //     });
    //     const response = await this.fetchWithAuth(
    //         `${this.getFullUrl(`/users/${userId}/quota-breakdown`)}?${queryParams}`
    //     );
    //     return response.json();
    // }
    EconomicsAPI.prototype.getUserBudgetBreakdown = function (userId, planId, bundleId) {
        return __awaiter(this, void 0, void 0, function () {
            var queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        queryParams = new URLSearchParams({
                            include_expired_override: 'true',
                            reservations_limit: '50',
                        });
                        if (planId) {
                            queryParams.set('plan_id', planId);
                        }
                        if (bundleId) {
                            queryParams.set('bundle_id', bundleId);
                        }
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl("/users/".concat(userId, "/budget-breakdown")), "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getAppBudgetBalance = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/app-budget/balance'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getAppBudgetAbsorptionReport = function (period, days, groupBy) {
        return __awaiter(this, void 0, void 0, function () {
            var queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        queryParams = new URLSearchParams({
                            period: period,
                            days: days.toString(),
                            group_by: groupBy,
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl('/app-budget/absorption-report'), "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getAppBudgetAbsorptionReportCsv = function (period, days, groupBy) {
        return __awaiter(this, void 0, void 0, function () {
            var queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        queryParams = new URLSearchParams({
                            period: period,
                            days: days.toString(),
                            group_by: groupBy,
                            format: 'csv',
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl('/app-budget/absorption-report'), "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.text()];
                }
            });
        });
    };
    // True spend from the OPEX aggregates, priced live from the descriptor
    // price table, grouped by user, agent, or app. Actual per-model dollars —
    // a different number than the quota-equivalent view in User Budget
    // Breakdown and the absorbed shortfall in the absorption report, by design.
    EconomicsAPI.prototype.getOpexCost = function (dimension, dateFrom, dateTo) {
        return __awaiter(this, void 0, void 0, function () {
            var baseUrl, path, queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        baseUrl = settings.getBaseUrl();
                        path = dimension === 'agent' ? 'agents' : dimension === 'app' ? 'apps' : 'users';
                        queryParams = new URLSearchParams({
                            tenant: settings.getDefaultTenant(),
                            project: settings.getDefaultProject(),
                            date_from: dateFrom,
                            date_to: dateTo,
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(baseUrl, "/api/opex/").concat(path, "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    // Backfill daily + monthly aggregates for a window (adds newly introduced
    // dimension files — e.g. per-app — to already-aggregated days). Safe to
    // re-run; shares the scheduler's per-date Redis lock.
    EconomicsAPI.prototype.runAggregationRange = function (startDate, endDate, includeToday) {
        return __awaiter(this, void 0, void 0, function () {
            var baseUrl, queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        baseUrl = settings.getBaseUrl();
                        queryParams = new URLSearchParams({
                            start_date: startDate,
                            end_date: endDate,
                            include_today: String(includeToday),
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(baseUrl, "/api/opex/admin/run-aggregation-range?").concat(queryParams), { method: 'POST' })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getRequestLineage = function (requestId) {
        return __awaiter(this, void 0, void 0, function () {
            var queryParams, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        queryParams = new URLSearchParams({
                            request_id: requestId,
                        });
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl('/economics/request-lineage'), "?").concat(queryParams))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getEconomicsReference = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/economics/reference'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.topupAppBudget = function (usdAmount, notes) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/app-budget/topup'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                usd_amount: usdAmount,
                                notes: notes
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.healthCheck = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/health'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.createSubscription = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            var _a, _b, _c;
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeUrl('/admin/subscriptions/create'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: payload.userId,
                                plan_id: payload.planId,
                                provider: payload.provider,
                                stripe_price_id: (_a = payload.stripePriceId) !== null && _a !== void 0 ? _a : null,
                                stripe_customer_id: (_b = payload.stripeCustomerId) !== null && _b !== void 0 ? _b : null,
                                monthly_price_cents_hint: (_c = payload.monthlyPriceCentsHint) !== null && _c !== void 0 ? _c : null,
                            })
                        })];
                    case 1:
                        response = _d.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listSubscriptionPlans = function (params) {
        return __awaiter(this, void 0, void 0, function () {
            var qp, response;
            var _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        qp = new URLSearchParams();
                        if (params === null || params === void 0 ? void 0 : params.provider)
                            qp.set('provider', params.provider);
                        if ((params === null || params === void 0 ? void 0 : params.activeOnly) != null)
                            qp.set('active_only', String(params.activeOnly));
                        qp.set('limit', String((_a = params === null || params === void 0 ? void 0 : params.limit) !== null && _a !== void 0 ? _a : 200));
                        qp.set('offset', String((_b = params === null || params === void 0 ? void 0 : params.offset) !== null && _b !== void 0 ? _b : 0));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl('/subscriptions/plans'), "?").concat(qp.toString()))];
                    case 1:
                        response = _c.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.upsertSubscriptionPlan = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            var _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/subscriptions/plans'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                plan_id: payload.planId,
                                provider: payload.provider,
                                stripe_price_id: (_a = payload.stripePriceId) !== null && _a !== void 0 ? _a : null,
                                monthly_price_cents: payload.monthlyPriceCents,
                                active: payload.active,
                                notes: (_b = payload.notes) !== null && _b !== void 0 ? _b : null,
                            })
                        })];
                    case 1:
                        response = _c.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.getSubscription = function (userId) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl("/subscriptions/user/".concat(userId)))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listSubscriptions = function (params) {
        return __awaiter(this, void 0, void 0, function () {
            var qp, response;
            var _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        qp = new URLSearchParams();
                        if (params === null || params === void 0 ? void 0 : params.provider)
                            qp.set('provider', params.provider);
                        if (params === null || params === void 0 ? void 0 : params.userId)
                            qp.set('user_id', params.userId);
                        qp.set('limit', String((_a = params === null || params === void 0 ? void 0 : params.limit) !== null && _a !== void 0 ? _a : 50));
                        qp.set('offset', String((_b = params === null || params === void 0 ? void 0 : params.offset) !== null && _b !== void 0 ? _b : 0));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl('/subscriptions/list'), "?").concat(qp.toString()))];
                    case 1:
                        response = _c.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listSubscriptionPeriods = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, status, limit, offset) {
            var qp, response;
            if (status === void 0) { status = 'closed'; }
            if (limit === void 0) { limit = 50; }
            if (offset === void 0) { offset = 0; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        qp = new URLSearchParams();
                        qp.set('status', status);
                        qp.set('limit', String(limit));
                        qp.set('offset', String(offset));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl("/subscriptions/periods/".concat(userId)), "?").concat(qp.toString()))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listSubscriptionLedger = function (userId_1, periodKey_1) {
        return __awaiter(this, arguments, void 0, function (userId, periodKey, limit, offset) {
            var qp, response;
            if (limit === void 0) { limit = 200; }
            if (offset === void 0) { offset = 0; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        qp = new URLSearchParams();
                        qp.set('period_key', periodKey);
                        qp.set('limit', String(limit));
                        qp.set('offset', String(offset));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getFullUrl("/subscriptions/ledger/".concat(userId)), "?").concat(qp.toString()))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.resetInternalQuota = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/subscriptions/internal/reset-quota'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ user_id: payload.userId }),
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.topupSubscriptionBudget = function (userId_1, usdAmount_1, notes_1) {
        return __awaiter(this, arguments, void 0, function (userId, usdAmount, notes, forceTopup) {
            var response;
            if (forceTopup === void 0) { forceTopup = false; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/subscriptions/budget/topup'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: userId,
                                usd_amount: usdAmount,
                                notes: notes,
                                force_topup: forceTopup
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.sweepSubscriptionRollovers = function (userId) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/subscriptions/rollover/sweep'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: userId || null,
                                limit: 200
                            })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.reapSubscriptionReservationsAll = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            var _a, _b, _c;
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getFullUrl('/subscriptions/reservations/reap-all'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: (_a = payload.userId) !== null && _a !== void 0 ? _a : null,
                                limit_periods: (_b = payload.limitPeriods) !== null && _b !== void 0 ? _b : 500,
                                per_period_limit: (_c = payload.perPeriodLimit) !== null && _c !== void 0 ? _c : 500,
                            })
                        })];
                    case 1:
                        response = _d.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.refundWallet = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeUrl('/admin/stripe/wallet/refund'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: payload.userId,
                                payment_intent_id: payload.paymentIntentId,
                                usd_amount: (_a = payload.usdAmount) !== null && _a !== void 0 ? _a : null,
                                notes: payload.notes
                            })
                        })];
                    case 1:
                        response = _b.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.cancelSubscription = function (payload) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            var _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeUrl('/admin/subscriptions/cancel'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: (_a = payload.userId) !== null && _a !== void 0 ? _a : null,
                                stripe_subscription_id: (_b = payload.stripeSubscriptionId) !== null && _b !== void 0 ? _b : null,
                                notes: payload.notes
                            })
                        })];
                    case 1:
                        response = _c.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.reconcileStripe = function () {
        return __awaiter(this, arguments, void 0, function (kind) {
            var response;
            if (kind === void 0) { kind = 'all'; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeUrl('/admin/stripe/reconcile'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ kind: kind, limit: 200 })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listPendingStripeRequests = function () {
        return __awaiter(this, arguments, void 0, function (kind, limit, offset) {
            var qp, response;
            if (kind === void 0) { kind = 'all'; }
            if (limit === void 0) { limit = 200; }
            if (offset === void 0) { offset = 0; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        qp = new URLSearchParams();
                        qp.set('kind', kind);
                        qp.set('limit', String(limit));
                        qp.set('offset', String(offset));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getStripeUrl('/admin/stripe/pending'), "?").concat(qp.toString()))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    EconomicsAPI.prototype.listPendingEconomicsEvents = function (kind_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (kind, userId, limit, offset) {
            var qp, response;
            if (limit === void 0) { limit = 200; }
            if (offset === void 0) { offset = 0; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        qp = new URLSearchParams();
                        if (kind)
                            qp.set('kind', kind);
                        if (userId)
                            qp.set('user_id', userId);
                        qp.set('limit', String(limit));
                        qp.set('offset', String(offset));
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getStripeUrl('/admin/stripe/pending'), "?").concat(qp.toString()))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    return EconomicsAPI;
}());
// =============================================================================
// UI Components (gentle styling)
// =============================================================================
var Card = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"rounded-xl border border-[#E6F1F0] bg-white shadow-[0_1px_2px_rgba(13,30,44,0.04)] ".concat(className)}>
        {children}
    </div>);
};
var CardHeader = function (_a) {
    var title = _a.title, subtitle = _a.subtitle, action = _a.action;
    return (<div className="shrink-0 border-b border-[#E6F1F0] px-3.5 py-2">
        <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
                <h2 className="text-[12.5px] font-semibold leading-5 text-[#10304B]">{title}</h2>
                {subtitle && <p className="truncate text-[11.5px] leading-4 text-[#7A99B0]">{subtitle}</p>}
            </div>
            {action && <div className="shrink-0">{action}</div>}
        </div>
    </div>);
};
var CardBody = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"px-3.5 py-3 ".concat(className)}>
        {children}
    </div>);
};
var Callout = function (_a) {
    var _b = _a.tone, tone = _b === void 0 ? 'neutral' : _b, title = _a.title, children = _a.children;
    var tones = {
        neutral: 'bg-[#F6FAFA] border-[#E6F1F0] text-[#3A5672]',
        info: 'bg-[rgba(67,114,195,0.08)] border-[rgba(67,114,195,0.35)] text-[#2B4B8A]',
        warning: 'bg-[rgba(245,158,11,0.1)] border-[rgba(245,158,11,0.4)] text-[#B45309]',
        success: 'bg-[rgba(34,197,94,0.08)] border-[rgba(34,197,94,0.35)] text-[#15803D]',
    };
    return (<div className={"rounded-lg border px-2.5 py-1.5 text-[11.5px] leading-snug ".concat(tones[tone])}>
            {title && <span className="font-semibold">{title} — </span>}
            {children}
        </div>);
};
var Button = function (_a) {
    var children = _a.children, onClick = _a.onClick, _b = _a.type, type = _b === void 0 ? 'button' : _b, _c = _a.variant, variant = _c === void 0 ? 'primary' : _c, _d = _a.disabled, disabled = _d === void 0 ? false : _d, _e = _a.className, className = _e === void 0 ? '' : _e;
    var variants = {
        primary: 'bg-[#4372C3] hover:bg-[#2B4B8A] text-white',
        secondary: 'bg-white hover:bg-[#F6FAFA] text-[#3A5672] border border-[#D8ECEB]',
        danger: 'bg-white hover:bg-[rgba(248,113,113,0.08)] text-[#B91C1C] border border-[#D8ECEB]',
    };
    return (<button type={type} onClick={onClick} disabled={disabled} className={"inline-flex h-8 items-center justify-center whitespace-nowrap rounded-lg px-3 text-[12.5px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ".concat(variants[variant], " ").concat(className)}>
            {children}
        </button>);
};
var Input = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, _b = _a.type, type = _b === void 0 ? 'text' : _b, placeholder = _a.placeholder, required = _a.required, min = _a.min, max = _a.max, step = _a.step, list = _a.list, _c = _a.className, className = _c === void 0 ? '' : _c;
    return (<div className={className}>
        {label && <label className="mb-1 block truncate text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">{label}</label>}
        <input type={type} value={value} onChange={onChange} placeholder={placeholder} required={required} min={min} max={max} step={step} list={list} className="h-8 w-full rounded-md border border-[#D8ECEB] bg-white px-2.5 text-[12.5px]
                 focus:ring-2 focus:ring-[#01BEB2]/30 focus:border-[#01BEB2] transition-colors
                 placeholder:text-[#7A99B0]"/>
    </div>);
};
var Select = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, options = _a.options, children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={className}>
        {label && <label className="mb-1 block truncate text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">{label}</label>}
        <select value={value} onChange={onChange} className="h-8 w-full rounded-md border border-[#D8ECEB] bg-white px-2 text-[12.5px]
                 focus:ring-2 focus:ring-[#01BEB2]/30 focus:border-[#01BEB2] transition-colors">
            {options ? options.map(function (o) { return <option key={o.value} value={o.value}>{o.label}</option>; }) : children}
        </select>
    </div>);
};
var TextArea = function (_a) {
    var label = _a.label, value = _a.value, onChange = _a.onChange, placeholder = _a.placeholder, _b = _a.rows, rows = _b === void 0 ? 2 : _b, _c = _a.className, className = _c === void 0 ? '' : _c;
    return (<div className={className}>
        {label && <label className="mb-1 block truncate text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">{label}</label>}
        <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} className="w-full rounded-md border border-[#D8ECEB] bg-white px-2.5 py-1.5 text-[12.5px]
                 focus:ring-2 focus:ring-[#01BEB2]/30 focus:border-[#01BEB2] transition-colors
                 placeholder:text-[#7A99B0]"/>
    </div>);
};
var StatCard = function (_a) {
    var label = _a.label, value = _a.value, hint = _a.hint;
    return (<div className="rounded-xl border border-[#E6F1F0] bg-white px-3 py-2 shadow-[0_1px_2px_rgba(13,30,44,0.04)]">
        <p className="truncate text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">{label}</p>
        <p className="mt-0.5 truncate font-mono text-[15px] font-semibold text-[#0D1E2C]">{value}</p>
        {hint && <p className="mt-0.5 truncate text-[11px] text-[#7A99B0]">{hint}</p>}
    </div>);
};
var LoadingSpinner = function () { return (<div className="flex items-center justify-center py-4">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#E6F1F0] border-t-[#01BEB2]"></div>
    </div>); };
var EmptyState = function (_a) {
    var message = _a.message, _b = _a.icon, icon = _b === void 0 ? '📭' : _b;
    return (<div className="py-4 text-center">
        <div className="mb-1 text-xl">{icon}</div>
        <p className="text-[12px] text-[#3A5672]">{message}</p>
    </div>);
};
// =============================================================================
// Subscription display helpers
// =============================================================================
var PLAN_OPTIONS = [
    { value: 'free', label: 'free' },
    { value: 'wallet', label: 'wallet' },
    { value: 'admin', label: 'admin' },
    { value: 'anonymous', label: 'anonymous' },
    { value: 'custom', label: 'custom…' },
];
var PROVIDER_LABEL = {
    internal: 'Manual',
    stripe: 'Stripe',
};
function providerLabel(provider) {
    var _a;
    if (!provider)
        return '—';
    return (_a = PROVIDER_LABEL[provider]) !== null && _a !== void 0 ? _a : provider;
}
function formatDateTime(iso) {
    if (!iso)
        return '—';
    var d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}
function stripeUrl(path) {
    var base = settings.getStripeDashboardBaseUrl().replace(/\/$/, '');
    var clean = path.replace(/^\//, '');
    return "".concat(base, "/").concat(clean);
}
function stripeLinkForPending(item) {
    var md = item.metadata || {};
    var refundId = md.stripe_refund_id;
    var subId = md.stripe_subscription_id;
    var piId = md.payment_intent_id;
    if (refundId)
        return { id: String(refundId), url: stripeUrl("refunds/".concat(refundId)) };
    if (subId)
        return { id: String(subId), url: stripeUrl("subscriptions/".concat(subId)) };
    if (piId)
        return { id: String(piId), url: stripeUrl("payments/".concat(piId)) };
    return null;
}
function getDueState(sub, now) {
    if (now === void 0) { now = new Date(); }
    if (sub.status !== 'active')
        return { state: 'inactive', label: 'Inactive' };
    // If there's no next_charge_at, it's simply not scheduled (free/admin, or legacy)
    if (!sub.next_charge_at)
        return { state: 'not_scheduled', label: 'Not scheduled' };
    var due = new Date(sub.next_charge_at);
    if (Number.isNaN(due.getTime()))
        return { state: 'not_scheduled', label: 'Not scheduled' };
    var ms = due.getTime() - now.getTime();
    if (ms <= 0)
        return { state: 'overdue', label: 'Overdue' };
    var days = ms / (1000 * 60 * 60 * 24);
    if (days <= 7)
        return { state: 'due_soon', label: 'Due soon' };
    return { state: 'scheduled', label: 'Scheduled' };
}
var Pill = function (_a) {
    var _b = _a.tone, tone = _b === void 0 ? 'neutral' : _b, children = _a.children;
    var tones = {
        neutral: 'bg-[#F6FAFA] text-[#3A5672] border-[#E6F1F0]',
        success: 'bg-[rgba(34,197,94,0.08)] text-[#15803D] border-[rgba(34,197,94,0.35)]',
        warning: 'bg-[rgba(245,158,11,0.1)] text-[#B45309] border-[rgba(245,158,11,0.4)]',
        danger: 'bg-[rgba(248,113,113,0.1)] text-[#B91C1C] border-[rgba(248,113,113,0.4)]',
    };
    return (<span className={"inline-flex items-center px-1.5 py-px rounded-md text-[10.5px] font-bold uppercase tracking-wide border ".concat(tones[tone])}>
      {children}
    </span>);
};
function DuePill(_a) {
    var sub = _a.sub;
    var due = getDueState(sub);
    var tone = due.state === 'overdue' ? 'danger' :
        due.state === 'due_soon' ? 'warning' :
            due.state === 'scheduled' ? 'neutral' :
                due.state === 'inactive' ? 'neutral' :
                    'neutral';
    return <Pill tone={tone}>{due.label}</Pill>;
}
function formatCount(value) {
    if (value == null)
        return '∞';
    return Number(value).toLocaleString();
}
function formatUsdLimit(value) {
    if (value == null)
        return '∞';
    return "$".concat(Number(value || 0).toFixed(2));
}
var CompactUsageRow = function (_a) {
    var label = _a.label, used = _a.used, limit = _a.limit, remaining = _a.remaining, usedUsd = _a.usedUsd, limitUsd = _a.limitUsd, remainingUsd = _a.remainingUsd;
    var hasUsd = usedUsd != null || limitUsd != null || remainingUsd != null;
    return (<div className="rounded-lg border border-[#E6F1F0] bg-white px-2.5 py-1.5 text-[12px]">
        <div className="flex items-center justify-between gap-3">
            <span className="text-[#3A5672]">{label}</span>
            <span className="font-mono font-semibold text-[#0D1E2C]">
                {hasUsd ? "$".concat(Number(usedUsd || 0).toFixed(2), " / ").concat(formatUsdLimit(limitUsd)) : "".concat(formatCount(used), " / ").concat(formatCount(limit))}
            </span>
        </div>
        <div className="mt-0.5 text-[11px] text-[#7A99B0]">
            Remaining: {hasUsd ? formatUsdLimit(remainingUsd) : formatCount(remaining)}
        </div>
        {hasUsd && (<div className="mt-0.5 text-[11px] text-[#7A99B0]">
                Tokens: {formatCount(used)} / {formatCount(limit)} · remaining {formatCount(remaining)}
            </div>)}
    </div>);
};
var PolicyMetricList = function (_a) {
    var policy = _a.policy;
    return (<div className="space-y-0.5 text-[11.5px] text-[#3A5672]">
        <div>req/day: <span className="font-mono font-semibold text-[#0D1E2C]">{formatCount(policy.requests_per_day)}</span></div>
        <div>req/month: <span className="font-mono font-semibold text-[#0D1E2C]">{formatCount(policy.requests_per_month)}</span></div>
        <div>tok/hour: <span className="font-mono font-semibold text-[#0D1E2C]">{formatCount(policy.tokens_per_hour)}</span></div>
        <div>tok/day: <span className="font-mono font-semibold text-[#0D1E2C]">{formatCount(policy.tokens_per_day)}</span></div>
        <div>tok/month: <span className="font-mono font-semibold text-[#0D1E2C]">{formatCount(policy.tokens_per_month)}</span></div>
        {policy.usd_per_month != null && (<div>month value: <span className="font-mono font-semibold text-[#0D1E2C]">${Number(policy.usd_per_month).toFixed(2)}</span></div>)}
    </div>);
};
var Tabs = function (_a) {
    var active = _a.active, onChange = _a.onChange, items = _a.items;
    return (<div className="flex flex-wrap gap-1">
        {items.map(function (t) {
            var isActive = active === t.id;
            return (<button key={t.id} onClick={function () { return onChange(t.id); }} className={[
                    "inline-flex h-8 items-center whitespace-nowrap rounded-lg border px-2.5 text-[12px] font-semibold transition-colors",
                    isActive
                        ? "border-[#01BEB2] bg-white text-[#10304B]"
                        : "border-transparent text-[#3A5672] hover:bg-white hover:text-[#0D1E2C]",
                ].join(' ')}>
                    {t.label}
                </button>);
        })}
    </div>);
};
var DividerTitle = function (_a) {
    var title = _a.title, subtitle = _a.subtitle;
    return (<div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-0.5">
        <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#009C92]">CONTROL PLANE</span>
        <h1 className="text-[15px] font-bold leading-6 text-[#0D1E2C]">
            {title}
        </h1>
        {subtitle && (<p className="truncate text-[11.5px] text-[#7A99B0]">
                {subtitle}
            </p>)}
    </div>);
};
// =============================================================================
// Economics Explainers
// =============================================================================
var Details = function (_a) {
    var title = _a.title, children = _a.children;
    return (<details className="rounded-xl border border-[#E6F1F0] bg-white px-3 py-2">
        <summary className="cursor-pointer text-[12px] font-semibold text-[#10304B]">{title}</summary>
        <div className="mt-2 space-y-2 text-[11.5px] leading-snug text-[#3A5672]">{children}</div>
    </details>);
};
var EconomicsOverview = function (_a) {
    var goTo = _a.goTo;
    return (<details className="rounded-xl border border-[#E6F1F0] bg-white px-3 py-1.5 shadow-[0_1px_2px_rgba(13,30,44,0.04)]">
        <summary className="cursor-pointer text-[12px] font-semibold text-[#10304B]">
            Funding rules and admin levers
        </summary>
        <div className="mt-2 space-y-2">
            <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg border border-[#E6F1F0] bg-white p-2 text-[11.5px] leading-snug text-[#3A5672]">
                    <div className="font-semibold text-[#10304B]">Plan quota</div>
                    <div className="mt-0.5">
                        The effective plan is the base plan, replaced by an active user override when one exists.
                        Plan quota is consumed first and is funded from the project budget.
                    </div>
                </div>
                <div className="rounded-lg border border-[#E6F1F0] bg-white p-2 text-[11.5px] leading-snug text-[#3A5672]">
                    <div className="font-semibold text-[#10304B]">Wallet / personal credits</div>
                    <div className="mt-0.5">
                        Personal credits cover the part that cannot be funded by remaining quota or available project budget.
                    </div>
                </div>
            </div>
            {goTo && (<div className="flex flex-wrap gap-1.5">
                    <Button variant="secondary" onClick={function () { return goTo('quotaBreakdown'); }}>Budget Breakdown</Button>
                    <Button variant="secondary" onClick={function () { return goTo('quotaPolicies'); }}>Plan Limits</Button>
                    <Button variant="secondary" onClick={function () { return goTo('lifetimeCredits'); }}>Wallet Credits</Button>
                    <Button variant="secondary" onClick={function () { return goTo('appBudget'); }}>App Budget</Button>
                </div>)}
        </div>
    </details>);
};
// =============================================================================
// Main Economics Admin Component
// =============================================================================
var EconomicsAdmin = function () {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _1, _2, _3, _4;
    var api = (0, react_1.useMemo)(function () { return new EconomicsAPI(); }, []);
    var _5 = (0, react_1.useState)('initializing'), configStatus = _5[0], setConfigStatus = _5[1];
    var _6 = (0, react_1.useState)('grantTrial'), viewMode = _6[0], setViewMode = _6[1];
    // separate loading channels: data loading vs actions
    var _7 = (0, react_1.useState)(false), loadingData = _7[0], setLoadingData = _7[1];
    var _8 = (0, react_1.useState)(false), loadingAction = _8[0], setLoadingAction = _8[1];
    var _9 = (0, react_1.useState)(null), error = _9[0], setError = _9[1];
    var _10 = (0, react_1.useState)(null), success = _10[0], setSuccess = _10[1];
    // Data
    var _11 = (0, react_1.useState)([]), quotaPolicies = _11[0], setQuotaPolicies = _11[1];
    var _12 = (0, react_1.useState)([]), budgetPolicies = _12[0], setBudgetPolicies = _12[1];
    var _13 = (0, react_1.useState)(null), appBudget = _13[0], setAppBudget = _13[1];
    var _14 = (0, react_1.useState)({}), reservation = _14[0], setReservation = _14[1];
    var _15 = (0, react_1.useState)('chat'), reservationFloor = _15[0], setReservationFloor = _15[1];
    var _16 = (0, react_1.useState)(''), reservationAmount = _16[0], setReservationAmount = _16[1];
    // Forms - Grant Trial
    var _17 = (0, react_1.useState)(''), trialUserId = _17[0], setTrialUserId = _17[1];
    var _18 = (0, react_1.useState)(7), trialDays = _18[0], setTrialDays = _18[1];
    var _19 = (0, react_1.useState)(100), trialRequests = _19[0], setTrialRequests = _19[1];
    var _20 = (0, react_1.useState)(''), trialTokensHour = _20[0], setTrialTokensHour = _20[1];
    var _21 = (0, react_1.useState)(''), trialTokensDay = _21[0], setTrialTokensDay = _21[1];
    var _22 = (0, react_1.useState)('300000000'), trialTokensMonth = _22[0], setTrialTokensMonth = _22[1];
    var _23 = (0, react_1.useState)(''), trialUsdHour = _23[0], setTrialUsdHour = _23[1];
    var _24 = (0, react_1.useState)(''), trialUsdDay = _24[0], setTrialUsdDay = _24[1];
    var _25 = (0, react_1.useState)(''), trialUsdMonth = _25[0], setTrialUsdMonth = _25[1];
    var _26 = (0, react_1.useState)(''), trialNotes = _26[0], setTrialNotes = _26[1];
    // Forms - Update Tier Budget
    var _27 = (0, react_1.useState)(''), updateUserId = _27[0], setUpdateUserId = _27[1];
    var _28 = (0, react_1.useState)(''), updateRequestsDay = _28[0], setUpdateRequestsDay = _28[1];
    var _29 = (0, react_1.useState)(''), updateRequestsMonth = _29[0], setUpdateRequestsMonth = _29[1];
    var _30 = (0, react_1.useState)(''), updateTokensHour = _30[0], setUpdateTokensHour = _30[1];
    var _31 = (0, react_1.useState)(''), updateTokensDay = _31[0], setUpdateTokensDay = _31[1];
    var _32 = (0, react_1.useState)(''), updateTokensMonth = _32[0], setUpdateTokensMonth = _32[1];
    var _33 = (0, react_1.useState)(''), updateUsdHour = _33[0], setUpdateUsdHour = _33[1];
    var _34 = (0, react_1.useState)(''), updateUsdDay = _34[0], setUpdateUsdDay = _34[1];
    var _35 = (0, react_1.useState)(''), updateUsdMonth = _35[0], setUpdateUsdMonth = _35[1];
    var _36 = (0, react_1.useState)(''), updateMaxConcurrent = _36[0], setUpdateMaxConcurrent = _36[1];
    var _37 = (0, react_1.useState)('30'), updateExpiresDays = _37[0], setUpdateExpiresDays = _37[1];
    var _38 = (0, react_1.useState)(''), updateNotes = _38[0], setUpdateNotes = _38[1];
    // Forms - Tier Balance Lookup
    var _39 = (0, react_1.useState)(''), lookupUserId = _39[0], setLookupUserId = _39[1];
    var _40 = (0, react_1.useState)(null), planBalance = _40[0], setPlanBalance = _40[1];
    // Forms - Quota Policy
    var _41 = (0, react_1.useState)('free'), policyPlanId = _41[0], setPolicyPlanId = _41[1];
    var _42 = (0, react_1.useState)(''), policyPlanIdCustom = _42[0], setPolicyPlanIdCustom = _42[1];
    var _43 = (0, react_1.useState)(''), policyMaxConcurrent = _43[0], setPolicyMaxConcurrent = _43[1];
    var _44 = (0, react_1.useState)(''), policyRequestsDay = _44[0], setPolicyRequestsDay = _44[1];
    var _45 = (0, react_1.useState)(''), policyRequestsMonth = _45[0], setPolicyRequestsMonth = _45[1];
    var _46 = (0, react_1.useState)(''), policyTokensHour = _46[0], setPolicyTokensHour = _46[1];
    var _47 = (0, react_1.useState)(''), policyTokensDay = _47[0], setPolicyTokensDay = _47[1];
    var _48 = (0, react_1.useState)(''), policyTokensMonth = _48[0], setPolicyTokensMonth = _48[1];
    var _49 = (0, react_1.useState)(''), policyUsdHour = _49[0], setPolicyUsdHour = _49[1];
    var _50 = (0, react_1.useState)(''), policyUsdDay = _50[0], setPolicyUsdDay = _50[1];
    var _51 = (0, react_1.useState)(''), policyUsdMonth = _51[0], setPolicyUsdMonth = _51[1];
    var _52 = (0, react_1.useState)(''), policyNotes = _52[0], setPolicyNotes = _52[1];
    // Forms - Budget Policy
    var _53 = (0, react_1.useState)(''), budgetProvider = _53[0], setBudgetProvider = _53[1];
    var _54 = (0, react_1.useState)(''), budgetUsdHour = _54[0], setBudgetUsdHour = _54[1];
    var _55 = (0, react_1.useState)(''), budgetUsdDay = _55[0], setBudgetUsdDay = _55[1];
    var _56 = (0, react_1.useState)(''), budgetUsdMonth = _56[0], setBudgetUsdMonth = _56[1];
    var _57 = (0, react_1.useState)(''), budgetNotes = _57[0], setBudgetNotes = _57[1];
    // Forms - Quota Breakdown
    var _58 = (0, react_1.useState)(''), breakdownUserId = _58[0], setBreakdownUserId = _58[1];
    var _59 = (0, react_1.useState)(null), quotaBreakdown = _59[0], setQuotaBreakdown = _59[1];
    var _60 = (0, react_1.useState)(''), breakdownBundleId = _60[0], setBreakdownBundleId = _60[1];
    // Forms - Lifetime Credits
    var _61 = (0, react_1.useState)(''), lifetimeUserId = _61[0], setLifetimeUserId = _61[1];
    var _62 = (0, react_1.useState)(''), lifetimeUsdAmount = _62[0], setLifetimeUsdAmount = _62[1];
    var _63 = (0, react_1.useState)(''), lifetimeNotes = _63[0], setLifetimeNotes = _63[1];
    var _64 = (0, react_1.useState)(null), lifetimeBalance = _64[0], setLifetimeBalance = _64[1];
    // App Budget
    var _65 = (0, react_1.useState)(''), appBudgetTopup = _65[0], setAppBudgetTopup = _65[1];
    var _66 = (0, react_1.useState)(''), appBudgetNotes = _66[0], setAppBudgetNotes = _66[1];
    var _67 = (0, react_1.useState)('month'), absorptionPeriod = _67[0], setAbsorptionPeriod = _67[1];
    var _68 = (0, react_1.useState)('none'), absorptionGroupBy = _68[0], setAbsorptionGroupBy = _68[1];
    var _69 = (0, react_1.useState)('90'), absorptionDays = _69[0], setAbsorptionDays = _69[1];
    var _70 = (0, react_1.useState)([]), absorptionItems = _70[0], setAbsorptionItems = _70[1];
    var _71 = (0, react_1.useState)(false), loadingAbsorption = _71[0], setLoadingAbsorption = _71[1];
    // Cost report — true per-model spend from the OPEX aggregates, grouped by
    // user, agent, or app.
    var _todayIso = new Date().toISOString().slice(0, 10);
    var _72 = (0, react_1.useState)('user'), costDim = _72[0], setCostDim = _72[1];
    var _73 = (0, react_1.useState)(_todayIso.slice(0, 8) + '01'), costFrom = _73[0], setCostFrom = _73[1];
    var _74 = (0, react_1.useState)(_todayIso), costTo = _74[0], setCostTo = _74[1];
    var _75 = (0, react_1.useState)(''), costFilter = _75[0], setCostFilter = _75[1];
    var _76 = (0, react_1.useState)([]), costRows = _76[0], setCostRows = _76[1];
    var _77 = (0, react_1.useState)(false), costLoaded = _77[0], setCostLoaded = _77[1];
    var _78 = (0, react_1.useState)(false), loadingCost = _78[0], setLoadingCost = _78[1];
    var _79 = (0, react_1.useState)({}), costExpanded = _79[0], setCostExpanded = _79[1];
    // Comma/space-separated tokens; a row matches when any token is a
    // case-insensitive substring of its id. Empty filter matches all.
    var costFilterTokens = costFilter.split(/[\s,]+/).map(function (t) { return t.trim().toLowerCase(); }).filter(Boolean);
    var costRowsFiltered = costFilterTokens.length
        ? costRows.filter(function (r) { return costFilterTokens.some(function (t) { return r.id.toLowerCase().includes(t); }); })
        : costRows;
    var costUserRows = costRowsFiltered.filter(function (r) { return !r.system; });
    var costSystemRows = costDim === 'user' ? costRowsFiltered.filter(function (r) { return r.system; }) : [];
    var _80 = (0, react_1.useState)(''), lineageRequestId = _80[0], setLineageRequestId = _80[1];
    var _81 = (0, react_1.useState)(null), lineageResult = _81[0], setLineageResult = _81[1];
    var _82 = (0, react_1.useState)(false), loadingLineage = _82[0], setLoadingLineage = _82[1];
    // Subscriptions
    var _83 = (0, react_1.useState)('internal'), subProvider = _83[0], setSubProvider = _83[1];
    var _84 = (0, react_1.useState)(''), subUserId = _84[0], setSubUserId = _84[1];
    var _85 = (0, react_1.useState)(''), subPlanId = _85[0], setSubPlanId = _85[1];
    var _86 = (0, react_1.useState)(''), subStripePriceId = _86[0], setSubStripePriceId = _86[1];
    var _87 = (0, react_1.useState)(''), subStripeCustomerId = _87[0], setSubStripeCustomerId = _87[1];
    var _88 = (0, react_1.useState)(''), subPriceHint = _88[0], setSubPriceHint = _88[1];
    var _89 = (0, react_1.useState)(''), planId = _89[0], setPlanId = _89[1];
    var _90 = (0, react_1.useState)('internal'), planProvider = _90[0], setPlanProvider = _90[1];
    var _91 = (0, react_1.useState)(''), planStripePriceId = _91[0], setPlanStripePriceId = _91[1];
    var _92 = (0, react_1.useState)('0'), planPriceCents = _92[0], setPlanPriceCents = _92[1];
    var _93 = (0, react_1.useState)(true), planActive = _93[0], setPlanActive = _93[1];
    var _94 = (0, react_1.useState)(''), planNotes = _94[0], setPlanNotes = _94[1];
    var _95 = (0, react_1.useState)([]), subscriptionPlans = _95[0], setSubscriptionPlans = _95[1];
    var _96 = (0, react_1.useState)(false), loadingPlans = _96[0], setLoadingPlans = _96[1];
    var _97 = (0, react_1.useState)(''), subLookupUserId = _97[0], setSubLookupUserId = _97[1];
    var _98 = (0, react_1.useState)(null), subscription = _98[0], setSubscription = _98[1];
    var _99 = (0, react_1.useState)(''), subBudgetUserId = _99[0], setSubBudgetUserId = _99[1];
    var _100 = (0, react_1.useState)(''), subBudgetUsdAmount = _100[0], setSubBudgetUsdAmount = _100[1];
    var _101 = (0, react_1.useState)(''), subBudgetNotes = _101[0], setSubBudgetNotes = _101[1];
    var _102 = (0, react_1.useState)(false), subBudgetForceTopup = _102[0], setSubBudgetForceTopup = _102[1];
    var _103 = (0, react_1.useState)(''), subSweepUserId = _103[0], setSubSweepUserId = _103[1];
    var _104 = (0, react_1.useState)(null), subscriptionBalance = _104[0], setSubscriptionBalance = _104[1];
    var _105 = (0, react_1.useState)(''), subReapUserId = _105[0], setSubReapUserId = _105[1];
    var _106 = (0, react_1.useState)('500'), subReapLimitPeriods = _106[0], setSubReapLimitPeriods = _106[1];
    var _107 = (0, react_1.useState)('500'), subReapPerPeriodLimit = _107[0], setSubReapPerPeriodLimit = _107[1];
    var _108 = (0, react_1.useState)(''), walletRefundUserId = _108[0], setWalletRefundUserId = _108[1];
    var _109 = (0, react_1.useState)(''), walletRefundPaymentIntentId = _109[0], setWalletRefundPaymentIntentId = _109[1];
    var _110 = (0, react_1.useState)(''), walletRefundUsdAmount = _110[0], setWalletRefundUsdAmount = _110[1];
    var _111 = (0, react_1.useState)(''), walletRefundNotes = _111[0], setWalletRefundNotes = _111[1];
    var _112 = (0, react_1.useState)(''), cancelSubUserId = _112[0], setCancelSubUserId = _112[1];
    var _113 = (0, react_1.useState)(''), cancelSubStripeId = _113[0], setCancelSubStripeId = _113[1];
    var _114 = (0, react_1.useState)(''), cancelSubNotes = _114[0], setCancelSubNotes = _114[1];
    var _115 = (0, react_1.useState)('all'), stripeReconcileKind = _115[0], setStripeReconcileKind = _115[1];
    var _116 = (0, react_1.useState)('all'), pendingStripeKind = _116[0], setPendingStripeKind = _116[1];
    var _117 = (0, react_1.useState)([]), pendingStripeItems = _117[0], setPendingStripeItems = _117[1];
    var _118 = (0, react_1.useState)(false), loadingPendingStripe = _118[0], setLoadingPendingStripe = _118[1];
    var _119 = (0, react_1.useState)(''), pendingEconomicsKind = _119[0], setPendingEconomicsKind = _119[1];
    var _120 = (0, react_1.useState)(''), pendingEconomicsUserId = _120[0], setPendingEconomicsUserId = _120[1];
    var _121 = (0, react_1.useState)([]), pendingEconomicsItems = _121[0], setPendingEconomicsItems = _121[1];
    var _122 = (0, react_1.useState)(false), loadingPendingEconomics = _122[0], setLoadingPendingEconomics = _122[1];
    var _123 = (0, react_1.useState)(''), subHistoryUserId = _123[0], setSubHistoryUserId = _123[1];
    var _124 = (0, react_1.useState)('closed'), subHistoryStatus = _124[0], setSubHistoryStatus = _124[1];
    var _125 = (0, react_1.useState)([]), subPeriods = _125[0], setSubPeriods = _125[1];
    var _126 = (0, react_1.useState)([]), subLedger = _126[0], setSubLedger = _126[1];
    var _127 = (0, react_1.useState)(''), subSelectedPeriodKey = _127[0], setSubSelectedPeriodKey = _127[1];
    var _128 = (0, react_1.useState)(false), loadingHistory = _128[0], setLoadingHistory = _128[1];
    var _129 = (0, react_1.useState)(''), subsProviderFilter = _129[0], setSubsProviderFilter = _129[1];
    var _130 = (0, react_1.useState)([]), subsList = _130[0], setSubsList = _130[1];
    var _131 = (0, react_1.useState)(null), economicsRef = _131[0], setEconomicsRef = _131[1];
    var safeNumber = function (v) { return (typeof v === 'number' && Number.isFinite(v) ? v : 0); };
    var usdToTokens = function (usdText) {
        if (!economicsRef)
            return null;
        var usd = parseFloat(usdText);
        if (!Number.isFinite(usd) || usd <= 0)
            return null;
        return Math.floor(usd / economicsRef.usd_per_token);
    };
    var tokensToUsd = function (tokenText) {
        if (!economicsRef)
            return null;
        var tokens = parseInt(tokenText);
        if (!Number.isFinite(tokens) || tokens <= 0)
            return null;
        return tokens * economicsRef.usd_per_token;
    };
    (0, react_1.useEffect)(function () {
        var initializeSettings = function () { return __awaiter(void 0, void 0, void 0, function () {
            var configReceived, err_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        console.log('[Admin] Initializing settings');
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 3, , 4]);
                        return [4 /*yield*/, settings.setupParentListener()];
                    case 2:
                        configReceived = _a.sent();
                        console.log('[Admin] Config received?', configReceived);
                        if (configReceived || !window.parent || window.parent === window) {
                            setConfigStatus('ready');
                        }
                        return [3 /*break*/, 4];
                    case 3:
                        err_1 = _a.sent();
                        console.error('[Admin] Error initializing settings:', err_1);
                        setConfigStatus('error');
                        return [3 /*break*/, 4];
                    case 4: return [2 /*return*/];
                }
            });
        }); };
        initializeSettings();
    }, []);
    (0, react_1.useEffect)(function () {
        if (configStatus === 'ready') {
            loadDataForView(viewMode);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [configStatus, viewMode]);
    (0, react_1.useEffect)(function () {
        var loadEconomicsRef = function () { return __awaiter(void 0, void 0, void 0, function () {
            var ref, err_2;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (configStatus !== 'ready')
                            return [2 /*return*/];
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 3, , 4]);
                        return [4 /*yield*/, api.getEconomicsReference()];
                    case 2:
                        ref = _a.sent();
                        if (ref.status === 'ok') {
                            setEconomicsRef(ref);
                        }
                        return [3 /*break*/, 4];
                    case 3:
                        err_2 = _a.sent();
                        console.warn('Failed to load economics reference:', err_2);
                        return [3 /*break*/, 4];
                    case 4: return [2 /*return*/];
                }
            });
        }); };
        loadEconomicsRef();
    }, [api, configStatus]);
    var loadDataForView = function (mode) { return __awaiter(void 0, void 0, void 0, function () {
        var needsData, result, result, balance, res, res, err_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    needsData = ['quotaPolicies', 'budgetPolicies', 'appBudget', 'plans', 'reservation'].includes(mode);
                    if (!needsData)
                        return [2 /*return*/];
                    setLoadingData(true);
                    setError(null);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 12, 13, 14]);
                    if (!(mode === 'quotaPolicies')) return [3 /*break*/, 3];
                    return [4 /*yield*/, api.listQuotaPolicies()];
                case 2:
                    result = _a.sent();
                    setQuotaPolicies(result.policies || []);
                    return [3 /*break*/, 11];
                case 3:
                    if (!(mode === 'budgetPolicies')) return [3 /*break*/, 5];
                    return [4 /*yield*/, api.listBudgetPolicies()];
                case 4:
                    result = _a.sent();
                    setBudgetPolicies(result.policies || []);
                    return [3 /*break*/, 11];
                case 5:
                    if (!(mode === 'appBudget')) return [3 /*break*/, 7];
                    return [4 /*yield*/, api.getAppBudgetBalance()];
                case 6:
                    balance = _a.sent();
                    setAppBudget(balance);
                    return [3 /*break*/, 11];
                case 7:
                    if (!(mode === 'reservation')) return [3 /*break*/, 9];
                    return [4 /*yield*/, api.getReservation()];
                case 8:
                    res = _a.sent();
                    setReservation(res.reservation || {});
                    return [3 /*break*/, 11];
                case 9:
                    if (!(mode === 'plans')) return [3 /*break*/, 11];
                    return [4 /*yield*/, api.listSubscriptionPlans({ limit: 200, offset: 0, activeOnly: false })];
                case 10:
                    res = _a.sent();
                    setSubscriptionPlans(res.plans || []);
                    _a.label = 11;
                case 11: return [3 /*break*/, 14];
                case 12:
                    err_3 = _a.sent();
                    setError(err_3.message);
                    console.error('Failed to load data:', err_3);
                    return [3 /*break*/, 14];
                case 13:
                    setLoadingData(false);
                    return [7 /*endfinally*/];
                case 14: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadSubscriptionPlans = function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingPlans(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.listSubscriptionPlans({ limit: 200, offset: 0, activeOnly: false })];
                case 2:
                    res = _a.sent();
                    setSubscriptionPlans(res.plans || []);
                    return [3 /*break*/, 5];
                case 3:
                    err_4 = _a.sent();
                    setError(err_4.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingPlans(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var clearMessages = function () {
        setError(null);
        setSuccess(null);
    };
    var handleGrantTrial = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var err_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.grantTrial({
                            userId: trialUserId,
                            days: trialDays,
                            requestsPerDay: trialRequests,
                            tokensPerHour: trialTokensHour ? parseInt(trialTokensHour) : undefined,
                            tokensPerDay: trialTokensDay ? parseInt(trialTokensDay) : undefined,
                            tokensPerMonth: trialTokensMonth ? parseInt(trialTokensMonth) : undefined,
                            usdPerHour: trialUsdHour ? parseFloat(trialUsdHour) : undefined,
                            usdPerDay: trialUsdDay ? parseFloat(trialUsdDay) : undefined,
                            usdPerMonth: trialUsdMonth ? parseFloat(trialUsdMonth) : undefined,
                            notes: trialNotes,
                        })];
                case 2:
                    _a.sent();
                    setSuccess("Trial granted to ".concat(trialUserId));
                    setTrialUserId('');
                    setTrialNotes('');
                    setTrialTokensHour('');
                    setTrialTokensDay('');
                    setTrialTokensMonth('300000000');
                    setTrialUsdHour('');
                    setTrialUsdDay('');
                    setTrialUsdMonth('');
                    return [3 /*break*/, 5];
                case 3:
                    err_5 = _a.sent();
                    setError(err_5.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleUpdateTierBudget = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var err_6;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.updatePlanOverride({
                            userId: updateUserId,
                            requestsPerDay: updateRequestsDay ? parseInt(updateRequestsDay) : undefined,
                            requestsPerMonth: updateRequestsMonth ? parseInt(updateRequestsMonth) : undefined,
                            tokensPerHour: updateTokensHour ? parseInt(updateTokensHour) : undefined,
                            tokensPerDay: updateTokensDay ? parseInt(updateTokensDay) : undefined,
                            tokensPerMonth: updateTokensMonth ? parseInt(updateTokensMonth) : undefined,
                            usdPerHour: updateUsdHour ? parseFloat(updateUsdHour) : undefined,
                            usdPerDay: updateUsdDay ? parseFloat(updateUsdDay) : undefined,
                            usdPerMonth: updateUsdMonth ? parseFloat(updateUsdMonth) : undefined,
                            maxConcurrent: updateMaxConcurrent ? parseInt(updateMaxConcurrent) : undefined,
                            expiresInDays: updateExpiresDays === '' ? null : parseInt(updateExpiresDays),
                            notes: updateNotes
                        })];
                case 2:
                    _a.sent();
                    setSuccess("Tier override updated for ".concat(updateUserId));
                    setUpdateUserId('');
                    setUpdateRequestsDay('');
                    setUpdateRequestsMonth('');
                    setUpdateTokensHour('');
                    setUpdateTokensDay('');
                    setUpdateTokensMonth('');
                    setUpdateUsdHour('');
                    setUpdateUsdDay('');
                    setUpdateUsdMonth('');
                    setUpdateMaxConcurrent('');
                    setUpdateExpiresDays('30');
                    setUpdateNotes('');
                    return [3 /*break*/, 5];
                case 3:
                    err_6 = _a.sent();
                    setError(err_6.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLookupPlanBalance = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var result, err_7;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setPlanBalance(null);
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.getPlanBalance(lookupUserId)];
                case 2:
                    result = _a.sent();
                    setPlanBalance(result);
                    return [3 /*break*/, 5];
                case 3:
                    err_7 = _a.sent();
                    setError(err_7.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleGetQuotaBreakdown = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var planId_1, bundleId, result, err_8;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setQuotaBreakdown(null);
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    planId_1 = undefined;
                    bundleId = breakdownBundleId.trim() || undefined;
                    return [4 /*yield*/, api.getUserBudgetBreakdown(breakdownUserId, planId_1, bundleId)];
                case 2:
                    result = _a.sent();
                    setQuotaBreakdown(result);
                    return [3 /*break*/, 5];
                case 3:
                    err_8 = _a.sent();
                    setError(err_8.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleSetQuotaPolicy = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var err_9;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    return [4 /*yield*/, api.setQuotaPolicy({
                            planId: policyPlanId === 'custom' ? policyPlanIdCustom : policyPlanId,
                            maxConcurrent: policyMaxConcurrent ? parseInt(policyMaxConcurrent) : undefined,
                            requestsPerDay: policyRequestsDay ? parseInt(policyRequestsDay) : undefined,
                            requestsPerMonth: policyRequestsMonth ? parseInt(policyRequestsMonth) : undefined,
                            tokensPerHour: policyTokensHour ? parseInt(policyTokensHour) : undefined,
                            tokensPerDay: policyTokensDay ? parseInt(policyTokensDay) : undefined,
                            tokensPerMonth: policyTokensMonth ? parseInt(policyTokensMonth) : undefined,
                            usdPerHour: policyUsdHour ? parseFloat(policyUsdHour) : undefined,
                            usdPerDay: policyUsdDay ? parseFloat(policyUsdDay) : undefined,
                            usdPerMonth: policyUsdMonth ? parseFloat(policyUsdMonth) : undefined,
                            notes: policyNotes
                        })];
                case 2:
                    _a.sent();
                    setSuccess("Quota policy set for ".concat(policyPlanId));
                    // setPolicyUserType(policyUserType);
                    setPolicyMaxConcurrent('');
                    setPolicyRequestsDay('');
                    setPolicyRequestsMonth('');
                    setPolicyTokensHour('');
                    setPolicyTokensDay('');
                    setPolicyTokensMonth('');
                    setPolicyUsdHour('');
                    setPolicyUsdDay('');
                    setPolicyUsdMonth('');
                    setPolicyPlanIdCustom('');
                    setPolicyNotes('');
                    return [4 /*yield*/, loadDataForView('quotaPolicies')];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 6];
                case 4:
                    err_9 = _a.sent();
                    setError(err_9.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var handleSetReservation = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var floor, amount, res, err_10;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    floor = (reservationFloor || 'chat').trim() || 'chat';
                    amount = parseFloat(reservationAmount);
                    if (Number.isNaN(amount)) {
                        throw new Error('Enter a numeric reservation amount (USD). Use <= 0 to disable the floor.');
                    }
                    return [4 /*yield*/, api.setReservation(floor, amount)];
                case 2:
                    res = _a.sent();
                    setReservation(res.reservation || {});
                    setReservationAmount('');
                    setSuccess(amount > 0
                        ? "Reservation floor for ".concat(floor, " set to $").concat(amount.toFixed(2))
                        : "Reservation floor for ".concat(floor, " disabled"));
                    return [3 /*break*/, 5];
                case 3:
                    err_10 = _a.sent();
                    setError(err_10.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleDeleteReservation = function (floor) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_11;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!window.confirm("Remove the \"".concat(floor, "\" reservation floor? The surface will inherit the platform/app default."))) {
                        return [2 /*return*/];
                    }
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.deleteReservation(floor)];
                case 2:
                    res = _a.sent();
                    setReservation(res.reservation || {});
                    setSuccess("Reservation floor \"".concat(floor, "\" removed."));
                    return [3 /*break*/, 5];
                case 3:
                    err_11 = _a.sent();
                    setError(err_11.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleSetBudgetPolicy = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var err_12;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    return [4 /*yield*/, api.setBudgetPolicy({
                            provider: budgetProvider,
                            usdPerHour: budgetUsdHour ? parseFloat(budgetUsdHour) : undefined,
                            usdPerDay: budgetUsdDay ? parseFloat(budgetUsdDay) : undefined,
                            usdPerMonth: budgetUsdMonth ? parseFloat(budgetUsdMonth) : undefined,
                            notes: budgetNotes
                        })];
                case 2:
                    _a.sent();
                    setSuccess("Budget policy set for ".concat(budgetProvider));
                    setBudgetProvider('');
                    setBudgetUsdHour('');
                    setBudgetUsdDay('');
                    setBudgetUsdMonth('');
                    setBudgetNotes('');
                    return [4 /*yield*/, loadDataForView('budgetPolicies')];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 6];
                case 4:
                    err_12 = _a.sent();
                    setError(err_12.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var handleAddLifetimeCredits = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var uid, balance, err_13;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    uid = lifetimeUserId.trim();
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    return [4 /*yield*/, api.addLifetimeCredits(uid, parseFloat(lifetimeUsdAmount), lifetimeNotes)];
                case 2:
                    _a.sent();
                    setSuccess("Added $".concat(lifetimeUsdAmount, " to ").concat(uid));
                    setLifetimeUsdAmount('');
                    setLifetimeNotes('');
                    return [4 /*yield*/, api.getLifetimeBalance(uid)];
                case 3:
                    balance = _a.sent();
                    setLifetimeBalance(balance);
                    return [3 /*break*/, 6];
                case 4:
                    err_13 = _a.sent();
                    setError(err_13.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var handleCheckLifetimeBalance = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var uid, balance, err_14;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLifetimeBalance(null);
                    setLoadingAction(true);
                    uid = lifetimeUserId.trim();
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.getLifetimeBalance(uid)];
                case 2:
                    balance = _a.sent();
                    setLifetimeBalance(balance);
                    return [3 /*break*/, 5];
                case 3:
                    err_14 = _a.sent();
                    setError(err_14.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleTopupAppBudget = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var balance, err_15;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    return [4 /*yield*/, api.topupAppBudget(parseFloat(appBudgetTopup), appBudgetNotes)];
                case 2:
                    _a.sent();
                    setSuccess("App budget topped up: $".concat(appBudgetTopup));
                    setAppBudgetTopup('');
                    setAppBudgetNotes('');
                    return [4 /*yield*/, api.getAppBudgetBalance()];
                case 3:
                    balance = _a.sent();
                    setAppBudget(balance);
                    return [3 /*break*/, 6];
                case 4:
                    err_15 = _a.sent();
                    setError(err_15.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadAbsorptionReport = function () { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, days, res, err_16;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingAbsorption(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    parsed = parseInt(absorptionDays || '90', 10);
                    days = Number.isFinite(parsed) ? Math.max(1, parsed) : 90;
                    return [4 /*yield*/, api.getAppBudgetAbsorptionReport(absorptionPeriod, days, absorptionGroupBy)];
                case 2:
                    res = _a.sent();
                    setAbsorptionItems(res.items || []);
                    if (!res.items || res.items.length === 0) {
                        setSuccess('No absorption events found for the selected period.');
                    }
                    return [3 /*break*/, 5];
                case 3:
                    err_16 = _a.sent();
                    setError(err_16.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAbsorption(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadCostReport = function (dim) { return __awaiter(void 0, void 0, void 0, function () {
        var dimension, res, entries_1, estimates_1, rows, err_17;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    dimension = dim || costDim;
                    clearMessages();
                    setLoadingCost(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.getOpexCost(dimension, costFrom, costTo)];
                case 2:
                    res = _a.sent();
                    entries_1 = (dimension === 'agent' ? res.agents : dimension === 'app' ? res.apps : res.users) || {};
                    estimates_1 = res.cost_estimate || {};
                    rows = Object.keys(entries_1).map(function (id) {
                        var e = entries_1[id] || {};
                        var total = (e.total || {});
                        var est = estimates_1[id];
                        return {
                            id: id,
                            system: dimension === 'user' && SYSTEM_PRINCIPAL_IDS.has(id),
                            costUsd: Number((est === null || est === void 0 ? void 0 : est.total_cost_usd) || 0),
                            inputTokens: Number(total.input_tokens || 0),
                            outputTokens: Number(total.output_tokens || 0),
                            events: Number(e.event_count || 0),
                            byModel: __spreadArray([], ((est === null || est === void 0 ? void 0 : est.breakdown) || []), true).sort(function (a, b) { return Number(b.cost_usd || 0) - Number(a.cost_usd || 0); }),
                        };
                    });
                    rows.sort(function (a, b) { return b.costUsd - a.costUsd; });
                    setCostRows(rows);
                    setCostExpanded({});
                    setCostLoaded(true);
                    if (!rows.length)
                        setSuccess('No usage found for the selected window.');
                    return [3 /*break*/, 5];
                case 3:
                    err_17 = _a.sent();
                    setError(err_17.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingCost(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleCostDimChange = function (dim) {
        setCostDim(dim);
        if (costLoaded)
            void handleLoadCostReport(dim);
    };
    var handleRebuildAggregates = function () { return __awaiter(void 0, void 0, void 0, function () {
        var today, err_18;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!window.confirm("Rebuild spend aggregates for ".concat(costFrom, " \u2014 ").concat(costTo, "? This re-reads that window's raw events ") +
                        "(needed once after new report dimensions are introduced). Safe to re-run."))
                        return [2 /*return*/];
                    clearMessages();
                    setLoadingCost(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, , 5]);
                    today = new Date().toISOString().slice(0, 10);
                    return [4 /*yield*/, api.runAggregationRange(costFrom, costTo, costTo >= today)];
                case 2:
                    _a.sent();
                    setSuccess('Aggregates rebuilt.');
                    return [4 /*yield*/, handleLoadCostReport()];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 5];
                case 4:
                    err_18 = _a.sent();
                    setError(err_18.message);
                    setLoadingCost(false);
                    return [3 /*break*/, 5];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleExportCostCsv = function () {
        var rows = costRowsFiltered;
        var header = "".concat(costDim, "_id,cost_usd,input_tokens,output_tokens,events");
        var lines = rows.map(function (r) {
            return [r.id, r.costUsd.toFixed(6), r.inputTokens, r.outputTokens, r.events].join(',');
        });
        var blob = new Blob([__spreadArray([header], lines, true).join('\n')], { type: 'text/csv;charset=utf-8;' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = "cost-by-".concat(costDim, "-").concat(costFrom, "-").concat(costTo, ".csv");
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
    var handleExportAbsorptionCsv = function () { return __awaiter(void 0, void 0, void 0, function () {
        var parsed, days, csv, blob, url, a, err_19;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingAbsorption(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    parsed = parseInt(absorptionDays || '90', 10);
                    days = Number.isFinite(parsed) ? Math.max(1, parsed) : 90;
                    return [4 /*yield*/, api.getAppBudgetAbsorptionReportCsv(absorptionPeriod, days, absorptionGroupBy)];
                case 2:
                    csv = _a.sent();
                    blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                    url = URL.createObjectURL(blob);
                    a = document.createElement('a');
                    a.href = url;
                    a.download = "budget-absorption-".concat(absorptionPeriod, "-").concat(absorptionGroupBy, "-").concat(days, "d.csv");
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    URL.revokeObjectURL(url);
                    return [3 /*break*/, 5];
                case 3:
                    err_19 = _a.sent();
                    setError(err_19.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAbsorption(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadRequestLineage = function () { return __awaiter(void 0, void 0, void 0, function () {
        var reqId, res, err_20;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingLineage(true);
                    setLineageResult(null);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    reqId = lineageRequestId.trim();
                    if (!reqId) {
                        throw new Error('request_id is required');
                    }
                    return [4 /*yield*/, api.getRequestLineage(reqId)];
                case 2:
                    res = _a.sent();
                    setLineageResult(res);
                    return [3 /*break*/, 5];
                case 3:
                    err_20 = _a.sent();
                    setError(err_20.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingLineage(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleCopyRequestId = function () { return __awaiter(void 0, void 0, void 0, function () {
        var reqId, _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    reqId = lineageRequestId.trim();
                    if (!reqId) {
                        setError('request_id is empty');
                        return [2 /*return*/];
                    }
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, navigator.clipboard.writeText(reqId)];
                case 2:
                    _b.sent();
                    setSuccess('request_id copied');
                    return [3 /*break*/, 4];
                case 3:
                    _a = _b.sent();
                    setError('Copy failed (clipboard not available)');
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var formatUsdFromCents = function (cents) {
        if (cents === null || cents === undefined)
            return '-';
        return "$".concat((Number(cents) / 100).toFixed(2));
    };
    var formatUsd = function (usd) {
        if (usd === null || usd === undefined)
            return '-';
        return "$".concat(Number(usd).toFixed(2));
    };
    var formatDate = function (value) {
        if (!value)
            return '-';
        var d = new Date(value);
        if (Number.isNaN(d.getTime()))
            return String(value);
        return d.toLocaleString();
    };
    var handleCreateSubscription = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_21;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.createSubscription({
                            userId: subUserId.trim(),
                            planId: subPlanId.trim(),
                            provider: subProvider,
                            stripePriceId: subProvider === 'stripe' ? subStripePriceId.trim() : undefined,
                            stripeCustomerId: subProvider === 'stripe' ? (subStripeCustomerId.trim() || undefined) : undefined,
                            monthlyPriceCentsHint: subProvider === 'stripe' && subPriceHint ? parseInt(subPriceHint) : undefined,
                        })];
                case 2:
                    res = _a.sent();
                    setSuccess(res.message || "Subscription created for ".concat(subUserId));
                    setSubUserId('');
                    setSubStripePriceId('');
                    setSubStripeCustomerId('');
                    setSubPriceHint('');
                    setSubPlanId('');
                    return [3 /*break*/, 5];
                case 3:
                    err_21 = _a.sent();
                    setError(err_21.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleUpsertSubscriptionPlan = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_22;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    return [4 /*yield*/, api.upsertSubscriptionPlan({
                            planId: planId.trim(),
                            provider: planProvider,
                            stripePriceId: planProvider === 'stripe' ? (planStripePriceId.trim() || null) : null,
                            monthlyPriceCents: parseInt(planPriceCents || '0'),
                            active: planActive,
                            notes: planNotes || undefined,
                        })];
                case 2:
                    res = _a.sent();
                    setSuccess(res.message || "Plan saved: ".concat(planId));
                    return [4 /*yield*/, handleLoadSubscriptionPlans()];
                case 3:
                    _a.sent();
                    setPlanId('');
                    setPlanStripePriceId('');
                    setPlanPriceCents('0');
                    setPlanActive(true);
                    setPlanNotes('');
                    return [3 /*break*/, 6];
                case 4:
                    err_22 = _a.sent();
                    setError(err_22.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    var handleLookupSubscription = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_23;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    setSubscription(null);
                    setSubscriptionBalance(null);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.getSubscription(subLookupUserId.trim())];
                case 2:
                    res = _a.sent();
                    setSubscription(res.subscription);
                    setSubscriptionBalance(res.subscription_balance || null);
                    if (!res.subscription)
                        setSuccess('No subscription found for this user.');
                    return [3 /*break*/, 5];
                case 3:
                    err_23 = _a.sent();
                    setError(err_23.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadSubscriptionsList = function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_24;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingData(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.listSubscriptions({
                            provider: subsProviderFilter || undefined,
                            limit: 50,
                            offset: 0,
                        })];
                case 2:
                    res = _a.sent();
                    setSubsList(res.subscriptions || []);
                    return [3 /*break*/, 5];
                case 3:
                    err_24 = _a.sent();
                    setError(err_24.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingData(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleTopupSubscriptionBudget = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_25, raw, friendly, m, parsed;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.topupSubscriptionBudget(subBudgetUserId.trim(), parseFloat(subBudgetUsdAmount), subBudgetNotes || undefined, subBudgetForceTopup)];
                case 2:
                    res = _a.sent();
                    setSuccess("Subscription balance topped up for ".concat(subBudgetUserId, ": $").concat(subBudgetUsdAmount));
                    setSubBudgetUsdAmount('');
                    setSubBudgetNotes('');
                    setSubBudgetForceTopup(false);
                    return [3 /*break*/, 5];
                case 3:
                    err_25 = _a.sent();
                    raw = err_25.message || 'Top-up failed';
                    friendly = raw;
                    m = raw.match(/\{[\s\S]*\}/);
                    if (m) {
                        try {
                            parsed = JSON.parse(m[0]);
                            if (parsed === null || parsed === void 0 ? void 0 : parsed.detail)
                                friendly = String(parsed.detail);
                        }
                        catch ( /* keep raw */_b) { /* keep raw */ }
                    }
                    setError(friendly);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleSweepSubscriptionRollovers = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, moved, err_26;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.sweepSubscriptionRollovers(subSweepUserId.trim() || undefined)];
                case 2:
                    res = _a.sent();
                    moved = (res === null || res === void 0 ? void 0 : res.moved_usd) != null ? "$".concat(Number(res.moved_usd).toFixed(2)) : 'N/A';
                    setSuccess("Sweep complete. Moved: ".concat(moved));
                    return [3 /*break*/, 5];
                case 3:
                    err_26 = _a.sent();
                    setError(err_26.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleReapSubscriptionReservations = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, expired, periods, err_27;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.reapSubscriptionReservationsAll({
                            userId: subReapUserId.trim() || undefined,
                            limitPeriods: subReapLimitPeriods ? parseInt(subReapLimitPeriods) : undefined,
                            perPeriodLimit: subReapPerPeriodLimit ? parseInt(subReapPerPeriodLimit) : undefined,
                        })];
                case 2:
                    res = _a.sent();
                    expired = (res === null || res === void 0 ? void 0 : res.expired) != null ? Number(res.expired) : 0;
                    periods = (res === null || res === void 0 ? void 0 : res.periods_processed) != null ? Number(res.periods_processed) : 0;
                    setSuccess("Reaped ".concat(expired, " expired reservations across ").concat(periods, " period(s)."));
                    return [3 /*break*/, 5];
                case 3:
                    err_27 = _a.sent();
                    setError(err_27.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadSubscriptionPeriods = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var uid, res, err_28;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingHistory(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    uid = subHistoryUserId.trim();
                    if (!uid) {
                        setError('User ID is required to load subscription periods.');
                        return [2 /*return*/];
                    }
                    return [4 /*yield*/, api.listSubscriptionPeriods(uid, subHistoryStatus, 50, 0)];
                case 2:
                    res = _a.sent();
                    setSubPeriods(res.periods || []);
                    setSubLedger([]);
                    setSubSelectedPeriodKey('');
                    if (!res.periods || res.periods.length === 0) {
                        setSuccess('No subscription periods found for this user.');
                    }
                    return [3 /*break*/, 5];
                case 3:
                    err_28 = _a.sent();
                    setError(err_28.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingHistory(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadSubscriptionLedger = function (periodKey) { return __awaiter(void 0, void 0, void 0, function () {
        var uid, res, err_29;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!periodKey)
                        return [2 /*return*/];
                    clearMessages();
                    setLoadingHistory(true);
                    setSubSelectedPeriodKey(periodKey);
                    setSubLedger([]);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    uid = subHistoryUserId.trim();
                    if (!uid) {
                        setError('User ID is required to load ledger entries.');
                        return [2 /*return*/];
                    }
                    return [4 /*yield*/, api.listSubscriptionLedger(uid, periodKey, 200, 0)];
                case 2:
                    res = _a.sent();
                    setSubLedger(res.ledger || []);
                    if (!res.ledger || res.ledger.length === 0) {
                        setSuccess('No ledger entries for this period.');
                    }
                    return [3 /*break*/, 5];
                case 3:
                    err_29 = _a.sent();
                    setError(err_29.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingHistory(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleWalletRefund = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var usdVal, res, err_30;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    usdVal = walletRefundUsdAmount.trim() === '' ? null : parseFloat(walletRefundUsdAmount);
                    return [4 /*yield*/, api.refundWallet({
                            userId: walletRefundUserId.trim(),
                            paymentIntentId: walletRefundPaymentIntentId.trim(),
                            usdAmount: usdVal,
                            notes: walletRefundNotes || undefined
                        })];
                case 2:
                    res = _a.sent();
                    setSuccess(res.message || 'Refund requested; awaiting Stripe confirmation.');
                    setWalletRefundUsdAmount('');
                    setWalletRefundNotes('');
                    return [3 /*break*/, 5];
                case 3:
                    err_30 = _a.sent();
                    setError(err_30.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleCancelSubscription = function (e) { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_31;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    e.preventDefault();
                    clearMessages();
                    setLoadingAction(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.cancelSubscription({
                            userId: cancelSubUserId.trim() || undefined,
                            stripeSubscriptionId: cancelSubStripeId.trim() || undefined,
                            notes: cancelSubNotes || undefined,
                        })];
                case 2:
                    res = _a.sent();
                    setSuccess(res.message || 'Cancellation requested; awaiting Stripe confirmation.');
                    setCancelSubNotes('');
                    return [3 /*break*/, 5];
                case 3:
                    err_31 = _a.sent();
                    setError(err_31.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleStripeReconcile = function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_32;
        var _a, _b;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    clearMessages();
                    setLoadingAction(true);
                    _c.label = 1;
                case 1:
                    _c.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.reconcileStripe(stripeReconcileKind)];
                case 2:
                    res = _c.sent();
                    setSuccess("Stripe reconcile complete. Applied=".concat((_a = res.applied) !== null && _a !== void 0 ? _a : 0, ", Failed=").concat((_b = res.failed) !== null && _b !== void 0 ? _b : 0));
                    return [3 /*break*/, 5];
                case 3:
                    err_32 = _c.sent();
                    setError(err_32.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingAction(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadPendingStripe = function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_33;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingPendingStripe(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, api.listPendingStripeRequests(pendingStripeKind, 200, 0)];
                case 2:
                    res = _a.sent();
                    setPendingStripeItems(res.items || []);
                    if (!res.items || res.items.length === 0) {
                        setSuccess('No pending Stripe requests.');
                    }
                    return [3 /*break*/, 5];
                case 3:
                    err_33 = _a.sent();
                    setError(err_33.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingPendingStripe(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleLoadPendingEconomics = function () { return __awaiter(void 0, void 0, void 0, function () {
        var kind, userId, res, err_34;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    clearMessages();
                    setLoadingPendingEconomics(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    kind = pendingEconomicsKind.trim() || undefined;
                    userId = pendingEconomicsUserId.trim() || undefined;
                    return [4 /*yield*/, api.listPendingEconomicsEvents(kind, userId, 200, 0)];
                case 2:
                    res = _a.sent();
                    setPendingEconomicsItems(res.items || []);
                    if (!res.items || res.items.length === 0) {
                        setSuccess('No pending economics events.');
                    }
                    return [3 /*break*/, 5];
                case 3:
                    err_34 = _a.sent();
                    setError(err_34.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoadingPendingEconomics(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    if (configStatus === 'initializing') {
        return (<div className="min-h-screen bg-[#EEF5F5] flex items-center justify-center p-8">
                <Card className="max-w-lg w-full">
                    <CardBody className="text-center">
                        <LoadingSpinner />
                        <p className="mt-4 text-[#3A5672]">Initializing Economics…</p>
                    </CardBody>
                </Card>
            </div>);
    }
    var tabs = [
        { id: 'grantTrial', label: 'Grant Trial' },
        { id: 'updateTier', label: 'Override Tier Limits for User' },
        { id: 'lookup', label: 'Lookup Balance' },
        { id: 'quotaBreakdown', label: 'User Budget Breakdown' },
        { id: 'costByUser', label: 'Cost Report' },
        { id: 'quotaPolicies', label: 'Plan Limits' },
        { id: 'reservation', label: 'Reservation Floors' },
        { id: 'budgetPolicies', label: 'Project Budget Policies' },
        { id: 'lifetimeCredits', label: 'Lifetime Credits' },
        { id: 'appBudget', label: 'App Budget' },
        { id: 'plans', label: 'Plans' },
    ];
    return (<div className="h-screen overflow-hidden bg-[#EEF5F5]">
            <div className="flex h-full max-w-none flex-col gap-2 overflow-hidden px-4 py-3">
                {/* Header */}
                <div className="flex shrink-0 flex-wrap items-start justify-between gap-x-4 gap-y-1.5">
                    <DividerTitle title="Economics" subtitle="User quota policies, overrides, wallet credits, and application budget."/>

                    <div className="w-full max-w-xl xl:w-auto xl:min-w-[320px]">
                        <EconomicsOverview goTo={function (tabId) { clearMessages(); setViewMode(tabId); }}/>
                    </div>
                </div>

                {/* Navigation */}
                <div className="shrink-0 border-b border-[#E6F1F0] pb-2">
                    <Tabs active={viewMode} onChange={function (id) { clearMessages(); setViewMode(id); }} items={tabs}/>
                </div>

                {/* Messages */}
                {(success || error) && (<div className="shrink-0 space-y-1.5">
                        {success && <Callout tone="success" title="Success">{success}</Callout>}
                        {error && <Callout tone="warning" title="Action failed">{error}</Callout>}
                    </div>)}

                {/* Views */}
                <div className="relative min-h-0 w-full flex-1 overflow-hidden">
                    {/* Grant Trial */}
                    {viewMode === 'grantTrial' && (<div className="grid h-full min-h-0 content-start gap-3 overflow-y-auto xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
                            <Card className="h-fit">
                                <CardHeader title="Grant Trial (temporary plan override)" subtitle="Gives the user a higher plan envelope for a limited time. Overrides base plan limits — it does not add."/>
                                <CardBody>
                                    <form onSubmit={handleGrantTrial} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID *" value={trialUserId} onChange={function (e) { return setTrialUserId(e.target.value); }} placeholder="user123" required/>
                                            <Input label="Duration (days)" type="number" value={trialDays.toString()} onChange={function (e) { return setTrialDays(parseInt(e.target.value || '7')); }} min={1}/>
                                            <Input label="Requests / day" type="number" value={trialRequests.toString()} onChange={function (e) { return setTrialRequests(parseInt(e.target.value || '0')); }} min={1}/>
                                            <div>
                                                <Input label="Tokens / hour" type="number" value={trialTokensHour} onChange={function (e) { return setTrialTokensHour(e.target.value); }} min={1}/>
                                                {trialTokensHour && tokensToUsd(trialTokensHour) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(trialTokensHour)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / day" type="number" value={trialTokensDay} onChange={function (e) { return setTrialTokensDay(e.target.value); }} min={1}/>
                                                {trialTokensDay && tokensToUsd(trialTokensDay) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(trialTokensDay)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / month" type="number" value={trialTokensMonth} onChange={function (e) { return setTrialTokensMonth(e.target.value); }} min={1}/>
                                                {trialTokensMonth && tokensToUsd(trialTokensMonth) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(trialTokensMonth)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / hour" type="number" value={trialUsdHour} onChange={function (e) { return setTrialUsdHour(e.target.value); }} min={0} step="0.01"/>
                                                {trialUsdHour && usdToTokens(trialUsdHour) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(trialUsdHour)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / day" type="number" value={trialUsdDay} onChange={function (e) { return setTrialUsdDay(e.target.value); }} min={0} step="0.01"/>
                                                {trialUsdDay && usdToTokens(trialUsdDay) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(trialUsdDay)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / month" type="number" value={trialUsdMonth} onChange={function (e) { return setTrialUsdMonth(e.target.value); }} min={0} step="0.01"/>
                                                {trialUsdMonth && usdToTokens(trialUsdMonth) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(trialUsdMonth)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                        </div>

                                        <TextArea label="Notes" value={trialNotes} onChange={function (e) { return setTrialNotes(e.target.value); }} placeholder="Welcome trial for new user"/>

                                        <div className="flex items-center justify-end gap-3">
                                            <span className="text-[11.5px] text-[#7A99B0]">USD overrides tokens for the same window.</span>
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Granting…' : 'Grant Trial'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card className="h-fit">
                                <CardHeader title="What this does"/>
                                <CardBody className="space-y-2 text-[11.5px] leading-snug text-[#3A5672]">
                                    <p>Use for onboarding, marketing trials, or time-limited upgrades. Daily/monthly counters keep resetting while the override is active.</p>
                                    <p>All limit fields are overrides: they replace the base plan envelope while the trial is active.</p>
                                </CardBody>
                            </Card>
                        </div>)}

                    {/* Update Tier */}
                    {viewMode === 'updateTier' && (<div className="grid h-full min-h-0 content-start gap-3 overflow-y-auto xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
                            <Card className="h-fit">
                                <CardHeader title="Update Tier Override (partial updates)" subtitle="Only fields you provide are updated; empty fields keep their current value."/>
                                <CardBody>
                                    <form onSubmit={handleUpdateTierBudget} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID *" value={updateUserId} onChange={function (e) { return setUpdateUserId(e.target.value); }} placeholder="user456" required/>
                                            <Input label="Requests / day" type="number" value={updateRequestsDay} onChange={function (e) { return setUpdateRequestsDay(e.target.value); }} placeholder="100"/>
                                            <Input label="Requests / month" type="number" value={updateRequestsMonth} onChange={function (e) { return setUpdateRequestsMonth(e.target.value); }} placeholder="3000"/>
                                            <div>
                                                <Input label="Tokens / hour" type="number" value={updateTokensHour} onChange={function (e) { return setUpdateTokensHour(e.target.value); }} placeholder="500000"/>
                                                {updateTokensHour && tokensToUsd(updateTokensHour) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(updateTokensHour)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / day" type="number" value={updateTokensDay} onChange={function (e) { return setUpdateTokensDay(e.target.value); }} placeholder="10000000"/>
                                                {updateTokensDay && tokensToUsd(updateTokensDay) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(updateTokensDay)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / month" type="number" value={updateTokensMonth} onChange={function (e) { return setUpdateTokensMonth(e.target.value); }} placeholder="300000000"/>
                                                {updateTokensMonth && tokensToUsd(updateTokensMonth) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ ${Number(tokensToUsd(updateTokensMonth)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / hour" type="number" value={updateUsdHour} onChange={function (e) { return setUpdateUsdHour(e.target.value); }} placeholder="5" min={0} step="0.01"/>
                                                {updateUsdHour && usdToTokens(updateUsdHour) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(updateUsdHour)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / day" type="number" value={updateUsdDay} onChange={function (e) { return setUpdateUsdDay(e.target.value); }} placeholder="50" min={0} step="0.01"/>
                                                {updateUsdDay && usdToTokens(updateUsdDay) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(updateUsdDay)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / month" type="number" value={updateUsdMonth} onChange={function (e) { return setUpdateUsdMonth(e.target.value); }} placeholder="500" min={0} step="0.01"/>
                                                {updateUsdMonth && usdToTokens(updateUsdMonth) != null && (<div className="pt-0.5 text-[11px] text-[#7A99B0]">
                                                        ≈ {Number(usdToTokens(updateUsdMonth)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <Input label="Max concurrent" type="number" value={updateMaxConcurrent} onChange={function (e) { return setUpdateMaxConcurrent(e.target.value); }} placeholder="5"/>
                                            <Input label="Expires in days (empty = never)" type="number" value={updateExpiresDays} onChange={function (e) { return setUpdateExpiresDays(e.target.value); }} placeholder="30"/>
                                            <TextArea label="Notes" value={updateNotes} onChange={function (e) { return setUpdateNotes(e.target.value); }} placeholder="Promotional campaign / compensation / beta program"/>
                                        </div>

                                        <div className="flex items-center justify-end gap-3">
                                            <span className="text-[11.5px] text-[#7A99B0]">Empty fields keep the current override value.</span>
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Updating…' : 'Update Override'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card className="h-fit">
                                <CardHeader title="Override semantics"/>
                                <CardBody>
                                    <Callout tone="warning">
                                        This does <strong>not</strong> top-up the base plan. It replaces it for as long as the override is active.
                                    </Callout>
                                </CardBody>
                            </Card>
                        </div>)}

                    {/* Lookup */}
                    {viewMode === 'lookup' && (<div className="flex h-full min-h-0 flex-col gap-3">
                        <Card className="shrink-0">
                            <CardHeader title="Lookup User Balance" subtitle="Shows active plan override (if any) and purchased lifetime credits (if any)."/>
                            <CardBody>
                                <form onSubmit={handleLookupPlanBalance}>
                                    <div className="flex items-center gap-2.5">
                                        <Input value={lookupUserId} onChange={function (e) { return setLookupUserId(e.target.value); }} placeholder="user123" required className="max-w-sm flex-1"/>
                                        <Button type="submit" disabled={loadingAction}>
                                            {loadingAction ? 'Loading…' : 'Lookup'}
                                        </Button>
                                    </div>
                                </form>
                            </CardBody>
                        </Card>

                                {planBalance && (<Card className="flex min-h-0 flex-1 flex-col">
                                    <CardBody className="min-h-0 flex-1 space-y-3 overflow-y-auto">
                                        <Callout tone="info" title="How requests are funded">
                                            <strong>First:</strong> use as much effective plan quota as possible, funded by the project budget.{' '}
                                            <strong>Then:</strong> use wallet credits for any shortfall caused by quota or project budget limits.
                                            Concurrency and provider budgets are enforced separately.
                                        </Callout>
                                        <div>
                                            <div className="flex items-baseline justify-between flex-wrap gap-2">
                                                <h3 className="font-mono text-[12px] font-semibold text-[#0D1E2C]">
                                                    {planBalance.user_id}
                                                </h3>
                                                <div className="text-[11.5px] text-[#7A99B0]">
                                                    {planBalance.message || ''}
                                                </div>
                                            </div>

                                            {!planBalance.has_plan_override && !planBalance.has_lifetime_budget ? (<EmptyState message="No plan override and no purchased credits (base plan only)." icon="📋"/>) : (<div className="grid grid-cols-2 gap-2.5 mt-2">
                                                    {planBalance.has_plan_override && planBalance.plan_override && (<div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                            <div className="flex items-center justify-between">
                                                                <div>
                                                                    <div className="text-[12.5px] font-semibold text-[#10304B]">Tier Override</div>
                                                                    <div className="text-[11px] text-[#3A5672] mt-1">
                                                                        Replaces base plan while active
                                                                    </div>
                                                                </div>
                                                                <div className="text-lg">🎯</div>
                                                            </div>

                                                            <div className="mt-2 space-y-1 text-[12px]">
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Requests / day</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">{(_a = planBalance.plan_override.requests_per_day) !== null && _a !== void 0 ? _a : '—'}</span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Tokens / hour</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                                                        {(_c = (_b = planBalance.plan_override.tokens_per_hour) === null || _b === void 0 ? void 0 : _b.toLocaleString()) !== null && _c !== void 0 ? _c : '—'}
                                                                        {planBalance.plan_override.usd_per_hour != null ? " ($".concat(Number(planBalance.plan_override.usd_per_hour).toFixed(2), ")") : ''}
                                                                    </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Tokens / day</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                                                        {(_e = (_d = planBalance.plan_override.tokens_per_day) === null || _d === void 0 ? void 0 : _d.toLocaleString()) !== null && _e !== void 0 ? _e : '—'}
                                                                        {planBalance.plan_override.usd_per_day != null ? " ($".concat(Number(planBalance.plan_override.usd_per_day).toFixed(2), ")") : ''}
                                                                    </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Tokens / month</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                                                        {(_g = (_f = planBalance.plan_override.tokens_per_month) === null || _f === void 0 ? void 0 : _f.toLocaleString()) !== null && _g !== void 0 ? _g : '—'}
                                                                        {planBalance.plan_override.usd_per_month != null ? " ($".concat(Number(planBalance.plan_override.usd_per_month).toFixed(2), ")") : ''}
                                                                    </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Expires</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                    {planBalance.plan_override.expires_at
                            ? new Date(planBalance.plan_override.expires_at).toLocaleString()
                            : 'Never'}
                                  </span>
                                                                </div>
                                                                {planBalance.plan_override.notes && (<div className="pt-3 border-t border-[#E6F1F0] text-[11px] text-[#3A5672] italic">
                                                                        {planBalance.plan_override.notes}
                                                                    </div>)}
                                                                {planBalance.plan_override.reference_model && (<div className="pt-2 text-[11px] text-[#7A99B0]">
                                                                        Reference: <span className="font-mono">{planBalance.plan_override.reference_model}</span>
                                                                    </div>)}
                                                            </div>
                                                        </div>)}

                                                    {planBalance.has_lifetime_budget && planBalance.lifetime_budget && (<div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                            <div className="flex items-center justify-between">
                                                                <div>
                                                                    <div className="text-[12.5px] font-semibold text-[#10304B]">Lifetime Credits</div>
                                                                    <div className="text-[11px] text-[#3A5672] mt-1">
                                                                        Purchased tokens (do not reset)
                                                                    </div>
                                                                </div>
                                                                <div className="text-lg">💳</div>
                                                            </div>

                                                            <div className="mt-2 space-y-1 text-[12px]">
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Purchased</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                    ${Number(planBalance.lifetime_budget.purchased_usd || 0).toFixed(2)}
                                  </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Spent</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                    ${Number(planBalance.lifetime_budget.spent_usd || 0).toFixed(2)}
                                  </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Reserved (in-flight)</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                    ${Number(planBalance.lifetime_budget.reserved_usd || 0).toFixed(2)}
                                  </span>
                                                                </div>
                                                                <div className="flex justify-between gap-3">
                                                                    <span className="text-[#3A5672]">Available now</span>
                                                                    <span className="font-semibold text-[#0D1E2C]">
                                    ${Number(planBalance.lifetime_budget.available_usd || 0).toFixed(2)}
                                  </span>
                                                                </div>
                                                            </div>
                                                        </div>)}
                                                </div>)}
                                        </div>
                                    </CardBody>
                                    </Card>)}
                        </div>)}

                    {/* Quota Breakdown */}
                    {viewMode === 'quotaBreakdown' && (<div className="flex h-full min-h-0 flex-col gap-3">
                        <Card className="shrink-0">
                            <CardHeader title="Budget Breakdown" subtitle="Effective policy = base plan with the active override applied; remaining quota from current counters; wallet shown separately."/>
                            <CardBody>
                                <form onSubmit={handleGetQuotaBreakdown}>
                                    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2.5">
                                        <Input label="User ID *" value={breakdownUserId} onChange={function (e) { return setBreakdownUserId(e.target.value); }} placeholder="user123" required/>
                                        <Input label="App ID (optional)" value={breakdownBundleId} onChange={function (e) { return setBreakdownBundleId(e.target.value); }} placeholder="e.g. __project__ (global)"/>
                                        <Button type="submit" disabled={loadingAction}>
                                            {loadingAction ? 'Analyzing…' : 'Get Breakdown'}
                                        </Button>
                                    </div>
                                </form>
                            </CardBody>
                        </Card>

                                {quotaBreakdown && (<div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
                                        <div className="grid grid-cols-5 gap-2.5">
                                            <StatCard label="Plan" value={quotaBreakdown.plan_id || '—'} hint={quotaBreakdown.plan_source ? "source: ".concat(quotaBreakdown.plan_source) : undefined}/>
                                            <StatCard label="Requests today" value={"".concat(formatCount(quotaBreakdown.current_usage.requests_today), " / ").concat(formatCount(quotaBreakdown.effective_policy.requests_per_day))} hint={"remaining ".concat(formatCount(quotaBreakdown.remaining.requests_today))}/>
                                            <StatCard label="Plan today" value={"$".concat(Number(quotaBreakdown.current_usage.tokens_today_usd || 0).toFixed(2), " / ").concat(formatUsdLimit(quotaBreakdown.effective_policy.usd_per_day))} hint={"".concat(formatCount(quotaBreakdown.current_usage.tokens_today), " / ").concat(formatCount(quotaBreakdown.effective_policy.tokens_per_day), " tokens")}/>
                                            <StatCard label="Plan reserved" value={"$".concat(Number(quotaBreakdown.current_usage.tokens_reserved_usd || 0).toFixed(2))} hint={"".concat(formatCount(quotaBreakdown.current_usage.tokens_reserved || 0), " tokens held")}/>
                                            <StatCard label="Wallet available" value={quotaBreakdown.lifetime_credits ? "$".concat(Number(quotaBreakdown.lifetime_credits.available_usd || 0).toFixed(2)) : '$0.00'} hint={quotaBreakdown.lifetime_credits ? 'available' : 'no wallet record'}/>
                                        </div>

                                        <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                            <div className="flex items-center justify-between gap-3">
                                                <div>
                                                    <div className="text-[12.5px] font-semibold text-[#10304B]">Plan limits</div>
                                                    <div className="mt-1 text-[11px] text-[#3A5672]">Base plan, active override, and enforced policy.</div>
                                                </div>
                                                {quotaBreakdown.reference_model && (<div className="text-right text-[11px] text-[#7A99B0]">Reference: <span className="font-mono">{quotaBreakdown.reference_model}</span></div>)}
                                            </div>

                                            <div className="mt-2 grid grid-cols-3 gap-2.5">
                                                <div className="rounded-lg border border-[#E6F1F0] bg-white p-3">
                                                    <div className="mb-2 text-[12.5px] font-semibold text-[#10304B]">Base</div>
                                                    <PolicyMetricList policy={quotaBreakdown.base_policy}/>
                                                </div>

                                                <div className="rounded-lg border border-[#E6F1F0] bg-white p-3">
                                                    <div className="mb-2 flex items-center justify-between gap-2">
                                                        <span className="text-[12.5px] font-semibold text-[#10304B]">Override</span>
                                                        {quotaBreakdown.plan_override ? (quotaBreakdown.plan_override.active ? (<Pill tone="success">Active</Pill>) : quotaBreakdown.plan_override.expired ? (<Pill tone="warning">Expired</Pill>) : (<Pill tone="neutral">Inactive</Pill>)) : (<Pill tone="neutral">None</Pill>)}
                                                    </div>
                                                    {quotaBreakdown.plan_override ? (<>
                                                            <PolicyMetricList policy={quotaBreakdown.plan_override.limits}/>
                                                            <div className="mt-2 text-[11px] text-[#7A99B0]">
                                                                Expires: {quotaBreakdown.plan_override.expires_at ? new Date(quotaBreakdown.plan_override.expires_at).toLocaleString() : '—'}
                                                            </div>
                                                        </>) : (<div className="text-[11px] text-[#3A5672]">No user override is configured.</div>)}
                                                </div>

                                                <div className="rounded-lg border border-[#E6F1F0] bg-white p-3">
                                                    <div className="mb-2 text-[12.5px] font-semibold text-[#10304B]">Effective</div>
                                                    <PolicyMetricList policy={quotaBreakdown.effective_policy}/>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                            <div className="text-[12.5px] font-semibold text-[#10304B]">Plan quota now</div>
                                            <div className="mt-1 text-[11px] text-[#3A5672]">Used / limit and remaining capacity for the effective plan.</div>
                                            <div className="mt-2 grid grid-cols-2 gap-2">
                                                <CompactUsageRow label="Tokens / hour" used={quotaBreakdown.current_usage.tokens_this_hour || 0} limit={quotaBreakdown.effective_policy.tokens_per_hour} remaining={quotaBreakdown.remaining.tokens_this_hour} usedUsd={quotaBreakdown.current_usage.tokens_this_hour_usd} limitUsd={quotaBreakdown.effective_policy.usd_per_hour} remainingUsd={quotaBreakdown.remaining.tokens_this_hour_usd}/>
                                                <CompactUsageRow label="Requests / day" used={quotaBreakdown.current_usage.requests_today} limit={quotaBreakdown.effective_policy.requests_per_day} remaining={quotaBreakdown.remaining.requests_today}/>
                                                <CompactUsageRow label="Tokens / day" used={quotaBreakdown.current_usage.tokens_today} limit={quotaBreakdown.effective_policy.tokens_per_day} remaining={quotaBreakdown.remaining.tokens_today} usedUsd={quotaBreakdown.current_usage.tokens_today_usd} limitUsd={quotaBreakdown.effective_policy.usd_per_day} remainingUsd={quotaBreakdown.remaining.tokens_today_usd}/>
                                                <CompactUsageRow label="Requests / 30 days" used={quotaBreakdown.current_usage.requests_this_month} limit={quotaBreakdown.effective_policy.requests_per_month} remaining={quotaBreakdown.remaining.requests_this_month}/>
                                                <CompactUsageRow label="Tokens / 30 days" used={quotaBreakdown.current_usage.tokens_this_month} limit={quotaBreakdown.effective_policy.tokens_per_month} remaining={quotaBreakdown.remaining.tokens_this_month} usedUsd={quotaBreakdown.current_usage.tokens_this_month_usd} limitUsd={quotaBreakdown.effective_policy.usd_per_month} remainingUsd={quotaBreakdown.remaining.tokens_this_month_usd}/>
                                                <div className="rounded-lg border border-[rgba(245,158,11,0.4)] bg-[rgba(245,158,11,0.1)] px-2.5 py-1.5 text-[12px]">
                                                    <div className="flex items-center justify-between gap-3">
                                                        <span className="text-[#B45309]">Plan reserved</span>
                                                        <span className="font-semibold text-[#B45309]">{formatUsdLimit(quotaBreakdown.current_usage.tokens_reserved_usd)}</span>
                                                    </div>
                                                    <div className="mt-1 text-xs text-[#B45309]">
                                                        {formatCount(quotaBreakdown.current_usage.tokens_reserved || 0)} tokens held by in-flight requests
                                                    </div>
                                                </div>
                                                <div className="rounded-lg border border-[#E6F1F0] bg-white px-2.5 py-1.5 text-[12px]">
                                                    <div className="flex items-center justify-between gap-3">
                                                        <span className="text-[#3A5672]">Concurrent</span>
                                                        <span className="font-semibold text-[#0D1E2C]">
                                                            {formatCount(quotaBreakdown.current_usage.concurrent)} / {formatCount(quotaBreakdown.effective_policy.max_concurrent)}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {quotaBreakdown.reset_windows ? (<div className="rounded-xl border border-[#E6F1F0] bg-white p-3 text-[12px] text-[#3A5672]">
                                                <div className="font-semibold text-[#0D1E2C]">Rolling resets</div>
                                                <div className="mt-1 grid grid-cols-3 gap-1 text-[11px] text-[#3A5672]">
                                                    <div>App: <span className="font-mono text-[#0D1E2C]">{quotaBreakdown.reset_windows.bundle_id}</span></div>
                                                    <div>Hourly: {quotaBreakdown.reset_windows.hour_reset_at ? new Date(quotaBreakdown.reset_windows.hour_reset_at).toLocaleString() : '—'}</div>
                                                    <div>30-day: {quotaBreakdown.reset_windows.month_reset_at ? new Date(quotaBreakdown.reset_windows.month_reset_at).toLocaleString() : '—'}</div>
                                                </div>
                                            </div>) : (<div className="text-[11px] text-[#7A99B0]">
                                                Provide an App ID (use <code>__project__</code> for global quotas) to see rolling reset timestamps.
                                            </div>)}

                                        <div className="rounded-xl border border-[rgba(34,197,94,0.35)] bg-[rgba(34,197,94,0.08)] p-4">
                                            <div className="text-[12.5px] font-semibold text-[#15803D]">Wallet / personal credits</div>
                                            <div className="mt-1 text-xs text-[#15803D]">Separate from plan quota. Used for shortfall capacity.</div>
                                            {!quotaBreakdown.lifetime_credits ? (<div className="mt-2 text-[12px] text-[#15803D]">No wallet record for this user.</div>) : (<div className="mt-2 grid grid-cols-4 gap-2">
                                                    <div className="rounded-lg border border-[rgba(34,197,94,0.35)] bg-white px-2.5 py-1.5 text-[12px]">
                                                        <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#15803D]">Purchased</div>
                                                        <div className="mt-1 font-semibold text-[#15803D]">${Number(quotaBreakdown.lifetime_credits.purchased_usd || 0).toFixed(2)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-[rgba(34,197,94,0.35)] bg-white px-2.5 py-1.5 text-[12px]">
                                                        <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#15803D]">Spent</div>
                                                        <div className="mt-1 font-semibold text-[#15803D]">${Number(quotaBreakdown.lifetime_credits.spent_usd || 0).toFixed(2)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-[rgba(34,197,94,0.35)] bg-white px-2.5 py-1.5 text-[12px]">
                                                        <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#15803D]">Reserved</div>
                                                        <div className="mt-1 font-semibold text-[#15803D]">${Number(quotaBreakdown.lifetime_credits.reserved_usd || 0).toFixed(2)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-[rgba(34,197,94,0.35)] bg-white px-2.5 py-1.5 text-[12px]">
                                                        <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#15803D]">Available</div>
                                                        <div className="mt-1 font-semibold text-[#15803D]">${Number(quotaBreakdown.lifetime_credits.available_usd || 0).toFixed(2)}</div>
                                                    </div>
                                                </div>)}
                                        </div>

                                        <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <div className="text-[12.5px] font-semibold text-[#10304B]">Subscription balance</div>
                                                    <div className="mt-1 text-[11px] text-[#3A5672]">Per-period subscription credits</div>
                                                </div>
                                            </div>

                                            {!quotaBreakdown.subscription_balance ? (<div className="mt-2 text-[12px] text-[#3A5672]">
                                                    No subscription balance record for this user.
                                                </div>) : (<div className="mt-2 space-y-2 text-[12px]">
                                                    <div className="grid grid-cols-2 gap-1.5 text-[11px] text-[#3A5672]">
                                                        {quotaBreakdown.subscription_balance.plan_id && (<div>
                                                                plan: <span className="font-mono text-[12px] font-semibold text-[#0D1E2C]">{quotaBreakdown.subscription_balance.plan_id}</span>
                                                            </div>)}
                                                        {quotaBreakdown.subscription_balance.status && (<div>
                                                                status: <span className="font-semibold text-[#0D1E2C]">{quotaBreakdown.subscription_balance.status}</span>
                                                            </div>)}
                                                        {quotaBreakdown.subscription_balance.provider && (<div>
                                                                provider: <span className="font-semibold text-[#0D1E2C]">{providerLabel(quotaBreakdown.subscription_balance.provider)}</span>
                                                            </div>)}
                                                        {quotaBreakdown.subscription_balance.monthly_price_cents != null && (<div>
                                                                monthly price: <span className="font-semibold text-[#0D1E2C]">
                                                                    ${Number(quotaBreakdown.subscription_balance.monthly_price_cents / 100).toFixed(2)}
                                                                </span>
                                                            </div>)}
                                                    </div>

                                                    {quotaBreakdown.subscription_balance.period_start && quotaBreakdown.subscription_balance.period_end && (<div className="text-[11px] text-[#3A5672]">
                                                            Period: {formatDateTime(quotaBreakdown.subscription_balance.period_start)} → {formatDateTime(quotaBreakdown.subscription_balance.period_end)}
                                                        </div>)}
                                                    {quotaBreakdown.subscription_balance.period_status && (<div className="text-[11px] text-[#3A5672]">
                                                            Period status: {quotaBreakdown.subscription_balance.period_status}
                                                        </div>)}

                                                    <div className="grid grid-cols-3 gap-2 text-[12px]">
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Balance</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(quotaBreakdown.subscription_balance.balance_usd || 0).toFixed(2)}
                                                            </div>
                                                            {quotaBreakdown.subscription_balance.balance_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(quotaBreakdown.subscription_balance.balance_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Reserved</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(quotaBreakdown.subscription_balance.reserved_usd || 0).toFixed(2)}
                                                            </div>
                                                            {quotaBreakdown.subscription_balance.reserved_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(quotaBreakdown.subscription_balance.reserved_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Available</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(quotaBreakdown.subscription_balance.available_usd || 0).toFixed(2)}
                                                            </div>
                                                            {quotaBreakdown.subscription_balance.available_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(quotaBreakdown.subscription_balance.available_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                    </div>

                                                    <div className="grid grid-cols-3 gap-2 text-[12px]">
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Period top-up</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number((_j = (_h = quotaBreakdown.subscription_balance.topup_usd) !== null && _h !== void 0 ? _h : quotaBreakdown.subscription_balance.lifetime_added_usd) !== null && _j !== void 0 ? _j : 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Period spent</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number((_l = (_k = quotaBreakdown.subscription_balance.spent_usd) !== null && _k !== void 0 ? _k : quotaBreakdown.subscription_balance.lifetime_spent_usd) !== null && _l !== void 0 ? _l : 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Rolled over</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(quotaBreakdown.subscription_balance.rolled_over_usd || 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <div className="pt-2 text-[11px] text-[#3A5672]">
                                                        Reference: <span className="font-mono">{quotaBreakdown.subscription_balance.reference_model || (economicsRef ? "".concat(economicsRef.reference_provider, "/").concat(economicsRef.reference_model) : '')}</span>
                                                    </div>
                                                </div>)}
                                        </div>

                                        {/* Reservations table */}
                                        {((_m = quotaBreakdown.active_reservations) === null || _m === void 0 ? void 0 : _m.length) > 0 && (<div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                <div className="text-[12.5px] font-semibold text-[#10304B] mb-2">Active credit reservations</div>
                                                <div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                                    <table className="w-full text-[12px]">
                                                        <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                        <tr>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Reservation</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">App</th>
                                                            <th className="px-2.5 py-1.5 text-right font-bold">Reserved (USD)</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Expires</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Notes</th>
                                                        </tr>
                                                        </thead>
                                                        <tbody className="divide-y divide-[#E6F1F0]">
                                                        {quotaBreakdown.active_reservations.map(function (r) {
                        var _a, _b;
                        return (<tr key={r.reservation_id} className="hover:bg-white transition-colors">
                                                                <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{r.reservation_id}</td>
                                                                <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{(_a = r.bundle_id) !== null && _a !== void 0 ? _a : '—'}</td>
                                                                <td className="px-2.5 py-1.5 text-right text-[#3A5672]">${Number(r.reserved_usd || 0).toFixed(2)}</td>
                                                                <td className="px-2.5 py-1.5 text-[#3A5672]">{r.expires_at ? new Date(r.expires_at).toLocaleString() : '—'}</td>
                                                                <td className="px-2.5 py-1.5 text-[#3A5672]">{(_b = r.notes) !== null && _b !== void 0 ? _b : '—'}</td>
                                                            </tr>);
                    })}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>)}
                                    </div>)}
                        </div>)}

                    {/* Cost report — true per-model spend from OPEX aggregates */}
                    {viewMode === 'costByUser' && (<div className="flex h-full min-h-0 flex-col gap-3">
                            <Card className="flex min-h-0 flex-1 flex-col">
                                <CardHeader title="Cost report" subtitle="Actual spend priced per model from the live price table, grouped by user, agent, or app. User Budget Breakdown shows quota-equivalent dollars; the absorption report shows absorbed shortfall — three different numbers by design." action={<div className="flex gap-1.5">
                                            <Button variant="secondary" onClick={function () { return handleLoadCostReport(); }} disabled={loadingCost}>
                                                {loadingCost ? 'Loading…' : 'Run report'}
                                            </Button>
                                            <Button variant="secondary" onClick={handleRebuildAggregates} disabled={loadingCost}>
                                                Rebuild aggregates
                                            </Button>
                                            <Button variant="secondary" onClick={handleExportCostCsv} disabled={loadingCost || !costRowsFiltered.length}>
                                                Export CSV
                                            </Button>
                                        </div>}/>
                                <CardBody className="flex min-h-0 flex-1 flex-col gap-2.5">
                                    <div className="grid shrink-0 grid-cols-4 gap-2.5 max-w-3xl">
                                        <Select label="Group by" value={costDim} onChange={function (e) { return handleCostDimChange(e.target.value); }}>
                                            <option value="user">User</option>
                                            <option value="agent">Agent</option>
                                            <option value="app">App</option>
                                        </Select>
                                        <Input label="From" type="date" value={costFrom} max={costTo || undefined} onChange={function (e) { return setCostFrom(e.target.value); }}/>
                                        <Input label="To" type="date" value={costTo} min={costFrom || undefined} onChange={function (e) { return setCostTo(e.target.value); }}/>
                                        <Input label="Filter ids" value={costFilter} placeholder="comma-separated, substring ok" onChange={function (e) { return setCostFilter(e.target.value); }}/>
                                    </div>

                                    {loadingCost ? (<LoadingSpinner />) : !costLoaded ? (<EmptyState message="Pick a grouping and window, then run the report." icon="🧾"/>) : !costRowsFiltered.length ? (<EmptyState message="No usage matches this window and filter." icon="🧾"/>) : (<>
                                            <div className="grid shrink-0 grid-cols-3 gap-2 max-w-xl">
                                                <StatCard label="Total spend" value={"$".concat(costRowsFiltered.reduce(function (s, r) { return s + r.costUsd; }, 0).toFixed(4))}/>
                                                <StatCard label={costDim === 'user' ? 'Users' : costDim === 'agent' ? 'Agents' : 'Apps'} value={costDim === 'user' ? costUserRows.length : costRowsFiltered.length} hint={costSystemRows.length ? "+ ".concat(costSystemRows.length, " system") : undefined}/>
                                                <StatCard label="Events" value={costRowsFiltered.reduce(function (s, r) { return s + r.events; }, 0)}/>
                                            </div>
                                            <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-[#E6F1F0]">
                                                <table className="w-full text-[12px]">
                                                    <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                        <tr>
                                                            <th className="px-2.5 py-1.5 text-left">{costDim === 'user' ? 'User' : costDim === 'agent' ? 'Agent' : 'App'}</th>
                                                            <th className="px-2.5 py-1.5 text-right">Cost (USD)</th>
                                                            <th className="px-2.5 py-1.5 text-right">Input tokens</th>
                                                            <th className="px-2.5 py-1.5 text-right">Output tokens</th>
                                                            <th className="px-2.5 py-1.5 text-right">Events</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {costUserRows.concat(costSystemRows).map(function (row, idx) { return (<react_1.default.Fragment key={row.id}>
                                                                {costSystemRows.length > 0 && idx === costUserRows.length && (<tr className="border-b border-[#E6F1F0] bg-[#FBF7EF]">
                                                                        <td colSpan={5} className="px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-[0.1em] text-[#B07C1B]">
                                                                            System principals — app runs without an attributed identity
                                                                        </td>
                                                                    </tr>)}
                                                                <tr className="cursor-pointer border-b border-[#F0F6F5] hover:bg-[#FAFCFC]" onClick={function () {
                        return setCostExpanded(function (prev) {
                            var _a;
                            return (__assign(__assign({}, prev), (_a = {}, _a[row.id] = !prev[row.id], _a)));
                        });
                    }}>
                                                                    <td className="px-2.5 py-1.5 font-mono text-[11.5px] text-[#0D1E2C]">
                                                                        <span className="mr-1 inline-block w-3 text-[#7A99B0]">
                                                                            {costExpanded[row.id] ? '▾' : '▸'}
                                                                        </span>
                                                                        {row.id}
                                                                        {row.system && <span className="ml-1.5 text-[10px] font-semibold text-[#B07C1B]">system</span>}
                                                                    </td>
                                                                    <td className="px-2.5 py-1.5 text-right font-mono font-semibold text-[#0D1E2C]">
                                                                        ${row.costUsd.toFixed(4)}
                                                                    </td>
                                                                    <td className="px-2.5 py-1.5 text-right font-mono text-[#3A5672]">{row.inputTokens.toLocaleString()}</td>
                                                                    <td className="px-2.5 py-1.5 text-right font-mono text-[#3A5672]">{row.outputTokens.toLocaleString()}</td>
                                                                    <td className="px-2.5 py-1.5 text-right font-mono text-[#3A5672]">{row.events.toLocaleString()}</td>
                                                                </tr>
                                                                {costExpanded[row.id] &&
                        row.byModel.map(function (line, i) { return (<tr key={"".concat(row.id, "-").concat(i)} className="border-b border-[#F0F6F5] bg-[#FAFCFC]">
                                                                            <td className="py-1 pl-9 pr-2.5 font-mono text-[11px] text-[#3A5672]">
                                                                                {line.service}
                                                                                {line.provider ? " \u00B7 ".concat(line.provider) : ''}
                                                                                {line.model ? " \u00B7 ".concat(line.model) : ''}
                                                                            </td>
                                                                            <td className="px-2.5 py-1 text-right font-mono text-[11px] text-[#3A5672]">
                                                                                ${Number(line.cost_usd || 0).toFixed(4)}
                                                                            </td>
                                                                            <td colSpan={3}></td>
                                                                        </tr>); })}
                                                            </react_1.default.Fragment>); })}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </>)}
                                </CardBody>
                            </Card>
                        </div>)}

                    {/* Quota Policies */}
                    {viewMode === 'quotaPolicies' && (<div className="grid h-full min-h-0 gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                            <Card className="flex min-h-0 flex-col">
                                <CardHeader title="Set Plan Policy" subtitle="Base limits per plan_id (global for tenant/project). No bundle_id." action={economicsRef ? (<span className="text-[11px] text-[#7A99B0]">
                                            Ref: <span className="font-mono">{economicsRef.reference_provider}/{economicsRef.reference_model}</span>
                                        </span>) : undefined}/>
                                <CardBody className="min-h-0 flex-1 space-y-3 overflow-y-auto">
                                    <form onSubmit={handleSetQuotaPolicy} className="space-y-3">
                                        <div className="grid grid-cols-2 gap-2.5">
                                            <Select label="Plan ID *" value={policyPlanId} onChange={function (e) { return setPolicyPlanId(e.target.value); }} options={PLAN_OPTIONS}/>
                                            {policyPlanId === 'custom' && (<Input label="Custom plan_id *" value={policyPlanIdCustom} onChange={function (e) { return setPolicyPlanIdCustom(e.target.value); }} placeholder="e.g. enterprise-plan" required/>)}
                                        </div>

                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="Max concurrent" type="number" value={policyMaxConcurrent} onChange={function (e) { return setPolicyMaxConcurrent(e.target.value); }} placeholder="1"/>
                                            <Input label="Requests / day" type="number" value={policyRequestsDay} onChange={function (e) { return setPolicyRequestsDay(e.target.value); }} placeholder="10"/>
                                            <Input label="Requests / month" type="number" value={policyRequestsMonth} onChange={function (e) { return setPolicyRequestsMonth(e.target.value); }} placeholder="300"/>
                                            <div>
                                                <Input label="Tokens / hour" type="number" value={policyTokensHour} onChange={function (e) { return setPolicyTokensHour(e.target.value); }} placeholder="500000"/>
                                                {policyTokensHour && tokensToUsd(policyTokensHour) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ ${Number(tokensToUsd(policyTokensHour)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / day" type="number" value={policyTokensDay} onChange={function (e) { return setPolicyTokensDay(e.target.value); }} placeholder="1000000"/>
                                                {policyTokensDay && tokensToUsd(policyTokensDay) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ ${Number(tokensToUsd(policyTokensDay)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="Tokens / month" type="number" value={policyTokensMonth} onChange={function (e) { return setPolicyTokensMonth(e.target.value); }} placeholder="30000000"/>
                                                {policyTokensMonth && tokensToUsd(policyTokensMonth) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ ${Number(tokensToUsd(policyTokensMonth)).toFixed(2)}
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / hour" type="number" value={policyUsdHour} onChange={function (e) { return setPolicyUsdHour(e.target.value); }} placeholder="5" min={0} step="0.01"/>
                                                {policyUsdHour && usdToTokens(policyUsdHour) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ {Number(usdToTokens(policyUsdHour)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / day" type="number" value={policyUsdDay} onChange={function (e) { return setPolicyUsdDay(e.target.value); }} placeholder="50" min={0} step="0.01"/>
                                                {policyUsdDay && usdToTokens(policyUsdDay) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ {Number(usdToTokens(policyUsdDay)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                            <div>
                                                <Input label="USD / month" type="number" value={policyUsdMonth} onChange={function (e) { return setPolicyUsdMonth(e.target.value); }} placeholder="500" min={0} step="0.01"/>
                                                {policyUsdMonth && usdToTokens(policyUsdMonth) != null && (<div className="text-[11px] text-[#7A99B0] pt-1">
                                                        ≈ {Number(usdToTokens(policyUsdMonth)).toLocaleString()} tokens
                                                    </div>)}
                                            </div>
                                        </div>

                                        <TextArea label="Notes" value={policyNotes} onChange={function (e) { return setPolicyNotes(e.target.value); }} placeholder="Free plan limits (global per tenant/project)"/>

                                        <div className="flex items-center justify-end gap-3">
                                            <span className="text-[11.5px] text-[#7A99B0]">USD overrides tokens for the same window.</span>
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Saving…' : 'Save Policy'}
                                            </Button>
                                        </div>
                                    </form>

                                    <Details title="How plan policies work">
                                        <p>This is the default quota envelope for a plan (free/wallet/admin). Daily is calendar day, hourly is a rolling 60‑minute window, and monthly is a rolling 30‑day window (anchored to first usage per app).</p>
                                        <p>The platform reservation floor is set in the <span className="font-semibold">Reservation Floors</span> tab
                                        (stored in the economics descriptor, picked up live). An app can still override per surface via
                                        app props <span className="font-mono"> economics.reservation.&lt;floor&gt;</span>.</p>
                                    </Details>
                                </CardBody>
                            </Card>

                            <Card className="flex min-h-0 flex-col">
                                <CardHeader title="Current Plan Quota Policies" subtitle={"".concat(quotaPolicies.length, " policy records")}/>
                                <CardBody className="flex min-h-0 flex-1 flex-col">
                                    {loadingData ? (<LoadingSpinner />) : quotaPolicies.length === 0 ? (<EmptyState message="No plan policies configured." icon="📋"/>) : (<div className="min-h-0 flex-1 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Plan ID</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Max concurrent</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Req/day</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Req/month</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Tok/hour</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Tok/day</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Tok/month</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/hour</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/day</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/month</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Notes</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {quotaPolicies.map(function (policy, idx) {
                    var _a, _b, _c, _d, _e, _f, _g, _h, _j;
                    return (<tr key={idx} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{policy.plan_id}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_a = policy.max_concurrent) !== null && _a !== void 0 ? _a : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_b = policy.requests_per_day) !== null && _b !== void 0 ? _b : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_c = policy.requests_per_month) !== null && _c !== void 0 ? _c : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_e = (_d = policy.tokens_per_hour) === null || _d === void 0 ? void 0 : _d.toLocaleString()) !== null && _e !== void 0 ? _e : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_g = (_f = policy.tokens_per_day) === null || _f === void 0 ? void 0 : _f.toLocaleString()) !== null && _g !== void 0 ? _g : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">{(_j = (_h = policy.tokens_per_month) === null || _h === void 0 ? void 0 : _h.toLocaleString()) !== null && _j !== void 0 ? _j : '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_hour != null ? "$".concat(Number(policy.usd_per_hour).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_day != null ? "$".concat(Number(policy.usd_per_day).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_month != null ? "$".concat(Number(policy.usd_per_month).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{policy.notes || '—'}</td>
                                                    </tr>);
                })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                    {quotaPolicies.length > 0 && quotaPolicies[0].reference_model && (<div className="pt-3 text-[11px] text-[#7A99B0]">
                                            Reference: <span className="font-mono">{quotaPolicies[0].reference_model}</span>
                                        </div>)}
                                </CardBody>
                            </Card>
                        </div>)}

                    {viewMode === 'reservation' && (<div className="grid h-full min-h-0 gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                            <Card className="h-fit">
                                <CardHeader title="Reservation Floors" subtitle="Platform default per-turn reservation floor (USD). Read live by apps each turn."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleSetReservation} className="space-y-3">
                                        <div className="grid grid-cols-2 gap-2.5">
                                            <Input label="Surface" value={reservationFloor} onChange={function (e) { return setReservationFloor(e.target.value); }} placeholder="chat"/>
                                            <Input label="Amount (USD) · ≤ 0 disables" type="number" value={reservationAmount} onChange={function (e) { return setReservationAmount(e.target.value); }} placeholder="2.0"/>
                                        </div>
                                        <div className="flex justify-end">
                                            <Button type="submit" disabled={loadingAction}>
                                                Save reservation floor
                                            </Button>
                                        </div>
                                    </form>

                                    <Details title="How reservation floors work">
                                        <p>The reservation floor is the minimum USD reserved before a turn runs. It is stored in the
                                        economics descriptor (not the database) and is picked up live by running apps. A value of
                                        <span className="font-mono"> 0 or less disables</span> the floor (estimate falls back to tokens).
                                        An app can override per surface via app props
                                        <span className="font-mono"> economics.reservation.&lt;floor&gt;</span>; omitting it inherits this default.
                                        Only <span className="font-mono">chat</span> is consumed today.</p>
                                    </Details>
                                </CardBody>
                            </Card>

                            <Card className="flex min-h-0 flex-col">
                                <CardHeader title="Current floors" subtitle={"".concat(Object.keys(reservation).length, " surface(s)")}/>
                                <CardBody className="flex min-h-0 flex-1 flex-col">
                                    {loadingData ? (<LoadingSpinner />) : Object.keys(reservation).length === 0 ? (<EmptyState message="No reservation floors configured." icon="🪙"/>) : (<div className="min-h-0 flex-1 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Surface</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Floor (USD)</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">State</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Actions</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {Object.entries(reservation).map(function (_a) {
                    var floor = _a[0], amount = _a[1];
                    return (<tr key={floor} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{floor}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {Number(amount) > 0 ? "$".concat(Number(amount).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            {Number(amount) > 0 ? 'enabled' : 'disabled'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-right">
                                                            <Button variant="danger" onClick={function () { return handleDeleteReservation(floor); }} disabled={loadingAction}>
                                                                Remove
                                                            </Button>
                                                        </td>
                                                    </tr>);
                })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>
                        </div>)}

                    {/* Budget Policies */}
                    {viewMode === 'budgetPolicies' && (<div className="grid h-full min-h-0 gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                            <Card className="h-fit">
                                <CardHeader title="Set Provider Budget Policy" subtitle="Hard per-provider spending ceiling for the tenant/project (no bundle_id)."/>
                                <CardBody>
                                    <form onSubmit={handleSetBudgetPolicy} className="space-y-3">
                                        <Input label="Provider *" value={budgetProvider} onChange={function (e) { return setBudgetProvider(e.target.value); }} placeholder="anthropic" required/>

                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="USD / hour" type="number" step="0.01" value={budgetUsdHour} onChange={function (e) { return setBudgetUsdHour(e.target.value); }} placeholder="10.00"/>
                                            <Input label="USD / day" type="number" step="0.01" value={budgetUsdDay} onChange={function (e) { return setBudgetUsdDay(e.target.value); }} placeholder="200.00"/>
                                            <Input label="USD / month" type="number" step="0.01" value={budgetUsdMonth} onChange={function (e) { return setBudgetUsdMonth(e.target.value); }} placeholder="5000.00"/>
                                        </div>

                                        <TextArea label="Notes" value={budgetNotes} onChange={function (e) { return setBudgetNotes(e.target.value); }} placeholder="Daily spending limit for provider"/>

                                        <div className="flex items-center justify-end gap-3">
                                            <span className="text-[11.5px] text-[#7A99B0]">Hard ceiling against runaway provider costs.</span>
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Saving…' : 'Save Budget Policy'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card className="flex min-h-0 flex-col">
                                <CardHeader title="Current Budget Policies" subtitle={"".concat(budgetPolicies.length, " policy records")}/>
                                <CardBody className="flex min-h-0 flex-1 flex-col">
                                    {loadingData ? (<LoadingSpinner />) : budgetPolicies.length === 0 ? (<EmptyState message="No budget policies configured." icon="💵"/>) : (<div className="min-h-0 flex-1 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Provider</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/hour</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/day</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">USD/month</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Notes</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {budgetPolicies.map(function (policy, idx) { return (<tr key={idx} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{policy.provider}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_hour != null ? "$".concat(policy.usd_per_hour.toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_day != null ? "$".concat(policy.usd_per_day.toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            {policy.usd_per_month != null ? "$".concat(policy.usd_per_month.toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{policy.notes || '—'}</td>
                                                    </tr>); })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>
                        </div>)}

                    {/* Lifetime Credits */}
                    {viewMode === 'lifetimeCredits' && (<div className="grid h-full min-h-0 content-start gap-3 overflow-y-auto xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
                            <Card className="h-fit">
                                <CardHeader title="Lifetime Credits (USD → tokens)" subtitle="One-time purchase adds tokens until depleted. These do not reset. Quoted using the backend reference model."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleAddLifetimeCredits} className="space-y-3">
                                        <div className="grid grid-cols-2 gap-2.5">
                                            <Input label="User ID *" value={lifetimeUserId} onChange={function (e) { return setLifetimeUserId(e.target.value); }} placeholder="user123" required/>
                                            <Input label="Amount (USD) *" type="number" step="0.01" value={lifetimeUsdAmount} onChange={function (e) { return setLifetimeUsdAmount(e.target.value); }} placeholder="10.00" required/>
                                        </div>

                                        <TextArea label="Purchase Notes" value={lifetimeNotes} onChange={function (e) { return setLifetimeNotes(e.target.value); }} placeholder="Stripe payment ID / invoice / manual purchase note"/>

                                        <div className="flex flex-wrap items-center justify-between gap-2.5">
                                            <div className="text-[11.5px] text-[#7A99B0]">
                                                Reference model: <span className="font-mono font-semibold text-[#0D1E2C]">{economicsRef ? "".concat(economicsRef.reference_provider, "/").concat(economicsRef.reference_model) : '…'}</span>
                                            </div>
                                            <div className="flex gap-2">
                                                <Button type="button" variant="secondary" onClick={function () { return handleCheckLifetimeBalance(new Event('submit')); }} disabled={loadingAction || !lifetimeUserId.trim()}>
                                                    Check Balance
                                                </Button>
                                                <Button type="submit" disabled={loadingAction}>
                                                    {loadingAction ? 'Processing…' : 'Add Credits'}
                                                </Button>
                                            </div>
                                        </div>
                                    </form>

                                    <Callout tone="info" title="Quick interpretation">
                                        “Balance tokens” is what the user can spend. If balance drops below the admission threshold, the system may block paid usage.
                                    </Callout>
                                </CardBody>
                            </Card>

                            <div className="flex min-h-0 flex-col gap-3">
                                {lifetimeBalance && (<Card className="shrink-0">
                                        <CardHeader title={"Current Balance: ".concat(lifetimeBalance.user_id)}/>
                                        <CardBody>
                                            {lifetimeBalance.has_purchased_credits ? (<StatCard label="Balance" value={"$".concat(Number(lifetimeBalance.balance_usd || 0).toFixed(2))}/>) : (<EmptyState message="No purchased credits found. This user operates on plan quotas only." icon="💳"/>)}
                                        </CardBody>
                                    </Card>)}

                                <Card className="h-fit">
                                    <CardHeader title="What the USD conversion means" subtitle="Purchases are quoted with a fixed reference model so USD→tokens is predictable."/>
                                    <CardBody>
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <div className="rounded-lg border border-[#E6F1F0] bg-[#F6FAFA] px-2.5 py-2">
                                                <div className="text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">Example</div>
                                                <div className="mt-0.5 font-mono text-[15px] font-semibold text-[#0D1E2C]">$5.00</div>
                                                <div className="mt-0.5 text-[11px] text-[#7A99B0]">Reference model rate</div>
                                            </div>
                                            <div className="rounded-lg border border-[#E6F1F0] bg-[#F6FAFA] px-2.5 py-2">
                                                <div className="text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">Example</div>
                                                <div className="mt-0.5 font-mono text-[15px] font-semibold text-[#0D1E2C]">$10.00</div>
                                                <div className="mt-0.5 text-[11px] text-[#7A99B0]">Reference model rate</div>
                                            </div>
                                            <div className="rounded-lg border border-[#E6F1F0] bg-[#F6FAFA] px-2.5 py-2">
                                                <div className="text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">Example</div>
                                                <div className="mt-0.5 font-mono text-[15px] font-semibold text-[#0D1E2C]">$50.00</div>
                                                <div className="mt-0.5 text-[11px] text-[#7A99B0]">Reference model rate</div>
                                            </div>
                                        </div>
                                    </CardBody>
                                </Card>
                            </div>
                        </div>)}

                    {/* App Budget */}
                    {viewMode === 'appBudget' && (<div className="flex h-full min-h-0 flex-col gap-3">
                            {loadingData ? (<Card className="shrink-0"><CardBody><LoadingSpinner /></CardBody></Card>) : !appBudget ? (<Card className="shrink-0"><CardBody><EmptyState message="No budget data loaded." icon="💰"/></CardBody></Card>) : (<div className="grid shrink-0 grid-cols-4 gap-2 xl:grid-cols-8">
                                    <StatCard label="Current balance" value={"$".concat(Number(appBudget.balance.balance_usd || 0).toFixed(2))}/>
                                    <StatCard label="Lifetime added" value={"$".concat(Number(appBudget.balance.lifetime_added_usd || 0).toFixed(2))}/>
                                    <StatCard label="Lifetime spent" value={"$".concat(Number(appBudget.balance.lifetime_spent_usd || 0).toFixed(2))}/>
                                    <StatCard label="Overdraft limit" value={appBudget.balance.overdraft_limit_usd == null
                    ? 'Unlimited'
                    : "$".concat(Number(appBudget.balance.overdraft_limit_usd).toFixed(2))}/>
                                    {appBudget.balance.available_usd != null && (<StatCard label="Available" value={"$".concat(Number(appBudget.balance.available_usd).toFixed(2))}/>)}
                                    <StatCard label="Spend this hour" value={"$".concat(Number(((_o = appBudget.current_month_spending) === null || _o === void 0 ? void 0 : _o.hour) || 0).toFixed(2))}/>
                                    <StatCard label="Spend today" value={"$".concat(Number(((_p = appBudget.current_month_spending) === null || _p === void 0 ? void 0 : _p.day) || 0).toFixed(2))}/>
                                    <StatCard label="Spend this month" value={"$".concat(Number(((_q = appBudget.current_month_spending) === null || _q === void 0 ? void 0 : _q.month) || 0).toFixed(2))}/>
                                </div>)}

                            <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-2">
                                <div className="flex min-h-0 flex-col gap-3">
                                    {!loadingData && appBudget && (<>

                                            {appBudget.by_bundle && Object.keys(appBudget.by_bundle).length > 0 && (<Card className="shrink-0">
                                                    <CardHeader title="Spending by app"/>
                                                    <CardBody className="max-h-36 space-y-1.5 overflow-y-auto">
                                                        {Object.entries(appBudget.by_bundle).map(function (_a) {
                        var bundleId = _a[0], spending = _a[1];
                        return (<div key={bundleId} className="flex items-center justify-between gap-2 rounded-lg border border-[#E6F1F0] bg-white px-2.5 py-1.5">
                                                                <div className="truncate font-mono text-[12px] font-semibold text-[#0D1E2C]">{bundleId}</div>
                                                                <div className="flex shrink-0 flex-wrap gap-3 text-[11.5px] text-[#3A5672]">
                                                                    <span>Hour: <strong className="font-mono text-[#0D1E2C]">${Number(spending.hour || 0).toFixed(2)}</strong></span>
                                                                    <span>Day: <strong className="font-mono text-[#0D1E2C]">${Number(spending.day || 0).toFixed(2)}</strong></span>
                                                                    <span>Month: <strong className="font-mono text-[#0D1E2C]">${Number(spending.month || 0).toFixed(2)}</strong></span>
                                                                </div>
                                                            </div>);
                    })}
                                                    </CardBody>
                                                </Card>)}

                                            <Card className="flex min-h-0 flex-1 flex-col">
                                                <CardHeader title="Budget absorption report" subtitle="Project budget absorbs shortfalls when plan + wallet can’t cover actual spend." action={<div className="flex gap-1.5">
                                                            <Button variant="secondary" onClick={handleLoadAbsorptionReport} disabled={loadingAbsorption}>
                                                                {loadingAbsorption ? 'Loading…' : 'Run report'}
                                                            </Button>
                                                            <Button variant="secondary" onClick={handleExportAbsorptionCsv} disabled={loadingAbsorption}>
                                                                {loadingAbsorption ? 'Exporting…' : 'Export CSV'}
                                                            </Button>
                                                        </div>}/>
                                                <CardBody className="flex min-h-0 flex-1 flex-col gap-2.5">
                                                <div className="grid shrink-0 grid-cols-3 gap-2.5">
                                                    <Select label="Period" value={absorptionPeriod} onChange={function (e) { return setAbsorptionPeriod(e.target.value); }}>
                                                        <option value="day">Daily</option>
                                                        <option value="month">Monthly</option>
                                                    </Select>
                                                    <Select label="Group by" value={absorptionGroupBy} onChange={function (e) { return setAbsorptionGroupBy(e.target.value); }}>
                                                        <option value="none">None</option>
                                                        <option value="user">User</option>
                                                        <option value="bundle">App</option>
                                                    </Select>
                                                    <Input label="Lookback (days)" type="number" min={1} max={730} value={absorptionDays} onChange={function (e) { return setAbsorptionDays(e.target.value); }}/>
                                                </div>

                                                {loadingAbsorption ? (<LoadingSpinner />) : absorptionItems.length === 0 ? (<EmptyState message="No absorption events found." icon="🧾"/>) : (<div className="min-h-0 flex-1 overflow-auto rounded-lg border border-[#E6F1F0]">
                                                        <table className="min-w-full text-[12px] text-left">
                                                            <thead className="sticky top-0 z-10 bg-[#F6FAFA] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                <tr>
                                                                    <th className="px-2.5 py-1.5">Period</th>
                                                                    {absorptionGroupBy !== 'none' && (<th className="px-2.5 py-1.5">{absorptionGroupBy === 'user' ? 'User' : 'App'}</th>)}
                                                                    <th className="px-2.5 py-1.5">Total absorbed</th>
                                                                    <th className="px-2.5 py-1.5">Subscription shortfall</th>
                                                                    <th className="px-2.5 py-1.5">Wallet plan shortfall</th>
                                                                    <th className="px-2.5 py-1.5">Subscription overage</th>
                                                                    <th className="px-2.5 py-1.5">Free plan overage</th>
                                                                    <th className="px-2.5 py-1.5">Events</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody className="text-[#3A5672]">
                                                                {absorptionItems.map(function (row, idx) { return (<tr key={"".concat(row.period_start, "-").concat(idx)} className="border-t border-[#E6F1F0]">
                                                                        <td className="px-2.5 py-1.5">{new Date(row.period_start).toLocaleString()}</td>
                                                                        {absorptionGroupBy !== 'none' && (<td className="px-2.5 py-1.5 font-mono text-[12px]">{row.group_key || '-'}</td>)}
                                                                        <td className="px-2.5 py-1.5">${row.total_shortfall_usd.toFixed(2)}</td>
                                                                        <td className="px-2.5 py-1.5">${row.wallet_subscription_shortfall_usd.toFixed(2)}</td>
                                                                        <td className="px-2.5 py-1.5">${row.wallet_plan_shortfall_usd.toFixed(2)}</td>
                                                                        <td className="px-2.5 py-1.5">${row.subscription_overage_shortfall_usd.toFixed(2)}</td>
                                                                        <td className="px-2.5 py-1.5">${row.free_plan_shortfall_usd.toFixed(2)}</td>
                                                                        <td className="px-2.5 py-1.5">{row.events}</td>
                                                                    </tr>); })}
                                                            </tbody>
                                                        </table>
                                                    </div>)}
                                                </CardBody>
                                            </Card>

                                            <Details title="Money journey (who pays, in order)">
                                                <p>The application budget is the master budget for the tenant/project; it also explains why the project budget can go negative.</p>
                                                    <pre className="text-[11px] leading-relaxed text-[#3A5672] bg-[#F6FAFA] border border-[#E6F1F0] rounded-lg p-2.5 whitespace-pre-wrap">
                {"External (Stripe) subscription \u2014 primary: subscription budget\nsubscription budget -> wallet overflow -> project absorbs\nactual over reserved: subscription headroom, then project (shortfall:subscription_overage)\nwallet short: project absorbs (shortfall:wallet_subscription)\n\nProject-funded (free, internal subscription, registered) \u2014 primary: project budget\nproject budget -> wallet overflow -> project absorbs\nuncovered: project absorbs (shortfall:wallet_plan with wallet, shortfall:free_plan without)\n\nWallet is always overflow, never a primary source.\n\nShortfall ledger notes:\n- shortfall:subscription_overage\n- shortfall:wallet_subscription\n- shortfall:wallet_plan\n- shortfall:free_plan"}
                                                    </pre>
                                            </Details>
                                        </>)}
                                </div>

                                <div className="flex min-h-0 flex-col gap-3">
                                    {!loadingData && appBudget && (<Card className="flex min-h-0 flex-1 flex-col">
                                                <CardHeader title="Request lineage (per request_id)" subtitle="Trace the full money journey for a single turn. request_id == turn_id."/>
                                                <CardBody className="flex min-h-0 flex-1 flex-col gap-2.5">
                                                    <div className="grid shrink-0 grid-cols-[minmax(0,1fr)_auto_auto] items-end gap-2.5">
                                                        <Input label="request_id (turn_id) *" value={lineageRequestId} onChange={function (e) { return setLineageRequestId(e.target.value); }} placeholder="turn_..."/>
                                                        <Button variant="secondary" onClick={handleLoadRequestLineage} disabled={loadingLineage}>
                                                            {loadingLineage ? 'Loading…' : 'Lookup'}
                                                        </Button>
                                                        <Button variant="secondary" onClick={handleCopyRequestId}>
                                                            Copy request_id
                                                        </Button>
                                                    </div>

                                                    {loadingLineage ? (<LoadingSpinner />) : lineageResult ? (<div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto">
                                                            <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                                <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0] mb-2">Project budget reservations</div>
                                                                {((_s = (_r = lineageResult.project_budget) === null || _r === void 0 ? void 0 : _r.reservations) === null || _s === void 0 ? void 0 : _s.length) ? (<table className="min-w-full text-xs text-left">
                                                                        <thead className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                            <tr>
                                                                                <th className="py-1 pr-3">Reservation</th>
                                                                                <th className="py-1 pr-3">Amount</th>
                                                                                <th className="py-1 pr-3">Actual</th>
                                                                                <th className="py-1 pr-3">Status</th>
                                                                                <th className="py-1 pr-3">Created</th>
                                                                                <th className="py-1 pr-3">Expires</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="text-[#3A5672]">
                                                                            {lineageResult.project_budget.reservations.map(function (r, i) { return (<tr key={"pr-".concat(i)} className="border-t border-[#E6F1F0]">
                                                                                    <td className="py-1 pr-3 font-mono">{r.reservation_id}</td>
                                                                                    <td className="py-1 pr-3">{formatUsdFromCents(r.amount_cents)}</td>
                                                                                    <td className="py-1 pr-3">{formatUsdFromCents(r.actual_spent_cents)}</td>
                                                                                    <td className="py-1 pr-3">{r.status}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.created_at)}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.expires_at)}</td>
                                                                                </tr>); })}
                                                                        </tbody>
                                                                    </table>) : (<EmptyState message="No project budget reservations found." icon="🧾"/>)}
                                                            </div>

                                                            <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                                <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0] mb-2">Project budget ledger</div>
                                                                {((_u = (_t = lineageResult.project_budget) === null || _t === void 0 ? void 0 : _t.ledger) === null || _u === void 0 ? void 0 : _u.length) ? (<table className="min-w-full text-xs text-left">
                                                                        <thead className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                            <tr>
                                                                                <th className="py-1 pr-3">ID</th>
                                                                                <th className="py-1 pr-3">Amount</th>
                                                                                <th className="py-1 pr-3">Kind</th>
                                                                                <th className="py-1 pr-3">Note</th>
                                                                                <th className="py-1 pr-3">Created</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="text-[#3A5672]">
                                                                            {lineageResult.project_budget.ledger.map(function (r, i) { return (<tr key={"pl-".concat(i)} className="border-t border-[#E6F1F0]">
                                                                                    <td className="py-1 pr-3">{r.id}</td>
                                                                                    <td className="py-1 pr-3">{formatUsd(r.amount_usd)}</td>
                                                                                    <td className="py-1 pr-3">{r.kind}</td>
                                                                                    <td className="py-1 pr-3">{r.note || '-'}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.created_at)}</td>
                                                                                </tr>); })}
                                                                        </tbody>
                                                                    </table>) : (<EmptyState message="No project ledger rows found." icon="🧾"/>)}
                                                            </div>

                                                            <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                                <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0] mb-2">Subscription reservations</div>
                                                                {((_w = (_v = lineageResult.subscription_budget) === null || _v === void 0 ? void 0 : _v.reservations) === null || _w === void 0 ? void 0 : _w.length) ? (<table className="min-w-full text-xs text-left">
                                                                        <thead className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                            <tr>
                                                                                <th className="py-1 pr-3">Reservation</th>
                                                                                <th className="py-1 pr-3">Period</th>
                                                                                <th className="py-1 pr-3">Amount</th>
                                                                                <th className="py-1 pr-3">Actual</th>
                                                                                <th className="py-1 pr-3">Status</th>
                                                                                <th className="py-1 pr-3">Created</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="text-[#3A5672]">
                                                                            {lineageResult.subscription_budget.reservations.map(function (r, i) { return (<tr key={"sr-".concat(i)} className="border-t border-[#E6F1F0]">
                                                                                    <td className="py-1 pr-3 font-mono">{r.reservation_id}</td>
                                                                                    <td className="py-1 pr-3 font-mono">{r.period_key}</td>
                                                                                    <td className="py-1 pr-3">{formatUsdFromCents(r.amount_cents)}</td>
                                                                                    <td className="py-1 pr-3">{formatUsdFromCents(r.actual_spent_cents)}</td>
                                                                                    <td className="py-1 pr-3">{r.status}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.created_at)}</td>
                                                                                </tr>); })}
                                                                        </tbody>
                                                                    </table>) : (<EmptyState message="No subscription reservations found." icon="🧾"/>)}
                                                            </div>

                                                            <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                                <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0] mb-2">Subscription ledger</div>
                                                                {((_y = (_x = lineageResult.subscription_budget) === null || _x === void 0 ? void 0 : _x.ledger) === null || _y === void 0 ? void 0 : _y.length) ? (<table className="min-w-full text-xs text-left">
                                                                        <thead className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                            <tr>
                                                                                <th className="py-1 pr-3">ID</th>
                                                                                <th className="py-1 pr-3">Period</th>
                                                                                <th className="py-1 pr-3">Amount</th>
                                                                                <th className="py-1 pr-3">Kind</th>
                                                                                <th className="py-1 pr-3">Note</th>
                                                                                <th className="py-1 pr-3">Created</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="text-[#3A5672]">
                                                                            {lineageResult.subscription_budget.ledger.map(function (r, i) { return (<tr key={"sl-".concat(i)} className="border-t border-[#E6F1F0]">
                                                                                    <td className="py-1 pr-3">{r.id}</td>
                                                                                    <td className="py-1 pr-3 font-mono">{r.period_key}</td>
                                                                                    <td className="py-1 pr-3">{formatUsd(r.amount_usd)}</td>
                                                                                    <td className="py-1 pr-3">{r.kind}</td>
                                                                                    <td className="py-1 pr-3">{r.note || '-'}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.created_at)}</td>
                                                                                </tr>); })}
                                                                        </tbody>
                                                                    </table>) : (<EmptyState message="No subscription ledger rows found." icon="🧾"/>)}
                                                            </div>

                                                            <div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3">
                                                                <div className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0] mb-2">Wallet reservations</div>
                                                                {((_0 = (_z = lineageResult.wallet) === null || _z === void 0 ? void 0 : _z.reservations) === null || _0 === void 0 ? void 0 : _0.length) ? (<table className="min-w-full text-xs text-left">
                                                                        <thead className="text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                                            <tr>
                                                                                <th className="py-1 pr-3">Reservation</th>
                                                                                <th className="py-1 pr-3">Reserved (USD)</th>
                                                                                <th className="py-1 pr-3">Spent (USD)</th>
                                                                                <th className="py-1 pr-3">Status</th>
                                                                                <th className="py-1 pr-3">Created</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="text-[#3A5672]">
                                                                            {lineageResult.wallet.reservations.map(function (r, i) { return (<tr key={"wr-".concat(i)} className="border-t border-[#E6F1F0]">
                                                                                    <td className="py-1 pr-3 font-mono">{r.reservation_id}</td>
                                                                                    <td className="py-1 pr-3">${Number(r.reserved_usd || 0).toFixed(2)}</td>
                                                                                    <td className="py-1 pr-3">{r.spent_usd == null ? '—' : "$".concat(Number(r.spent_usd).toFixed(2))}</td>
                                                                                    <td className="py-1 pr-3">{r.status}</td>
                                                                                    <td className="py-1 pr-3">{formatDate(r.created_at)}</td>
                                                                                </tr>); })}
                                                                        </tbody>
                                                                    </table>) : (<EmptyState message="No wallet reservations found." icon="🧾"/>)}
                                                            </div>
                                                        </div>) : (<EmptyState message="No request lineage loaded." icon="🔎"/>)}
                                                </CardBody>
                                            </Card>)}

                                    <Card className="shrink-0">
                                        <CardHeader title="Top up application budget" subtitle="Adds funds to the tenant/project wallet."/>
                                        <CardBody>
                                            <form onSubmit={handleTopupAppBudget}>
                                                <div className="grid grid-cols-[140px_minmax(0,1fr)_auto] items-end gap-2.5">
                                                    <Input label="Amount (USD) *" type="number" step="0.01" value={appBudgetTopup} onChange={function (e) { return setAppBudgetTopup(e.target.value); }} placeholder="100.00" required/>
                                                    <TextArea label="Notes" rows={1} value={appBudgetNotes} onChange={function (e) { return setAppBudgetNotes(e.target.value); }} placeholder="Monthly budget allocation"/>
                                                    <Button type="submit" disabled={loadingAction}>
                                                        {loadingAction ? 'Processing…' : 'Add funds'}
                                                    </Button>
                                                </div>
                                                <p className="mt-1.5 text-[11.5px] text-[#7A99B0]">Keep enough budget when company-funding plan usage to prevent service disruption.</p>
                                            </form>
                                        </CardBody>
                                    </Card>

                                    <Details title="Budget flow examples">
                                        <Callout tone="info" title="Scenario: plan-funded usage">
                                            User operates within effective plan limits → request allowed → company budget is charged (typical policy).
                                        </Callout>
                                        <Callout tone="success" title="Scenario: user-funded fallback">
                                            User exceeds plan → purchased credits present → user credits are charged → app budget not used.
                                        </Callout>
                                        <Callout tone="warning" title="Scenario: mixed / policy-dependent">
                                            Some flows may split charges depending on limiter policy and reservations (in-flight holds).
                                        </Callout>
                                    </Details>
                                </div>
                            </div>
                        </div>)}
                    {/* Subscriptions */}
                    {viewMode === 'plans' && (<div className="grid h-full min-h-0 gap-3 xl:grid-cols-2">
                            <div className="min-h-0 space-y-3 overflow-y-auto pr-1">
                            <Card className="shrink-0">
                                <CardHeader title="Plans" subtitle="Define plan_id → price mapping (internal or Stripe). Plan IDs drive quota policies." action={<Button variant="secondary" onClick={handleLoadSubscriptionPlans} disabled={loadingPlans || loadingData}>
                                            {(loadingPlans || loadingData) ? 'Loading…' : 'Refresh'}
                                        </Button>}/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleUpsertSubscriptionPlan} className="space-y-3">
                                        <div className="grid grid-cols-4 gap-2.5">
                                            <Input label="Plan ID *" value={planId} onChange={function (e) { return setPlanId(e.target.value); }} placeholder="wallet" required/>
                                            <Select label="Provider" value={planProvider} onChange={function (e) { return setPlanProvider(e.target.value); }} options={[
                { value: 'internal', label: 'internal' },
                { value: 'stripe', label: 'stripe' },
            ]}/>
                                            <Input label="Monthly price (cents) *" type="number" value={planPriceCents} onChange={function (e) { return setPlanPriceCents(e.target.value); }} placeholder="2000" min={0} required/>
                                            <div className="flex flex-col">
                                                <label className="mb-1 block text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#7A99B0]">Active</label>
                                                <div className="flex h-8 items-center gap-2">
                                                    <input type="checkbox" checked={planActive} onChange={function (e) { return setPlanActive(e.target.checked); }} className="h-4 w-4 rounded border-[#D8ECEB] text-[#01BEB2] focus:ring-[#01BEB2]/30"/>
                                                    <span className="text-[12px] text-[#3A5672]">{planActive ? 'enabled' : 'disabled'}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {planProvider === 'stripe' && (<Input label="stripe_price_id *" value={planStripePriceId} onChange={function (e) { return setPlanStripePriceId(e.target.value); }} placeholder="price_..." required/>)}

                                        <TextArea label="Notes" value={planNotes} onChange={function (e) { return setPlanNotes(e.target.value); }} placeholder="Plan description, intended plan, or internal notes"/>

                                        <div className="flex justify-end">
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Saving…' : 'Save Plan'}
                                            </Button>
                                        </div>
                                    </form>

                                    {(loadingPlans || loadingData) ? (<LoadingSpinner />) : subscriptionPlans.length === 0 ? (<EmptyState message="No plans configured yet." icon="🧾"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Plan ID</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Provider</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Stripe price</th>
                                                    <th className="px-2.5 py-1.5 text-right font-bold">Monthly price</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Active</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Updated</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Notes</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {subscriptionPlans.map(function (plan) { return (<tr key={"".concat(plan.tenant, ":").concat(plan.project, ":").concat(plan.plan_id)} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{plan.plan_id}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{plan.provider}</td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{plan.stripe_price_id || '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-right text-[#3A5672]">
                                                            ${(Number(plan.monthly_price_cents || 0) / 100).toFixed(2)} ({plan.monthly_price_cents}¢)
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{plan.active ? 'yes' : 'no'}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(plan.updated_at)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{plan.notes || '—'}</td>
                                                    </tr>); })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Create Subscription" subtitle="Creates an internal or Stripe subscription using a plan_id."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleCreateSubscription} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Select label="Provider" value={subProvider} onChange={function (e) { return setSubProvider(e.target.value); }} options={[
                { value: 'internal', label: 'Manual' },
                { value: 'stripe', label: 'Stripe' },
            ]}/>
                                            <Input label="User ID *" value={subUserId} onChange={function (e) { return setSubUserId(e.target.value); }} placeholder="user123" required/>
                                            <Input label="Plan ID *" value={subPlanId} onChange={function (e) { return setSubPlanId(e.target.value); }} placeholder="plan_basic" list="subscription-plan-options" required/>
                                        </div>
                                        <datalist id="subscription-plan-options">
                                            {subscriptionPlans.map(function (plan) { return (<option key={plan.plan_id} value={plan.plan_id}>
                                                    {plan.plan_id}{plan.active ? '' : ' (inactive)'}
                                                </option>); })}
                                        </datalist>

                                        {subProvider === 'stripe' && (<div className="grid grid-cols-3 gap-2.5">
                                                <Input label="stripe_price_id (optional override)" value={subStripePriceId} onChange={function (e) { return setSubStripePriceId(e.target.value); }} placeholder="price_..."/>
                                                <Input label="stripe_customer_id (optional)" value={subStripeCustomerId} onChange={function (e) { return setSubStripeCustomerId(e.target.value); }} placeholder="cus_..."/>
                                                <Input label="monthly_price_cents_hint (optional)" type="number" value={subPriceHint} onChange={function (e) { return setSubPriceHint(e.target.value); }} placeholder="2000"/>
                                            </div>)}

                                        <div className="flex justify-end">
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Creating…' : 'Create Subscription'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Lookup Subscription (by user)" subtitle="Shows the current subscription row stored in user_plans."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleLookupSubscription} className="space-y-3">
                                        <div className="flex items-end gap-2.5">
                                            <Input value={subLookupUserId} onChange={function (e) { return setSubLookupUserId(e.target.value); }} placeholder="user123" required className="flex-1"/>
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Loading…' : 'Lookup'}
                                            </Button>
                                        </div>
                                    </form>

                                    {subscription && (<div className="rounded-xl border border-[#E6F1F0] bg-[#F6FAFA] p-3 text-[12px] space-y-2">
                                            <div className="flex items-center justify-between">
                                                <div className="font-semibold text-[#0D1E2C]">Subscription</div>
                                                <DuePill sub={subscription}/>
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">plan_id</span>
                                                    <strong className="font-mono text-[12px]">{subscription.plan_id || '—'}</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">billing</span>
                                                    <strong>{providerLabel(subscription.provider)}</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">status</span>
                                                    <strong>{subscription.status}</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">monthly price</span>
                                                    <strong>${(Number(subscription.monthly_price_cents || 0) / 100).toFixed(2)} ({subscription.monthly_price_cents}¢)</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">started</span>
                                                    <strong>{formatDateTime(subscription.started_at)}</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">last charge</span>
                                                    <strong>{formatDateTime(subscription.last_charged_at)}</strong>
                                                </div>

                                                <div className="flex justify-between">
                                                    <span className="text-[#3A5672]">next charge</span>
                                                    <strong>{formatDateTime(subscription.next_charge_at)}</strong>
                                                </div>

                                                {subscription.provider === 'stripe' && (<>
                                                        <div className="flex justify-between">
                                                            <span className="text-[#3A5672]">stripe_customer_id</span>
                                                            <strong className="font-mono text-[12px]">{subscription.stripe_customer_id || '—'}</strong>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span className="text-[#3A5672]">stripe_subscription_id</span>
                                                            <strong className="font-mono text-[12px]">{subscription.stripe_subscription_id || '—'}</strong>
                                                        </div>
                                                    </>)}
                                            </div>

                                            {subscriptionBalance && (<div className="pt-2.5 border-t border-[#E6F1F0] space-y-2">
                                                    <div className="text-[12.5px] font-semibold text-[#10304B]">Subscription balance</div>
                                                    <div className="text-[11px] text-[#3A5672]">
                                                        Reference: <span className="font-mono">{subscriptionBalance.reference_model || (economicsRef ? "".concat(economicsRef.reference_provider, "/").concat(economicsRef.reference_model) : '')}</span>
                                                    </div>
                                                    {subscriptionBalance.period_start && subscriptionBalance.period_end && (<div className="text-[11px] text-[#3A5672]">
                                                            Period: {formatDateTime(subscriptionBalance.period_start)} → {formatDateTime(subscriptionBalance.period_end)}
                                                        </div>)}

                                                    <div className="grid grid-cols-3 gap-2 text-[12px]">
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Balance</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(subscriptionBalance.balance_usd || 0).toFixed(2)}
                                                            </div>
                                                            {subscriptionBalance.balance_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(subscriptionBalance.balance_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Reserved</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(subscriptionBalance.reserved_usd || 0).toFixed(2)}
                                                            </div>
                                                            {subscriptionBalance.reserved_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(subscriptionBalance.reserved_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Available</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(subscriptionBalance.available_usd || 0).toFixed(2)}
                                                            </div>
                                                            {subscriptionBalance.available_tokens != null && (<div className="text-[11px] text-[#7A99B0]">
                                                                    {Number(subscriptionBalance.available_tokens).toLocaleString()} tokens
                                                                </div>)}
                                                        </div>
                                                    </div>

                                                    <div className="grid grid-cols-3 gap-2 text-[12px]">
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Period top-up</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number((_2 = (_1 = subscriptionBalance.topup_usd) !== null && _1 !== void 0 ? _1 : subscriptionBalance.lifetime_added_usd) !== null && _2 !== void 0 ? _2 : 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Period spent</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number((_4 = (_3 = subscriptionBalance.spent_usd) !== null && _3 !== void 0 ? _3 : subscriptionBalance.lifetime_spent_usd) !== null && _4 !== void 0 ? _4 : 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                        <div className="rounded-lg bg-white border border-[#E6F1F0] p-2">
                                                            <div className="text-[#3A5672]">Rolled over</div>
                                                            <div className="font-semibold text-[#0D1E2C]">
                                                                ${Number(subscriptionBalance.rolled_over_usd || 0).toFixed(2)}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>)}

                                            {/* Internal ops */}
                                            {subscription.provider === 'internal' &&
                    subscription.status === 'active' && (<div className="pt-2.5 border-t border-[#E6F1F0] flex flex-wrap items-center justify-between gap-3">
                                                        <div className="text-[11px] text-[#3A5672]">
                                                            Internal plans draw from the project budget bounded by quota. Reset re-anchors the month + day windows and clears hour buckets so all rolling counters start fresh.
                                                        </div>

                                                        <Button type="button" variant="secondary" disabled={loadingAction} onClick={function () { return __awaiter(void 0, void 0, void 0, function () {
                        var res, fresh, err_35;
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0:
                                    clearMessages();
                                    setLoadingAction(true);
                                    _a.label = 1;
                                case 1:
                                    _a.trys.push([1, 4, 5, 6]);
                                    return [4 /*yield*/, api.resetInternalQuota({ userId: subscription.user_id })];
                                case 2:
                                    res = _a.sent();
                                    setSuccess(res.message || "Reset quota for ".concat(subscription.user_id));
                                    return [4 /*yield*/, api.getSubscription(subscription.user_id)];
                                case 3:
                                    fresh = _a.sent();
                                    setSubscription(fresh.subscription);
                                    setSubscriptionBalance(fresh.subscription_balance || null);
                                    return [3 /*break*/, 6];
                                case 4:
                                    err_35 = _a.sent();
                                    setError(err_35.message);
                                    return [3 /*break*/, 6];
                                case 5:
                                    setLoadingAction(false);
                                    return [7 /*endfinally*/];
                                case 6: return [2 /*return*/];
                            }
                        });
                    }); }}>
                                                            {loadingAction ? 'Resetting…' : 'Reset quota'}
                                                        </Button>
                                                    </div>)}
                                        </div>)}
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Subscription Balance Admin" subtitle="Manual top-ups for a user's subscription balance."/>
                                <CardBody className="space-y-3">
                                    <div className="text-[11px] text-[#3A5672]">
                                        Manual top-ups apply to external subscription balances. Internal plans have no
                                        balance — use “Reset quota” in the lookup card to refresh their rolling quota windows.
                                    </div>
                                    <form onSubmit={handleTopupSubscriptionBudget} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID *" value={subBudgetUserId} onChange={function (e) { return setSubBudgetUserId(e.target.value); }} placeholder="user123" required/>
                                            <Input label="Top-up USD *" type="number" value={subBudgetUsdAmount} onChange={function (e) { return setSubBudgetUsdAmount(e.target.value); }} placeholder="50" required/>
                                            <Input label="Notes" value={subBudgetNotes} onChange={function (e) { return setSubBudgetNotes(e.target.value); }} placeholder="Optional notes"/>
                                        </div>
                                        <label className="flex items-center gap-2 text-[12px] text-[#3A5672]">
                                            <input type="checkbox" checked={subBudgetForceTopup} onChange={function (e) { return setSubBudgetForceTopup(e.target.checked); }} className="h-4 w-4 rounded border-[#D8ECEB] text-[#01BEB2] focus:ring-[#01BEB2]/30"/>
                                            Force topup (allow multiple in the same billing period)
                                        </label>
                                        <div className="flex justify-end">
                                            <Button type="submit" disabled={loadingAction}>
                                                {loadingAction ? 'Processing…' : 'Top-up Subscription Balance'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Wallet Refund (Stripe)" subtitle="Refund a Stripe payment_intent. Credits are removed immediately; finalization happens via Stripe webhook."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleWalletRefund} className="space-y-3">
                                        <div className="grid grid-cols-4 gap-2.5">
                                            <Input label="User ID *" value={walletRefundUserId} onChange={function (e) { return setWalletRefundUserId(e.target.value); }} placeholder="user123" required/>
                                            <Input label="Payment Intent ID *" value={walletRefundPaymentIntentId} onChange={function (e) { return setWalletRefundPaymentIntentId(e.target.value); }} placeholder="pi_..." required/>
                                            <Input label="Refund USD (blank = full)" type="number" value={walletRefundUsdAmount} onChange={function (e) { return setWalletRefundUsdAmount(e.target.value); }} placeholder="25.00"/>
                                            <Input label="Notes" value={walletRefundNotes} onChange={function (e) { return setWalletRefundNotes(e.target.value); }} placeholder="Optional notes"/>
                                        </div>
                                        <div className="flex justify-end">
                                            <Button type="submit" variant="danger" disabled={loadingAction}>
                                                {loadingAction ? 'Processing…' : 'Request Refund'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Cancel Stripe Subscription" subtitle="Request cancellation at period end (current balance remains usable)."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleCancelSubscription} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID" value={cancelSubUserId} onChange={function (e) { return setCancelSubUserId(e.target.value); }} placeholder="user123"/>
                                            <Input label="Stripe Subscription ID" value={cancelSubStripeId} onChange={function (e) { return setCancelSubStripeId(e.target.value); }} placeholder="sub_..."/>
                                            <Input label="Notes" value={cancelSubNotes} onChange={function (e) { return setCancelSubNotes(e.target.value); }} placeholder="Optional notes"/>
                                        </div>
                                        <div className="flex justify-end">
                                            <Button type="submit" variant="secondary" disabled={loadingAction}>
                                                {loadingAction ? 'Submitting…' : 'Request Cancellation'}
                                            </Button>
                                        </div>
                                    </form>
                                    <div className="text-[11.5px] text-[#7A99B0]">
                                        Provide either User ID or Stripe Subscription ID.
                                    </div>
                                </CardBody>
                            </Card>
                            </div>

                            <div className="min-h-0 space-y-3 overflow-y-auto pr-1">
                            <Card className="shrink-0">
                                <CardHeader title="Stripe Reconcile" subtitle="Check pending Stripe refund/cancel requests if a webhook was missed."/>
                                <CardBody className="space-y-3">
                                    <div className="grid grid-cols-3 gap-2.5 items-end">
                                        <Select label="Kind" value={stripeReconcileKind} onChange={function (e) { return setStripeReconcileKind(e.target.value); }} options={[
                { value: 'all', label: 'all' },
                { value: 'wallet_refund', label: 'wallet_refund' },
                { value: 'subscription_cancel', label: 'subscription_cancel' },
            ]}/>
                                        <div className="md:col-span-2">
                                            <Button type="button" variant="secondary" disabled={loadingAction} onClick={handleStripeReconcile}>
                                                {loadingAction ? 'Reconciling…' : 'Run Reconcile'}
                                            </Button>
                                        </div>
                                    </div>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Pending Stripe Requests" subtitle="Audit view for pending refunds/cancellations." action={<Button variant="secondary" onClick={handleLoadPendingStripe} disabled={loadingPendingStripe}>
                                            {loadingPendingStripe ? 'Loading…' : 'Refresh'}
                                        </Button>}/>
                                <CardBody className="space-y-3">
                                    <div className="grid grid-cols-3 gap-2.5">
                                        <Select label="Kind filter" value={pendingStripeKind} onChange={function (e) { return setPendingStripeKind(e.target.value); }} options={[
                { value: 'all', label: 'all' },
                { value: 'wallet_refund', label: 'wallet_refund' },
                { value: 'subscription_cancel', label: 'subscription_cancel' },
            ]}/>
                                    </div>

                                    {pendingStripeItems.length === 0 ? (<EmptyState message="No pending Stripe requests loaded." icon="🧾"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Kind</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">User</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Amount</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Tokens</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Stripe ID</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Open</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">External ID</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Created</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {pendingStripeItems.map(function (p) {
                    var stripeLink = stripeLinkForPending(p);
                    return (<tr key={"".concat(p.kind, ":").concat(p.external_id)} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{p.kind}</td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{p.user_id || '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            {p.amount_usd != null ? "$".concat(Number(p.amount_usd).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            {p.tokens != null ? Number(p.tokens).toLocaleString() : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{(stripeLink === null || stripeLink === void 0 ? void 0 : stripeLink.id) || '—'}</td>
                                                        <td className="px-2.5 py-1.5">
                                                            {stripeLink ? (<a href={stripeLink.url} target="_blank" rel="noreferrer" className="text-[#4372C3] underline hover:text-[#2B4B8A]">
                                                                    Open
                                                                </a>) : (<span className="text-[#7A99B0]">—</span>)}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#3A5672]">{p.external_id}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(p.created_at)}</td>
                                                    </tr>);
                })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Pending Economics Events" subtitle="All pending internal economics events (not just Stripe)." action={<Button variant="secondary" onClick={handleLoadPendingEconomics} disabled={loadingPendingEconomics}>
                                            {loadingPendingEconomics ? 'Loading…' : 'Refresh'}
                                        </Button>}/>
                                <CardBody className="space-y-3">
                                    <div className="grid grid-cols-3 gap-2.5">
                                        <Input label="Kind filter (optional)" value={pendingEconomicsKind} onChange={function (e) { return setPendingEconomicsKind(e.target.value); }} placeholder="subscription_rollover"/>
                                        <Input label="User ID filter (optional)" value={pendingEconomicsUserId} onChange={function (e) { return setPendingEconomicsUserId(e.target.value); }} placeholder="user123"/>
                                        <div className="flex items-end">
                                            <Button type="button" variant="secondary" disabled={loadingPendingEconomics} onClick={handleLoadPendingEconomics}>
                                                {loadingPendingEconomics ? 'Loading…' : 'Load'}
                                            </Button>
                                        </div>
                                    </div>

                                    {pendingEconomicsItems.length === 0 ? (<EmptyState message="No pending economics events loaded." icon="🧾"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Kind</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">User</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Amount</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Tokens</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Stripe ID</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Open</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">External ID</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Created</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {pendingEconomicsItems.map(function (p) {
                    var stripeLink = stripeLinkForPending(p);
                    return (<tr key={"".concat(p.kind, ":").concat(p.external_id)} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{p.kind}</td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{p.user_id || '—'}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            {p.amount_usd != null ? "$".concat(Number(p.amount_usd).toFixed(2)) : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            {p.tokens != null ? Number(p.tokens).toLocaleString() : '—'}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{(stripeLink === null || stripeLink === void 0 ? void 0 : stripeLink.id) || '—'}</td>
                                                        <td className="px-2.5 py-1.5">
                                                            {stripeLink ? (<a href={stripeLink.url} target="_blank" rel="noreferrer" className="text-[#4372C3] underline hover:text-[#2B4B8A]">
                                                                    Open
                                                                </a>) : (<span className="text-[#7A99B0]">—</span>)}
                                                        </td>
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#3A5672]">{p.external_id}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(p.created_at)}</td>
                                                    </tr>);
                })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Sweep Unused Subscription Balances" subtitle="Moves unused subscription balance to project budget for due subscriptions."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleSweepSubscriptionRollovers} className="space-y-3">
                                        <div className="flex items-end gap-2.5">
                                            <Input label="User ID (optional)" value={subSweepUserId} onChange={function (e) { return setSubSweepUserId(e.target.value); }} placeholder="user123" className="flex-1"/>
                                            <Button type="submit" variant="secondary" disabled={loadingAction}>
                                                {loadingAction ? 'Sweeping…' : 'Sweep Now'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Reap Expired Subscription Reservations" subtitle="Cleans up expired reservation holds across subscription periods."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleReapSubscriptionReservations} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID (optional)" value={subReapUserId} onChange={function (e) { return setSubReapUserId(e.target.value); }} placeholder="user123"/>
                                            <Input label="Max periods" value={subReapLimitPeriods} onChange={function (e) { return setSubReapLimitPeriods(e.target.value); }} placeholder="500"/>
                                            <Input label="Max per period" value={subReapPerPeriodLimit} onChange={function (e) { return setSubReapPerPeriodLimit(e.target.value); }} placeholder="500"/>
                                        </div>
                                        <div className="flex items-end">
                                            <Button type="submit" variant="secondary" disabled={loadingAction}>
                                                {loadingAction ? 'Reaping…' : 'Reap Now'}
                                            </Button>
                                        </div>
                                    </form>
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Subscription Period History" subtitle="Closed periods and ledger entries for a user's subscription."/>
                                <CardBody className="space-y-3">
                                    <form onSubmit={handleLoadSubscriptionPeriods} className="space-y-3">
                                        <div className="grid grid-cols-3 gap-2.5">
                                            <Input label="User ID *" value={subHistoryUserId} onChange={function (e) { return setSubHistoryUserId(e.target.value); }} placeholder="user123" required/>
                                            <Select label="Period status" value={subHistoryStatus} onChange={function (e) { return setSubHistoryStatus(e.target.value); }} options={[
                { value: 'closed', label: 'closed' },
                { value: 'open', label: 'open' },
                { value: 'all', label: 'all' },
            ]}/>
                                            <div className="flex items-end">
                                                <Button type="submit" variant="secondary" disabled={loadingHistory}>
                                                    {loadingHistory ? 'Loading…' : 'Load Periods'}
                                                </Button>
                                            </div>
                                        </div>
                                    </form>

                                    {subPeriods.length === 0 ? (<EmptyState message="No subscription periods loaded." icon="📚"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Period</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Status</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Topup</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Spent</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Rolled</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Balance</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Closed</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Actions</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {subPeriods.map(function (p) { return (<tr key={p.period_key} className={p.period_key === subSelectedPeriodKey ? 'bg-[rgba(1,190,178,0.07)]' : 'hover:bg-[#F6FAFA] transition-colors'}>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">
                                                            <div className="font-medium text-[#0D1E2C]">{formatDateTime(p.period_start)} → {formatDateTime(p.period_end)}</div>
                                                            <div className="font-mono text-[11px] text-[#7A99B0]">{p.period_key}</div>
                                                        </td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{p.status}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">${Number(p.topup_usd || 0).toFixed(2)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">${Number(p.spent_usd || 0).toFixed(2)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">${Number(p.rolled_over_usd || 0).toFixed(2)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">${Number(p.balance_usd || 0).toFixed(2)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(p.closed_at)}</td>
                                                        <td className="px-2.5 py-1.5">
                                                            <Button type="button" variant="secondary" disabled={loadingHistory} onClick={function () { return handleLoadSubscriptionLedger(p.period_key); }}>
                                                                {loadingHistory && p.period_key === subSelectedPeriodKey ? 'Loading…' : 'View Ledger'}
                                                            </Button>
                                                        </td>
                                                    </tr>); })}
                                                </tbody>
                                            </table>
                                        </div>)}

                                    {subSelectedPeriodKey && (<div className="pt-2.5 border-t border-[#E6F1F0] space-y-3">
                                            <div className="flex items-center justify-between">
                                                <div className="text-[12px] text-[#3A5672]">
                                                    Ledger for period: <span className="font-medium text-[#0D1E2C]">{subSelectedPeriodKey}</span>
                                                </div>
                                                <Button type="button" variant="secondary" disabled={loadingHistory} onClick={function () { return handleLoadSubscriptionLedger(subSelectedPeriodKey); }}>
                                                    {loadingHistory ? 'Refreshing…' : 'Refresh Ledger'}
                                                </Button>
                                            </div>

                                            {subLedger.length === 0 ? (<EmptyState message="No ledger entries for this period." icon="🧾"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                                    <table className="w-full text-[12px]">
                                                        <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                        <tr>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Time</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Kind</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Amount</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Provider</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Note</th>
                                                            <th className="px-2.5 py-1.5 text-left font-bold">Request</th>
                                                        </tr>
                                                        </thead>
                                                        <tbody className="divide-y divide-[#E6F1F0]">
                                                        {subLedger.map(function (l) {
                        var amt = Number(l.amount_usd || 0);
                        var sign = amt >= 0 ? '+' : '-';
                        return (<tr key={l.id} className="hover:bg-[#F6FAFA] transition-colors">
                                                                    <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(l.created_at)}</td>
                                                                    <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#0D1E2C]">{l.kind}</td>
                                                                    <td className="px-2.5 py-1.5 font-semibold text-[#0D1E2C]">
                                                                        {sign}${Math.abs(amt).toFixed(2)}
                                                                    </td>
                                                                    <td className="px-2.5 py-1.5 text-[#3A5672]">{l.provider || '—'}</td>
                                                                    <td className="px-2.5 py-1.5 text-[#3A5672]">{l.note || '—'}</td>
                                                                    <td className="px-2.5 py-1.5 font-mono text-[12px] text-[#3A5672]">{l.request_id || '—'}</td>
                                                                </tr>);
                    })}
                                                        </tbody>
                                                    </table>
                                                </div>)}
                                        </div>)}
                                </CardBody>
                            </Card>

                            <Card>
                                <CardHeader title="Recent Subscriptions" subtitle="Lists last updated subscriptions for this tenant/project." action={<Button variant="secondary" onClick={handleLoadSubscriptionsList} disabled={loadingData}>
                                            {loadingData ? 'Loading…' : 'Refresh'}
                                        </Button>}/>
                                <CardBody className="space-y-3">
                                    <div className="grid grid-cols-3 gap-2.5">
                                        <Select label="Provider filter" value={subsProviderFilter} onChange={function (e) { return setSubsProviderFilter(e.target.value); }} options={[
                { value: '', label: 'all' },
                { value: 'internal', label: 'internal' },
                { value: 'stripe', label: 'stripe' },
            ]}/>
                                    </div>

                                    {subsList.length === 0 ? (<EmptyState message="No subscriptions loaded (click Refresh)." icon="🧾"/>) : (<div className="max-h-72 overflow-auto rounded-lg border border-[#E6F1F0]">
                                            <table className="w-full text-[12px]">
                                                <thead className="sticky top-0 z-10 bg-[#F6FAFA] border-b border-[#E6F1F0] text-[10.5px] font-bold tracking-[0.1em] uppercase text-[#7A99B0]">
                                                <tr>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">User</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Billing</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Plan</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Due</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Next</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Last</th>
                                                    <th className="px-2.5 py-1.5 text-left font-bold">Updated</th>
                                                </tr>
                                                </thead>
                                                <tbody className="divide-y divide-[#E6F1F0]">
                                                {subsList.map(function (s) { return (<tr key={"".concat(s.tenant, ":").concat(s.project, ":").concat(s.user_id)} className="hover:bg-[#F6FAFA] transition-colors">
                                                        <td className="px-2.5 py-1.5 font-mono text-[12px] font-semibold text-[#0D1E2C]">{s.user_id}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{providerLabel(s.provider)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{s.plan_id || '—'}</td>
                                                        <td className="px-2.5 py-1.5"><DuePill sub={s}/></td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(s.next_charge_at)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(s.last_charged_at)}</td>
                                                        <td className="px-2.5 py-1.5 text-[#3A5672]">{formatDateTime(s.updated_at)}</td>
                                                    </tr>); })}
                                                </tbody>
                                            </table>
                                        </div>)}
                                </CardBody>
                            </Card>
                            </div>
                        </div>)}
                    {/* Data lists loading indicator (global hint) */}
                    {(viewMode === 'quotaPolicies' || viewMode === 'budgetPolicies' || viewMode === 'appBudget' || viewMode === 'reservation') && loadingData && (<div className="pointer-events-none absolute bottom-2 right-4 text-[11.5px] text-[#7A99B0]">Loading…</div>)}


                </div>
            </div>
        </div>);
};
// Render
var rootElement = document.getElementById('root');
if (rootElement) {
    var root = client_1.default.createRoot(rootElement);
    root.render(<EconomicsAdmin />);
}
