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
function parseJwt(token) {
    try {
        var base64Url = token.split('.')[1];
        var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        var jsonPayload = decodeURIComponent(atob(base64).split('').map(function (c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    }
    catch (e) {
        return null;
    }
}
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
            defaultAppBundleId: '{{DEFAULT_APP_BUNDLE_ID}}',
            stripeDashboardBaseUrl: '',
        };
        this.configReceivedCallback = null;
    }
    SettingsManager.prototype.getBaseUrl = function () {
        if (this.settings.baseUrl === this.PLACEHOLDER_BASE_URL) {
            return window.location.origin;
        }
        try {
            var url = new URL(this.settings.baseUrl);
            if (url.port === 'None' || url.hostname.includes('None'))
                return window.location.origin;
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
        return this.settings.idTokenHeader === this.PLACEHOLDER_ID_TOKEN_HEADER ? 'X-ID-Token' : this.settings.idTokenHeader;
    };
    SettingsManager.prototype.getDefaultTenant = function () {
        return this.settings.defaultTenant === this.PLACEHOLDER_TENANT ? 'home' : this.settings.defaultTenant;
    };
    SettingsManager.prototype.getDefaultProject = function () {
        return this.settings.defaultProject === this.PLACEHOLDER_PROJECT ? 'demo' : this.settings.defaultProject;
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
        var identity = "CONTROL_PLANE_ADMIN";
        window.addEventListener('message', function (event) {
            if (event.data.type === 'CONN_RESPONSE' || event.data.type === 'CONFIG_RESPONSE') {
                var requestedIdentity = event.data.identity;
                if (requestedIdentity !== identity)
                    return;
                console.log('[UserBilling] RECEIVED CONFIG:', event.data.config);
                if (event.data.config) {
                    if (_this.applyRuntimeConfig(event.data.config)) {
                        console.log('[UserBilling] Applying updates to settings');
                    }
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
                    var timeout = window.setTimeout(function () {
                        console.warn('[UserBilling] Config request timeout - proceeding with current settings');
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
        return Promise.resolve(!this.hasPlaceholderSettings());
    };
    return SettingsManager;
}());
var settings = new SettingsManager();
function makeAuthHeaders(base) {
    var headers = new Headers(base);
    var accessToken = settings.getAccessToken();
    var idToken = settings.getIdToken();
    var idTokenHeader = settings.getIdTokenHeader();
    if (accessToken)
        headers.set('Authorization', "Bearer ".concat(accessToken));
    if (idToken)
        headers.set(idTokenHeader, idToken);
    return headers;
}
// =============================================================================
// API Client
// =============================================================================
var BillingAPI = /** @class */ (function () {
    function BillingAPI() {
    }
    BillingAPI.prototype.getMeUrl = function (path) { return "".concat(settings.getBaseUrl(), "/api/economics/me").concat(path); };
    BillingAPI.prototype.getStripeCheckoutUrl = function (path) { return "".concat(settings.getBaseUrl(), "/api/economics/stripe/checkout").concat(path); };
    BillingAPI.prototype.fetchWithAuth = function (url_1) {
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
                        throw new Error("API error: ".concat(response.status, " - ").concat(errorText));
                    case 3: return [2 /*return*/, response];
                }
            });
        });
    };
    BillingAPI.prototype.getBudgetBreakdown = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getMeUrl('/budget-breakdown'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    // Actual per-model spend (aggregates-only on the server — always fast).
    BillingAPI.prototype.getCostBreakdown = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getMeUrl('/cost-breakdown'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.listSubscriptionPlans = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getMeUrl('/subscription-plans'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.getSubscription = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getMeUrl('/subscription'))];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.openCustomerPortal = function () {
        return __awaiter(this, void 0, void 0, function () {
            var returnUrl, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        returnUrl = currentFrameReturnUrl();
                        return [4 /*yield*/, this.fetchWithAuth("".concat(this.getMeUrl('/stripe/customer-portal'), "?return_url=").concat(encodeURIComponent(returnUrl)), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.cancelSubscription = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getMeUrl('/subscription/cancel'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: '{}'
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.createCheckoutTopup = function (amountUsd, successUrl, cancelUrl) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeCheckoutUrl('/topup'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ amount_usd: amountUsd, success_url: successUrl, cancel_url: cancelUrl })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    BillingAPI.prototype.createCheckoutSubscription = function (planId, successUrl, cancelUrl) {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.fetchWithAuth(this.getStripeCheckoutUrl('/subscription'), {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ plan_id: planId, success_url: successUrl, cancel_url: cancelUrl })
                        })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.json()];
                }
            });
        });
    };
    return BillingAPI;
}());
// =============================================================================
// UI Components
// =============================================================================
var Card = function (_a) {
    var children = _a.children, _b = _a.className, className = _b === void 0 ? '' : _b;
    return (<div className={"bg-white rounded-2xl shadow-sm border border-gray-200/70 overflow-hidden ".concat(className)}>
        {children}
    </div>);
};
var Button = function (_a) {
    var children = _a.children, onClick = _a.onClick, _b = _a.disabled, disabled = _b === void 0 ? false : _b, _c = _a.variant, variant = _c === void 0 ? 'primary' : _c, _d = _a.className, className = _d === void 0 ? '' : _d;
    var variants = {
        primary: 'bg-gray-900 hover:bg-gray-800 text-white',
        secondary: 'bg-white hover:bg-gray-50 text-gray-900 border border-gray-200',
    };
    return (<button onClick={onClick} disabled={disabled} className={"px-4 py-2.5 rounded-xl text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ".concat(variants[variant], " ").concat(className)}>
            {children}
        </button>);
};
var LoadingSpinner = function () { return (<div className="flex justify-center py-10">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-200 border-t-gray-900"></div>
    </div>); };
function formatCount(value) {
    if (value == null)
        return '∞';
    return value.toLocaleString();
}
function formatUsd(value) {
    var amount = Number(value || 0);
    return "$".concat(amount.toFixed(2));
}
function formatUsdLimit(value) {
    if (value == null)
        return '∞';
    return formatUsd(value);
}
function formatDateTime(value) {
    if (!value)
        return 'Not available';
    return new Date(value).toLocaleString();
}
function currentFrameReturnUrl() {
    try {
        return window.parent !== window ? window.parent.location.href : window.location.href;
    }
    catch (_a) {
        return window.location.href;
    }
}
function navigateTopLevel(url) {
    try {
        window.top.location.href = url;
    }
    catch (_a) {
        window.location.href = url;
    }
}
var MetricRow = function (_a) {
    var label = _a.label, used = _a.used, limit = _a.limit, remaining = _a.remaining, usedUsd = _a.usedUsd, limitUsd = _a.limitUsd, remainingUsd = _a.remainingUsd;
    var hasUsd = usedUsd != null || limitUsd != null || remainingUsd != null;
    return (<div className="rounded-xl border border-gray-100 bg-gray-50 px-4 py-3">
            <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-gray-500">{label}</span>
                <span className="font-semibold text-gray-900">
                    {hasUsd ? "".concat(formatUsd(usedUsd), " / ").concat(formatUsdLimit(limitUsd)) : "".concat(formatCount(used), " / ").concat(formatCount(limit))}
                </span>
            </div>
            <div className="mt-1 flex items-center justify-between gap-3 text-xs text-gray-500">
                <span>
                    Remaining: {hasUsd ? formatUsdLimit(remainingUsd) : formatCount(remaining)}
                </span>
            </div>
            {hasUsd && (<div className="mt-1 text-xs text-gray-400">
                    Tokens: {formatCount(used)} / {formatCount(limit)} · remaining {formatCount(remaining)}
                </div>)}
        </div>);
};
var PlanReservationMetric = function (_a) {
    var tokens = _a.tokens, usd = _a.usd;
    return (<div className="rounded-xl border border-amber-100 bg-amber-50 px-4 py-3">
        <div className="flex items-center justify-between gap-3 text-sm">
            <span className="text-amber-800">Plan reserved</span>
            <span className="font-semibold text-amber-950">{formatUsd(usd)}</span>
        </div>
        <div className="mt-1 text-xs text-amber-800">
            {formatCount(tokens)} tokens held by in-flight requests
        </div>
    </div>);
};
var WalletMetric = function (_a) {
    var label = _a.label, value = _a.value, hint = _a.hint;
    return (<div className="rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-emerald-700">{label}</div>
        <div className="mt-1 text-sm font-semibold text-emerald-950">{value}</div>
        {hint && <div className="mt-1 text-xs text-emerald-800">{hint}</div>}
    </div>);
};
// =============================================================================
// Main Component
// =============================================================================
var UserBillingDashboard = function () {
    var _a, _b, _c, _d, _e, _f, _g;
    var api = (0, react_1.useMemo)(function () { return new BillingAPI(); }, []);
    var _h = (0, react_1.useState)('initializing'), configStatus = _h[0], setConfigStatus = _h[1];
    var _j = (0, react_1.useState)(false), loading = _j[0], setLoading = _j[1];
    var _k = (0, react_1.useState)(null), error = _k[0], setError = _k[1];
    var _l = (0, react_1.useState)(null), breakdown = _l[0], setBreakdown = _l[1];
    var _m = (0, react_1.useState)(null), costBreakdown = _m[0], setCostBreakdown = _m[1];
    var _o = (0, react_1.useState)([]), plans = _o[0], setPlans = _o[1];
    var _p = (0, react_1.useState)(null), subscription = _p[0], setSubscription = _p[1];
    var _q = (0, react_1.useState)('10'), topupAmount = _q[0], setTopupAmount = _q[1];
    var _r = (0, react_1.useState)(false), cancelConfirm = _r[0], setCancelConfirm = _r[1];
    (0, react_1.useEffect)(function () {
        settings.setupParentListener().then(function () {
            setConfigStatus('ready');
        }).catch(function () { return setConfigStatus('error'); });
    }, []);
    (0, react_1.useEffect)(function () {
        if (configStatus === 'ready') {
            loadData();
        }
    }, [configStatus]);
    var loadData = function () { return __awaiter(void 0, void 0, void 0, function () {
        var _a, brk, cost, pData, subData, err_1;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    setLoading(true);
                    setError(null);
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, Promise.all([
                            api.getBudgetBreakdown().catch(function () { return null; }),
                            api.getCostBreakdown().catch(function () { return null; }),
                            api.listSubscriptionPlans().catch(function () { return ({ plans: [] }); }),
                            api.getSubscription().catch(function () { return ({ subscription: null }); })
                        ])];
                case 2:
                    _a = _b.sent(), brk = _a[0], cost = _a[1], pData = _a[2], subData = _a[3];
                    if (brk)
                        setBreakdown(brk);
                    setCostBreakdown(cost);
                    if (pData)
                        setPlans(pData.plans);
                    if (subData)
                        setSubscription(subData.subscription);
                    return [3 /*break*/, 5];
                case 3:
                    err_1 = _b.sent();
                    setError(err_1.message);
                    return [3 /*break*/, 5];
                case 4:
                    setLoading(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var handleTopup = function () { return __awaiter(void 0, void 0, void 0, function () {
        var amt, returnUrl, res, err_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    amt = parseFloat(topupAmount);
                    if (isNaN(amt) || amt < 0.5) {
                        setError("Amount must be at least $0.50");
                        return [2 /*return*/];
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    setLoading(true);
                    returnUrl = currentFrameReturnUrl();
                    return [4 /*yield*/, api.createCheckoutTopup(amt, returnUrl, returnUrl)];
                case 2:
                    res = _a.sent();
                    if (res.checkout_url)
                        navigateTopLevel(res.checkout_url);
                    return [3 /*break*/, 4];
                case 3:
                    err_2 = _a.sent();
                    setError(err_2.message);
                    setLoading(false);
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleSubscribe = function (planId) { return __awaiter(void 0, void 0, void 0, function () {
        var returnUrl, res, err_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    setLoading(true);
                    returnUrl = currentFrameReturnUrl();
                    return [4 /*yield*/, api.createCheckoutSubscription(planId, returnUrl, returnUrl)];
                case 1:
                    res = _a.sent();
                    if (res.checkout_url)
                        navigateTopLevel(res.checkout_url);
                    return [3 /*break*/, 3];
                case 2:
                    err_3 = _a.sent();
                    setError(err_3.message);
                    setLoading(false);
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); };
    var handleCustomerPortal = function () { return __awaiter(void 0, void 0, void 0, function () {
        var res, err_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, 3, 4]);
                    setLoading(true);
                    return [4 /*yield*/, api.openCustomerPortal()];
                case 1:
                    res = _a.sent();
                    if (res.portal_url)
                        navigateTopLevel(res.portal_url);
                    return [3 /*break*/, 4];
                case 2:
                    err_4 = _a.sent();
                    setError(err_4.message);
                    return [3 /*break*/, 4];
                case 3:
                    setLoading(false);
                    return [7 /*endfinally*/];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleCancelSubscription = function () { return __awaiter(void 0, void 0, void 0, function () {
        var err_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!cancelConfirm) {
                        setCancelConfirm(true);
                        return [2 /*return*/];
                    }
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 4, 5, 6]);
                    setLoading(true);
                    setCancelConfirm(false);
                    return [4 /*yield*/, api.cancelSubscription()];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, loadData()];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 6];
                case 4:
                    err_5 = _a.sent();
                    setError(err_5.message);
                    return [3 /*break*/, 6];
                case 5:
                    setLoading(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    if (configStatus === 'initializing')
        return <LoadingSpinner />;
    var currentPlanId = (breakdown === null || breakdown === void 0 ? void 0 : breakdown.plan_id) || (subscription === null || subscription === void 0 ? void 0 : subscription.plan_id) || 'free';
    var personalCredits = breakdown === null || breakdown === void 0 ? void 0 : breakdown.lifetime_credits;
    var availablePersonalCreditsUsd = (personalCredits === null || personalCredits === void 0 ? void 0 : personalCredits.available_usd) || 0;
    var hasPersonalOverflowCover = availablePersonalCreditsUsd > 0;
    return (<div className="h-screen overflow-hidden bg-gray-50/50 p-3 md:p-4 font-sans text-gray-900">
            <div className="mx-auto flex h-full max-w-5xl flex-col gap-4 overflow-hidden">
                
                <div className="flex shrink-0 items-start justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">Billing & Plans</h1>
                        <p className="mt-1 text-sm text-gray-500">Manage your plan quota and personal wallet.</p>
                    </div>
                    <div className="shrink-0 flex items-center gap-2">
                        <button onClick={loadData} disabled={loading} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200 disabled:opacity-50">
                            <svg xmlns="http://www.w3.org/2000/svg" className={"h-4 w-4 ".concat(loading ? 'animate-spin' : '')} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/>
                                <path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>
                            </svg>
                            Refresh
                        </button>
                        {(subscription === null || subscription === void 0 ? void 0 : subscription.stripe_customer_id) && (<button onClick={handleCustomerPortal} disabled={loading} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 hover:bg-indigo-100 disabled:opacity-50">
                                Manage Billing ↗
                            </button>)}
                    </div>
                </div>

                {error && (<div className="p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl">
                        {error}
                    </div>)}

                {loading && !breakdown && <LoadingSpinner />}

                {!loading && breakdown && (<div className="min-h-0 overflow-y-auto pr-1">
                        <div className="flex flex-col gap-4">
                            {/* Current Plan Overview */}
                            <Card className="p-4">
                                <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">Plan quota</h3>
                                <div className="text-xl font-bold capitalize">{currentPlanId}</div>
                                {(subscription === null || subscription === void 0 ? void 0 : subscription.status) === 'active' && (<div className="mt-2 inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-800">
                                        Active Subscription
                                    </div>)}
                                <div className="mt-4 rounded-xl border border-sky-100 bg-sky-50 px-4 py-3 text-sm text-sky-900">
                                    <div className="font-semibold">Active quota buckets</div>
                                    <p className="mt-1 text-sky-800">
                                        The numbers below reflect your combined usage across all apps in this workspace. Hourly usage is the last 60 minutes; daily and monthly usage are the current quota periods since their last reset.
                                    </p>
                                </div>
                                <div className="mt-4 space-y-3">
                                    {(subscription === null || subscription === void 0 ? void 0 : subscription.started_at) && (<div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Started</span>
                                            <span className="font-medium">{new Date(subscription.started_at).toLocaleString()}</span>
                                        </div>)}
                                    {(subscription === null || subscription === void 0 ? void 0 : subscription.next_charge_at) && (<div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Next renewal</span>
                                            <span className="font-medium">{new Date(subscription.next_charge_at).toLocaleString()}</span>
                                        </div>)}
                                    <div className="border-t border-gray-100 pt-3">
                                        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">Last 60 minutes</div>
                                        <MetricRow label="Tokens" used={breakdown.current_usage.tokens_this_hour} limit={breakdown.effective_policy.tokens_per_hour} remaining={breakdown.remaining.tokens_this_hour} usedUsd={breakdown.current_usage.tokens_this_hour_usd} limitUsd={breakdown.effective_policy.usd_per_hour} remainingUsd={breakdown.remaining.tokens_this_hour_usd}/>
                                        {((_a = breakdown.reset_windows) === null || _a === void 0 ? void 0 : _a.hour_reset_at) && (<div className="mt-2 text-xs text-gray-500">
                                                Hourly window resets at {formatDateTime(breakdown.reset_windows.hour_reset_at)}
                                            </div>)}
                                    </div>
                                    <div className="border-t border-gray-100 pt-3">
                                        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">Current 24h quota period</div>
                                        <MetricRow label="Requests" used={breakdown.current_usage.requests_today} limit={breakdown.effective_policy.requests_per_day} remaining={breakdown.remaining.requests_today}/>
                                        <div className="mt-2">
                                            <MetricRow label="Tokens" used={breakdown.current_usage.tokens_today} limit={breakdown.effective_policy.tokens_per_day} remaining={breakdown.remaining.tokens_today} usedUsd={breakdown.current_usage.tokens_today_usd} limitUsd={breakdown.effective_policy.usd_per_day} remainingUsd={breakdown.remaining.tokens_today_usd}/>
                                        </div>
                                        {((_b = breakdown.reset_windows) === null || _b === void 0 ? void 0 : _b.day_reset_at) && (<div className="mt-2 text-xs text-gray-500">
                                                Daily quota resets at {formatDateTime(breakdown.reset_windows.day_reset_at)}
                                            </div>)}
                                    </div>
                                    <div className="border-t border-gray-100 pt-3">
                                        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">Current 30-day quota period</div>
                                        <MetricRow label="Requests" used={breakdown.current_usage.requests_this_month} limit={breakdown.effective_policy.requests_per_month} remaining={breakdown.remaining.requests_this_month}/>
                                        <div className="mt-2">
                                            <MetricRow label="Tokens" used={breakdown.current_usage.tokens_this_month} limit={breakdown.effective_policy.tokens_per_month} remaining={breakdown.remaining.tokens_this_month} usedUsd={breakdown.current_usage.tokens_this_month_usd} limitUsd={breakdown.effective_policy.usd_per_month} remainingUsd={breakdown.remaining.tokens_this_month_usd}/>
                                        </div>
                                        {((_c = breakdown.reset_windows) === null || _c === void 0 ? void 0 : _c.month_reset_at) && (<div className="mt-2 text-xs text-gray-500">
                                                Monthly quota resets at {formatDateTime(breakdown.reset_windows.month_reset_at)}
                                            </div>)}
                                    </div>
                                    <PlanReservationMetric tokens={breakdown.current_usage.tokens_reserved || 0} usd={breakdown.current_usage.tokens_reserved_usd}/>
                                    <div className={"rounded-xl border px-4 py-3 text-sm ".concat(hasPersonalOverflowCover ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : 'border-amber-200 bg-amber-50 text-amber-900')}>
                                        <div className="font-semibold">If one request is larger than your remaining plan tokens</div>
                                        <p className="mt-1">
                                            The part above your remaining plan quota is covered by personal credits. You currently have {formatUsd(availablePersonalCreditsUsd)} available.
                                        </p>
                                        {!hasPersonalOverflowCover && (<p className="mt-1">
                                                If a request is larger than the remaining plan quota shown above, you will need to wait for the rolling window reset or add funds.
                                            </p>)}
                                    </div>
                                </div>
                                {(subscription === null || subscription === void 0 ? void 0 : subscription.status) === 'active' && (subscription === null || subscription === void 0 ? void 0 : subscription.provider) === 'stripe' && (<div className="mt-6 pt-4 border-t border-gray-100">
                                        {cancelConfirm ? (<div className="space-y-2">
                                                <p className="text-xs text-gray-500">Cancel at end of current billing period?</p>
                                                <div className="flex gap-2">
                                                    <button onClick={handleCancelSubscription} disabled={loading} className="flex-1 py-1.5 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 rounded-lg disabled:opacity-50">
                                                        Yes, cancel
                                                    </button>
                                                    <button onClick={function () { return setCancelConfirm(false); }} className="flex-1 py-1.5 text-xs font-semibold text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-lg">
                                                        Keep plan
                                                    </button>
                                                </div>
                                            </div>) : (<button onClick={handleCancelSubscription} disabled={loading} className="text-xs text-rose-600 hover:text-rose-800 font-medium disabled:opacity-50">
                                                Cancel subscription
                                            </button>)}
                                    </div>)}
                            </Card>

                            {/* Wallet Overview */}
                            <Card className="p-4">
                                <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">Wallet / Personal Credits</h3>
                                <div className="text-xl font-bold">
                                    ${(((_d = breakdown.lifetime_credits) === null || _d === void 0 ? void 0 : _d.available_usd) || 0).toFixed(2)}
                                </div>
                                <div className="mb-4 text-sm text-gray-500">available</div>
                                <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-4">
                                    <WalletMetric label="Purchased" value={formatUsd((personalCredits === null || personalCredits === void 0 ? void 0 : personalCredits.purchased_usd) || 0)}/>
                                    <WalletMetric label="Spent" value={formatUsd((personalCredits === null || personalCredits === void 0 ? void 0 : personalCredits.spent_usd) || 0)}/>
                                    <WalletMetric label="Reserved" value={formatUsd((personalCredits === null || personalCredits === void 0 ? void 0 : personalCredits.reserved_usd) || 0)}/>
                                    <WalletMetric label="Available" value={formatUsd((personalCredits === null || personalCredits === void 0 ? void 0 : personalCredits.available_usd) || 0)}/>
                                </div>
                                
                                <div className="rounded-xl border border-gray-100 bg-gray-50 p-3">
                                    <label className="block text-xs font-semibold text-gray-700 mb-2">Top up balance (USD)</label>
                                    <div className="flex gap-2">
                                        <div className="relative flex-1">
                                            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                                            <input type="number" value={topupAmount} onChange={function (e) { return setTopupAmount(e.target.value); }} className="w-full pl-7 pr-4 py-2 bg-white border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900/10 focus:border-gray-300 outline-none"/>
                                        </div>
                                        <Button onClick={handleTopup} disabled={loading}>Add Funds</Button>
                                    </div>
                                </div>
                            </Card>

                            {/* Actual spend this month — per-model dollars from the live
                price table. The quota meters above are quota-equivalent
                units for enforcement; the two differ by design. */}
                            {costBreakdown && (<Card className="p-4">
                                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">Actual Spend (This Month)</h3>
                                    {costBreakdown.coverage === 'no_aggregates' ? (<div className="rounded-xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                                            Spend reports for this window are not aggregated yet. Check back shortly.
                                        </div>) : (<>
                                            <div className="text-xl font-bold">${(costBreakdown.total_cost_usd || 0).toFixed(4)}</div>
                                            <div className="mb-4 text-sm text-gray-500">
                                                {costBreakdown.date_from} — {costBreakdown.date_to} · priced per model at actual rates
                                            </div>
                                            <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                                                <WalletMetric label="Input tokens" value={formatCount((_e = costBreakdown.tokens) === null || _e === void 0 ? void 0 : _e.input_tokens)}/>
                                                <WalletMetric label="Output tokens" value={formatCount((_f = costBreakdown.tokens) === null || _f === void 0 ? void 0 : _f.output_tokens)}/>
                                                <WalletMetric label="Calls" value={formatCount(costBreakdown.event_count)}/>
                                            </div>
                                            {costBreakdown.by_model.length > 0 && (<div className="overflow-hidden rounded-xl border border-gray-100">
                                                    <table className="w-full text-sm">
                                                        <thead className="bg-gray-50 text-left text-xs font-semibold uppercase tracking-wider text-gray-400">
                                                            <tr>
                                                                <th className="px-4 py-2">Model</th>
                                                                <th className="px-4 py-2 text-right">Cost</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {costBreakdown.by_model.map(function (line, i) { return (<tr key={i} className="border-t border-gray-100">
                                                                    <td className="px-4 py-2 text-gray-700">
                                                                        {line.model || line.provider || line.service}
                                                                        {line.model && line.provider ? <span className="ml-1 text-xs text-gray-400">· {line.provider}</span> : null}
                                                                    </td>
                                                                    <td className="px-4 py-2 text-right font-medium text-gray-900">${Number(line.cost_usd || 0).toFixed(4)}</td>
                                                                </tr>); })}
                                                        </tbody>
                                                    </table>
                                                </div>)}
                                        </>)}
                                </Card>)}

                            {((_g = breakdown.subscription_balance) === null || _g === void 0 ? void 0 : _g.has_subscription) && (<Card className="p-4">
                                    <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-500">Subscription Balance</h3>
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                                        <div className="rounded-xl border border-gray-100 bg-gray-50 px-4 py-3">
                                            <div className="text-xs font-semibold uppercase tracking-wider text-gray-400">Available</div>
                                            <div className="mt-1 text-sm font-medium text-gray-900">
                                                {formatUsd(breakdown.subscription_balance.available_usd)}
                                            </div>
                                        </div>
                                        <div className="rounded-xl border border-gray-100 bg-gray-50 px-4 py-3">
                                            <div className="text-xs font-semibold uppercase tracking-wider text-gray-400">Reserved</div>
                                            <div className="mt-1 text-sm font-medium text-gray-900">
                                                {formatUsd(breakdown.subscription_balance.reserved_usd || 0)}
                                            </div>
                                        </div>
                                        <div className="rounded-xl border border-gray-100 bg-gray-50 px-4 py-3">
                                            <div className="text-xs font-semibold uppercase tracking-wider text-gray-400">Spent This Period</div>
                                            <div className="mt-1 text-sm font-medium text-gray-900">
                                                {formatUsd(breakdown.subscription_balance.spent_usd)}
                                            </div>
                                        </div>
                                        <div className="rounded-xl border border-gray-100 bg-gray-50 px-4 py-3">
                                            <div className="text-xs font-semibold uppercase tracking-wider text-gray-400">Period Ends</div>
                                            <div className="mt-1 text-sm font-medium text-gray-900">
                                                {formatDateTime(breakdown.subscription_balance.period_end)}
                                            </div>
                                        </div>
                                    </div>
                                </Card>)}
                        </div>

                        {/* Available Plans */}
                        <div className="mt-4">
                            <h3 className="mb-3 text-lg font-bold">Available Subscriptions</h3>
                            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                                {plans.map(function (plan) {
                var isCurrent = plan.plan_id === currentPlanId;
                return (<Card key={plan.plan_id} className={"flex flex-col p-4 ".concat(isCurrent ? 'ring-2 ring-gray-900' : '')}>
                                            {isCurrent && (<span className="self-start px-2 py-1 bg-gray-900 text-white text-xs font-bold rounded mb-4">CURRENT</span>)}
                                            <div className="text-lg font-bold capitalize mb-1">{plan.plan_id}</div>
                                            <div className="text-2xl font-bold mb-4">${(plan.monthly_price_cents / 100).toFixed(2)}<span className="text-sm font-normal text-gray-500">/mo</span></div>
                                            {plan.notes && <p className="text-sm text-gray-600 mb-6 flex-1">{plan.notes}</p>}
                                            {!isCurrent && (<Button className="w-full mt-auto" onClick={function () { return handleSubscribe(plan.plan_id); }} disabled={loading}>
                                                    Subscribe
                                                </Button>)}
                                            {isCurrent && (<Button variant="secondary" className="w-full mt-auto" disabled>
                                                    Active
                                                </Button>)}
                                        </Card>);
            })}
                                {plans.length === 0 && (<div className="col-span-full text-center py-10 text-gray-500 border-2 border-dashed border-gray-200 rounded-2xl">
                                        No subscription plans available right now.
                                    </div>)}
                            </div>
                        </div>
                    </div>)}

            </div>
        </div>);
};
var root = client_1.default.createRoot(document.getElementById('root'));
root.render(<UserBillingDashboard />);
