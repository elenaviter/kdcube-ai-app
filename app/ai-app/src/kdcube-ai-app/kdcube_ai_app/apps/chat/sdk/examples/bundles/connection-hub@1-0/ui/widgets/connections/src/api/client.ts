// Operation client for the Connections widget. GET for reads (catalog/status),
// POST (with the { data } envelope) for actions; responses are unwrapped on the
// alias key. An optional host bridge lets the scene host route calls when the
// widget runs embedded.

import { settings } from './settings';

export type ConnectionsCallOperation = <T>(
  method: 'GET' | 'POST',
  operation: string,
  payload?: Record<string, unknown>,
  options?: { headers?: Record<string, string> },
) => Promise<T>;

let hostCallOperation: ConnectionsCallOperation | null = null;

export function setConnectionsCallOperation(callOperation: ConnectionsCallOperation): () => void {
  hostCallOperation = callOperation;
  return () => {
    if (hostCallOperation === callOperation) hostCallOperation = null;
  };
}

/** The absolute URL of an operation on this deployment — the same endpoint the
 *  widget calls, so a surface can show an operator the exact request to script. */
export function operationUrl(operation: string): string {
  return apiUrl('operations', operation);
}

function apiUrl(route: 'operations' | 'public', operation: string): string {
  const tenant = encodeURIComponent(settings.getTenant());
  const project = encodeURIComponent(settings.getProject());
  const bundleId = encodeURIComponent(settings.getBundleId());
  return `${settings.getBaseUrl()}/api/integrations/bundles/${tenant}/${project}/${bundleId}/${route}/${operation}`;
}

function operationCsrfUrl(operation: string): string {
  return `${apiUrl('operations', operation)}/csrf`;
}

async function operationCsrfToken(operation: string, headers: Headers): Promise<string> {
  const response = await fetch(operationCsrfUrl(operation), {
    method: 'GET',
    credentials: 'include',
    cache: 'no-store',
    headers,
  });
  const text = await response.text();
  let parsed: Record<string, unknown> = {};
  try {
    parsed = text ? JSON.parse(text) as Record<string, unknown> : {};
  } catch {
    parsed = {};
  }
  if (!response.ok) {
    const detail = parsed.detail ? String(parsed.detail) : text || response.statusText;
    throw new Error(detail || `CSRF preparation failed: ${response.status}`);
  }
  if (!parsed.csrf_required) return '';
  const token = String(parsed.csrf_token || '').trim();
  if (!token) throw new Error('Operation CSRF token was not returned.');
  return token;
}

function unwrap<T>(operation: string, parsed: unknown): T {
  if (parsed && typeof parsed === 'object' && operation in parsed) {
    return (parsed as Record<string, unknown>)[operation] as T;
  }
  return parsed as T;
}

async function request<T>(
  method: 'GET' | 'POST',
  operation: string,
  payload: Record<string, unknown> = {},
  route: 'operations' | 'public' = 'operations',
): Promise<T> {
  const headers = settings.authHeaders({ Accept: 'application/json' });
  const init: RequestInit = { method, credentials: 'include', headers };
  const bridgeHeaders: Record<string, string> = {};
  if (method === 'POST') {
    headers.set('Content-Type', 'application/json');
    if (route === 'operations') {
      const csrfToken = await operationCsrfToken(operation, headers);
      if (csrfToken) {
        headers.set('X-KDCube-CSRF-Token', csrfToken);
        bridgeHeaders['X-KDCube-CSRF-Token'] = csrfToken;
      }
    }
    init.body = JSON.stringify({ data: payload });
  }
  if (hostCallOperation && route === 'operations') {
    return hostCallOperation<T>(method, operation, payload, { headers: bridgeHeaders });
  }
  const response = await fetch(apiUrl(route, operation), init);
  const text = await response.text();
  let parsed: unknown = {};
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    parsed = { raw: text };
  }
  if (!response.ok) {
    const detail = typeof parsed === 'object' && parsed && 'detail' in parsed
      ? String((parsed as Record<string, unknown>).detail)
      : text || response.statusText;
    throw new Error(detail || `${operation} failed: ${response.status}`);
  }
  return unwrap<T>(operation, parsed);
}

export function getOp<T>(operation: string): Promise<T> {
  return request<T>('GET', operation);
}

export function postOp<T>(operation: string, payload: Record<string, unknown> = {}): Promise<T> {
  return request<T>('POST', operation, payload);
}

export function postPublicOp<T>(operation: string, payload: Record<string, unknown> = {}): Promise<T> {
  return request<T>('POST', operation, payload, 'public');
}

export function getPublicOp<T>(operation: string): Promise<T> {
  return request<T>('GET', operation, {}, 'public');
}
