function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function externalUrlField(record: Record<string, unknown>): string {
  for (const key of ['external_url', 'externalUrl']) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

/**
 * Return the provider-authorized browser URL from an object-open response.
 * Generic clients use only the resolver's explicit action result; URLs inside
 * object display metadata do not authorize navigation.
 */
export function objectActionExternalUrl(
  response: Record<string, unknown>,
  baseUrl?: string,
): string {
  const uiEvent = asRecord(response.ui_event)
  const extra = asRecord(response.extra)
  const candidate =
    externalUrlField(uiEvent) ||
    externalUrlField(response) ||
    externalUrlField(extra)
  if (!candidate) return ''
  try {
    const url = baseUrl ? new URL(candidate, baseUrl) : new URL(candidate)
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : ''
  } catch {
    return ''
  }
}
