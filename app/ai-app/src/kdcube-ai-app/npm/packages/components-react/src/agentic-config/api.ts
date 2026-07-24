/**
 * Agentic-config client — transport-injected, host-agnostic.
 *
 * The instruction-set operations ride the `agentic_instructions` op on the
 * bundle that owns the `instr` namespace (`kdcube-services@1-0` by default);
 * app/agent discovery, config merge-writes, and secrets use the platform
 * admin routes. Writes are admin-gated SERVER-SIDE — this client sends the
 * caller's identity and renders the structured denial when it is not enough.
 */

/** Host-supplied runtime: base URL, auth headers, scope, and the ops bundle. */
export interface AgenticConfigTransport {
  baseUrl(): string
  authHeaders(extra?: Record<string, string>): Record<string, string>
  tenant(): string
  project(): string
  /** The bundle serving the `agentic_instructions` operation. */
  opsBundle(): string
}

export interface InstructionVersionRow {
  version: number
  status: string
  created_by: string
  created_at?: string | null
}
export interface InstructionRecord {
  ref: string
  instruction_id: string
  version: number
  name: string
  description: string
  tags: string[]
  signals: string[]
  items: string[]
  status: string
  created_by: string
  created_at?: string | null
  updated_by?: string
  updated_at?: string | null
  versions?: InstructionVersionRow[]
}
export interface BuiltinBlock {
  name: string
  tier: string
  description: string
  signals: string[]
  tags: string[]
  /** moderate blocks: profiles whose expansion includes this block */
  profiles?: string[]
  /** full block text for the details view */
  text?: string
  /** token weight of the block body (cl100k) */
  tokens?: number
}
export interface ComposedSegment {
  item: string
  body: string
  /** token weight of this segment (cl100k) */
  tokens?: number
}
/** The server projection of a DRAFT agent config: the token breakdown of
 *  the REAL system instruction it would produce. */
export interface ProjectionResult {
  tokens: {
    total: number
    protocol_and_instructions: number
    tool_catalog: number
    skill_gallery: number
  }
  facets: Record<string, unknown>
  tools: { included_ids: string[]; skipped: string[] }
  items_expanded: string[]
  text?: string
}
export interface OpError {
  code: string
  message: string
}
interface OpEnvelope {
  ok: boolean
  ret?: {
    object?: InstructionRecord
    items?: InstructionRecord[]
    attrs?: Record<string, unknown>
  }
  body?: string
  items_expanded?: string[]
  segments?: ComposedSegment[]
  tokens?: number
  blocks?: BuiltinBlock[]
  error?: OpError | string | null
  message?: string
  /** FastAPI-level rejection (auth/route), not the op envelope. */
  detail?: string
  /** transport status, attached client-side for legible errors */
  http_status?: number
}
export interface AppEntry {
  bundleId: string
  name: string
}
export interface AgentSlot {
  key: string
  container: 'agents' | 'root'
}
interface ProfileOption {
  id: string
  label?: string
  description?: string
  blocks?: string[]
  [key: string]: unknown
}
export interface AssignSource {
  optionId: string
  label: string
  description?: string
  ref: string
}

/** One agent as the CONFIG sees it: which roots hold its blocks.
 *  - reactContainer: where its react block lives (`react.agents.<key>` or
 *    direct `react.<key>`), or null when only the consumer block exists.
 *  - hasConsumer: a `surfaces.as_consumer.agents.<key>` block exists. */
export interface AgentProfile {
  key: string
  reactContainer: 'agents' | 'root' | null
  hasConsumer: boolean
}

export type AgentRoot = 'react' | 'consumer'

export function createAgenticConfigApi(transport: AgenticConfigTransport) {








  function opsUrl(): string {
    const tenant = encodeURIComponent(transport.tenant())
    const project = encodeURIComponent(transport.project())
    const bundle = encodeURIComponent(transport.opsBundle())
    return `${transport.baseUrl()}/api/integrations/bundles/${tenant}/${project}/${bundle}/operations/agentic_instructions`
  }

  async function call(data: Record<string, unknown>): Promise<OpEnvelope> {
    const url = opsUrl()
    const response = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: transport.authHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
      body: JSON.stringify({ data }),
    })
    const raw = (await response.json().catch(() => null)) as Record<string, unknown> | null
    if (!raw) throw new Error(`request failed (HTTP ${response.status} at ${url})`)
    // The operations route wraps the op's own result under its ALIAS key:
    // {status: "ok", tenant, project, bundle_id, agentic_instructions: {ok, ...}}.
    const inner = raw['agentic_instructions']
    const envelope = (
      inner && typeof inner === 'object' ? inner : raw
    ) as OpEnvelope
    envelope.http_status = response.status
    return envelope
  }

  function errorText(envelope: OpEnvelope): string {
    const err = envelope.error
    const status = envelope.http_status ? ` (HTTP ${envelope.http_status})` : ''
    if (err && typeof err === 'object') return (err.message || err.code) + status
    const text = String(err || envelope.message || envelope.detail || 'request failed')
    return text + status
  }

  async function listInstructions(
    includeRetired: boolean,
    q = '',
    tags: string[] = [],
  ): Promise<InstructionRecord[]> {
    const envelope = await call({ action: 'list', include_retired: includeRetired, q, tags })
    if (!envelope.ok) throw new Error(errorText(envelope))
    return envelope.ret?.items ?? []
  }

  /** The built-in block catalog (name, tier, description, tags) — what the
   *  constructor offers alongside stored units. */
  async function listBuiltinBlocks(): Promise<BuiltinBlock[]> {
    const envelope = await call({ action: 'blocks' })
    if (!envelope.ok) throw new Error(errorText(envelope))
    return envelope.blocks ?? []
  }

  async function getInstruction(ref: string): Promise<InstructionRecord> {
    const envelope = await call({ action: 'get', ref })
    if (!envelope.ok) throw new Error(errorText(envelope))
    const record = envelope.ret?.object
    if (!record) throw new Error('instruction payload missing')
    return record
  }

  async function saveVersion(input: {
    instruction_id: string
    name: string
    description: string
    tags: string[]
    signals: string[]
    items: string[]
  }): Promise<InstructionRecord> {
    const envelope = await call({ action: 'save', ...input })
    if (!envelope.ok) throw new Error(errorText(envelope))
    const record = envelope.ret?.object
    if (!record) throw new Error('instruction payload missing')
    return record
  }

  async function retireInstruction(ref: string): Promise<void> {
    const envelope = await call({ action: 'retire', ref })
    if (!envelope.ok) throw new Error(errorText(envelope))
  }

  async function previewBody(items: string[], workspaceImplementation = 'custom'): Promise<{
    body: string
    items_expanded: string[]
    segments: ComposedSegment[]
    tokens: number
  }> {
    const envelope = await call({
      action: 'preview',
      items,
      workspace_implementation: workspaceImplementation,
    })
    if (!envelope.ok) throw new Error(errorText(envelope))
    return {
      body: envelope.body ?? '',
      items_expanded: envelope.items_expanded ?? [],
      segments: envelope.segments ?? [],
      tokens: envelope.tokens ?? 0,
    }
  }

  /** Token weight of one composition item/set (its EXPANDED composition, as
   *  the runtime would build it) — used for the sidebar set counts. */
  async function tokensForItems(items: string[]): Promise<number> {
    const result = await previewBody(items)
    return result.tokens
  }

  /** The forge's cost meter: project a DRAFT agent config server-side and
   *  return the token breakdown of the REAL system instruction it would
   *  produce. `draft` mirrors the two config roots: {react?, consumer?}. */
  async function projectAgentConfig(
    draft: Record<string, unknown>,
    includeText = false,
  ): Promise<ProjectionResult> {
    const envelope = (await call({
      action: 'project',
      draft,
      include_text: includeText,
    })) as unknown as Partial<ProjectionResult> & { ok: boolean }
    if (!envelope.ok) throw new Error(errorText(envelope as OpEnvelope))
    return {
      tokens: envelope.tokens ?? {
        total: 0, protocol_and_instructions: 0, tool_catalog: 0, skill_gallery: 0,
      },
      facets: envelope.facets ?? {},
      tools: envelope.tools ?? { included_ids: [], skipped: [] },
      items_expanded: envelope.items_expanded ?? [],
      text: envelope.text,
    }
  }

  // ── assignment: wire a stored instruction to an application agent ────────────
  // Uses the platform admin routes (list bundles, read props, merge-write props).
  // Assignment adds/updates an instruction-profile OPTION on the target agent
  // whose id is the instruction slug and whose blocks wire the pinned ref —
  // user-pickable immediately; optionally also made the profile default.


  function adminBase(): string {
    return `${transport.baseUrl()}/admin/integrations/bundles`
  }

  function scopeQuery(): string {
    return `tenant=${encodeURIComponent(transport.tenant())}&project=${encodeURIComponent(transport.project())}`
  }

  async function adminGet(url: string): Promise<Record<string, unknown>> {
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include',
      headers: transport.authHeaders({ Accept: 'application/json' }),
      cache: 'no-store',
    })
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null
    if (!response.ok || !payload) {
      throw new Error(`${(payload as { detail?: string } | null)?.detail || 'request failed'} (HTTP ${response.status})`)
    }
    return payload
  }

  async function listApps(): Promise<AppEntry[]> {
    const payload = await adminGet(`${adminBase()}?${scopeQuery()}`)
    const entries = (payload.available_bundles ?? {}) as Record<string, { name?: string }>
    return Object.keys(entries)
      .sort()
      .map((bundleId) => ({ bundleId, name: entries[bundleId]?.name || bundleId }))
  }

  function deepMerge(base: unknown, over: unknown): unknown {
    if (over === undefined || over === null) return base
    if (typeof base !== 'object' || base === null || Array.isArray(base)) return over
    if (typeof over !== 'object' || Array.isArray(over)) return over
    const out: Record<string, unknown> = { ...(base as Record<string, unknown>) }
    for (const key of Object.keys(over as Record<string, unknown>)) {
      out[key] = deepMerge((base as Record<string, unknown>)[key], (over as Record<string, unknown>)[key])
    }
    return out
  }

  /** One discoverable agent: its key and WHERE it lives in the react config —
   *  `react.agents.<key>` (the agents container) or `react.<key>` (direct). */

  /** react-root keys that are configuration, not agents — mirrors the runtime's
   *  agent-key resolution (`react.<agent>` direct or `react.agents.<agent>`,
   *  with `default_agent` as the fallback block). */
  const NON_AGENT_REACT_KEYS = new Set([
    'agents', 'instructions', 'instruction_profiles', 'story_snapshots',
    'event_source_pipeline', 'supported_models', 'role_models', 'subagents',
    'render_thinking', 'line_numbers_mode', 'multi_action_mode', 'model',
  ])

  /** The app's EFFECTIVE config (code defaults ← stored props) + its agents. */
  async function getAppAgents(bundleId: string): Promise<{ agents: AgentSlot[]; config: Record<string, unknown> }> {
    const payload = await adminGet(`${adminBase()}/${encodeURIComponent(bundleId)}/props?${scopeQuery()}`)
    const config = deepMerge(payload.defaults ?? {}, payload.props ?? {}) as Record<string, unknown>
    const react = (config.react && typeof config.react === 'object' && !Array.isArray(config.react))
      ? (config.react as Record<string, unknown>)
      : {}
    const slots: AgentSlot[] = []
    const container = react.agents
    if (container && typeof container === 'object' && !Array.isArray(container)) {
      for (const key of Object.keys(container as Record<string, unknown>)) {
        const value = (container as Record<string, unknown>)[key]
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          slots.push({ key, container: 'agents' })
        }
      }
    }
    for (const key of Object.keys(react)) {
      if (NON_AGENT_REACT_KEYS.has(key)) continue
      const value = react[key]
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        slots.push({ key, container: 'root' })
      }
    }
    if (!slots.length) slots.push({ key: 'default_agent', container: 'root' })
    return { agents: slots, config }
  }


  /** What an assignment wires: a built-in set (instr:profile:*) or a stored
   *  set's pinned ref — one option shape either way. */

  async function writeAppProps(bundleId: string, props: Record<string, unknown>): Promise<void> {
    const response = await fetch(`${adminBase()}/${encodeURIComponent(bundleId)}/props`, {
      method: 'POST',
      credentials: 'include',
      headers: transport.authHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
      body: JSON.stringify({
        tenant: transport.tenant(),
        project: transport.project(),
        op: 'merge',
        props,
      }),
    })
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null
    if (!response.ok || !payload || payload.status !== 'ok') {
      throw new Error(`${(payload as { detail?: string } | null)?.detail || 'write failed'} (HTTP ${response.status})`)
    }
  }

  /** Add/update the instruction-profile option wiring `source` on the target
   *  agent (in its real container: react.agents.<key> or react.<key>), and
   *  merge-write the WHOLE instruction_profiles subtree (the admin merge
   *  replaces arrays, so the full options list rides the patch). */
  async function assignInstruction(
    bundleId: string,
    agent: AgentSlot,
    source: AssignSource,
    options: { makeDefault?: boolean } = {},
  ): Promise<{ optionId: string }> {
    const { config } = await getAppAgents(bundleId)
    const react = (config.react ?? {}) as Record<string, unknown>
    const holder = agent.container === 'agents'
      ? ((react.agents ?? {}) as Record<string, unknown>)
      : react
    const agentBlock = (holder[agent.key] ?? {}) as Record<string, unknown>
    const profiles = (agentBlock.instruction_profiles ?? {}) as Record<string, unknown>
    const existing: ProfileOption[] = Array.isArray(profiles.options)
      ? (profiles.options as ProfileOption[]).map((row) => ({ ...row }))
      : []
    const option: ProfileOption = {
      id: source.optionId,
      label: source.label || source.optionId,
      ...(source.description ? { description: source.description } : {}),
      blocks: [source.ref],
    }
    const index = existing.findIndex((row) => row.id === source.optionId)
    if (index >= 0) existing[index] = { ...existing[index], ...option }
    else existing.push(option)
    const nextProfiles: Record<string, unknown> = {
      ...profiles,
      options: existing,
      default: options.makeDefault
        ? source.optionId
        : (profiles.default ?? existing[0]?.id ?? source.optionId),
    }
    const agentPatch = { [agent.key]: { instruction_profiles: nextProfiles } }
    await writeAppProps(bundleId, {
      react: agent.container === 'agents' ? { agents: agentPatch } : agentPatch,
    })
    return { optionId: source.optionId }
  }

  // ── app settings: secrets (admin routes, values write-only) ──────────────────

  /** Redacted view of the app's stored secret keys. */
  async function getBundleSecretsRedacted(bundleId: string): Promise<Record<string, unknown>> {
    const payload = await adminGet(`${adminBase()}/${encodeURIComponent(bundleId)}/secrets?${scopeQuery()}`)
    return (payload.secrets ?? payload) as Record<string, unknown>
  }

  /** Set (or clear) app secrets from a nested map. Values are write-only. */
  async function setBundleSecrets(
    bundleId: string,
    secrets: Record<string, unknown>,
    mode: 'set' | 'clear' = 'set',
  ): Promise<void> {
    const response = await fetch(`${adminBase()}/${encodeURIComponent(bundleId)}/secrets`, {
      method: 'POST',
      credentials: 'include',
      headers: transport.authHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
      body: JSON.stringify({
        tenant: transport.tenant(),
        project: transport.project(),
        mode,
        secrets,
      }),
    })
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null
    if (!response.ok || !payload || payload.status !== 'ok') {
      throw new Error(`${(payload as { detail?: string } | null)?.detail || 'secrets write failed'} (HTTP ${response.status})`)
    }
  }


  /** Union view over BOTH per-agent roots: react slots + as_consumer agents. */
  async function getAgentProfiles(bundleId: string): Promise<{
    agents: AgentProfile[]
    config: Record<string, unknown>
  }> {
    const { agents: slots, config } = await getAppAgents(bundleId)
    const consumer = agentConsumerAgents(config)
    const byKey = new Map<string, AgentProfile>()
    for (const slot of slots) {
      byKey.set(slot.key, {
        key: slot.key,
        reactContainer: slot.container === 'agents' ? 'agents' : 'root',
        hasConsumer: false,
      })
    }
    for (const key of Object.keys(consumer)) {
      const existing = byKey.get(key)
      if (existing) existing.hasConsumer = true
      else byKey.set(key, { key, reactContainer: null, hasConsumer: true })
    }
    return { agents: [...byKey.values()].sort((a, b) => a.key.localeCompare(b.key)), config }
  }

  function agentConsumerAgents(config: Record<string, unknown>): Record<string, unknown> {
    const surfaces = (config.surfaces ?? {}) as Record<string, unknown>
    const asConsumer = (surfaces.as_consumer ?? {}) as Record<string, unknown>
    const agents = asConsumer.agents
    return agents && typeof agents === 'object' && !Array.isArray(agents)
      ? (agents as Record<string, unknown>)
      : {}
  }

  /** The agent's react block from the effective config (empty when absent). */
  function agentReactBlock(config: Record<string, unknown>, agent: AgentProfile): Record<string, unknown> {
    const react = (config.react ?? {}) as Record<string, unknown>
    const holder = agent.reactContainer === 'agents'
      ? ((react.agents ?? {}) as Record<string, unknown>)
      : react
    const block = holder[agent.key]
    if (block && typeof block === 'object' && !Array.isArray(block)) return block as Record<string, unknown>
    // fallback: the shared default_agent block configures agents without their own
    const fallback = react.default_agent
    return fallback && typeof fallback === 'object' && !Array.isArray(fallback)
      ? (fallback as Record<string, unknown>)
      : {}
  }

  /** The agent's as_consumer block from the effective config (empty when absent). */
  function agentConsumerBlock(config: Record<string, unknown>, agent: AgentProfile): Record<string, unknown> {
    const block = agentConsumerAgents(config)[agent.key]
    return block && typeof block === 'object' && !Array.isArray(block)
      ? (block as Record<string, unknown>)
      : {}
  }

  /** Merge-write one section of one agent's config into its REAL location.
   *  `section` null merges `value` as the whole agent block. New react blocks
   *  land in `react.agents.<key>` unless the agent already lives at the root. */
  async function writeAgentSection(
    bundleId: string,
    agent: AgentProfile,
    root: AgentRoot,
    section: string | null,
    value: unknown,
  ): Promise<void> {
    const payload = section === null ? value : { [section]: value }
    let props: Record<string, unknown>
    if (root === 'consumer') {
      props = { surfaces: { as_consumer: { agents: { [agent.key]: payload } } } }
    } else {
      const agentPatch = { [agent.key]: payload }
      props = {
        react: agent.reactContainer === 'root' ? agentPatch : { agents: agentPatch },
      }
    }
    await writeAppProps(bundleId, props)
  }


  return {
    errorText,
    listInstructions,
    listBuiltinBlocks,
    getInstruction,
    saveVersion,
    retireInstruction,
    previewBody,
    tokensForItems,
    projectAgentConfig,
    listApps,
    getAppAgents,
    writeAppProps,
    getAgentProfiles,
    agentReactBlock,
    agentConsumerBlock,
    writeAgentSection,
    assignInstruction,
    getBundleSecretsRedacted,
    setBundleSecrets,
    deepMerge,
  }
}

export type AgenticConfigApi = ReturnType<typeof createAgenticConfigApi>
