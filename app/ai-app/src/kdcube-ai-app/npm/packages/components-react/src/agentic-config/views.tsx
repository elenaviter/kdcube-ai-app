/**
 * Agentic-config views — the instruction constructor (sets/blocks/signals,
 * continuous segmented composed rendering, Assign), the per-app AGENTS
 * editor (real agent slots as YAML), and APP SETTINGS (YAML/JSON merge
 * patches + secrets). Host-agnostic: mount <AgenticConfigProvider transport>
 * around <AgenticConfigTabs/> (or any single view). Writes stay admin-gated
 * server-side and land live via the platform admin routes.
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { dump as yamlDump, load as yamlLoad } from 'js-yaml'
import {
  createAgenticConfigApi,
  type AgentSlot,
  type AgenticConfigApi,
  type AgenticConfigTransport,
  type AppEntry,
  type AssignSource,
  type BuiltinBlock,
  type ComposedSegment,
  type InstructionRecord,
} from './api.ts'

const ApiContext = createContext<AgenticConfigApi | null>(null)

export function AgenticConfigProvider({
  transport,
  children,
}: {
  transport: AgenticConfigTransport
  children: ReactNode
}) {
  const [api] = useState(() => createAgenticConfigApi(transport))
  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
}

function useApi(): AgenticConfigApi {
  const api = useContext(ApiContext)
  if (!api) throw new Error('AgenticConfig views need <AgenticConfigProvider transport>.')
  return api
}

type Notice = { kind: 'success' | 'error'; text: string } | null

const useNotice = () => {
  const [notice, setNotice] = useState<Notice>(null)
  const fail = (err: unknown) =>
    setNotice({ kind: 'error', text: err instanceof Error ? err.message : String(err) })
  const ok = (text: string) => setNotice({ kind: 'success', text })
  return { notice, setNotice, fail, ok }
}

/** Parse an editor payload as YAML (JSON is valid YAML) into an object. */
function parseStructured(text: string): Record<string, unknown> {
  const parsed: unknown = yamlLoad(text)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('the payload must be a mapping (YAML or JSON object)')
  }
  return parsed as Record<string, unknown>
}

interface BuiltinSet {
  optionId: string
  ref: string
  label: string
  description: string
  signals: string[]
}

const BUILTIN_SETS: BuiltinSet[] = [
  {
    optionId: 'full', ref: 'instr:profile:full', label: 'Full',
    description: 'The complete default instruction body.',
    signals: ['Every signal the platform default body carries.'],
  },
  {
    optionId: 'lite', ref: 'instr:profile:lite', label: 'Lite (moderate)',
    description: 'The whole moderate profile (all capabilities).',
    signals: ['All moderate blocks across every capability.'],
  },
  {
    optionId: 'extra-lite', ref: 'instr:profile:extra-lite', label: 'Extra Lite',
    description: 'The distilled set for serving-constrained models.',
    signals: ['The extra-lite distillation for small models.'],
  },
]


type View = 'sets' | 'agents' | 'app'

export function AgenticConfigTabs() {
  const [view, setView] = useState<View>('sets')
  return (
    <div className="agc-shell">
      <nav className="agc-tabs">
        <button className={view === 'sets' ? 'is-active' : ''} onClick={() => setView('sets')}>
          Instruction sets
        </button>
        <button className={view === 'agents' ? 'is-active' : ''} onClick={() => setView('agents')}>
          Agents
        </button>
        <button className={view === 'app' ? 'is-active' : ''} onClick={() => setView('app')}>
          App settings
        </button>
      </nav>
      {view === 'sets' ? <SetsView /> : view === 'agents' ? <AgentsView /> : <AppSettingsView />}
    </div>
  )
}

/** Shared app+agent pickers backed by the platform admin list. */
function useApps() {
  const api = useApi()
  const [apps, setApps] = useState<AppEntry[]>([])
  const load = async () => {
    if (!apps.length) setApps(await api.listApps())
  }
  return { apps, load }
}

// ═══════════════════════════════ SETS VIEW ═══════════════════════════════

interface Draft {
  instruction_id: string
  name: string
  description: string
  /** comma-separated in the input; split on save */
  tags: string
  /** one signal statement per line; split on save */
  signals: string
  items: string[]
  loaded: InstructionRecord | null
}

const EMPTY_DRAFT: Draft = {
  instruction_id: '', name: '', description: '', tags: '', signals: '', items: [], loaded: null,
}

function SetsView() {
  const api = useApi()
  const { notice, setNotice, fail, ok } = useNotice()
  const [rows, setRows] = useState<InstructionRecord[]>([])
  const [includeRetired, setIncludeRetired] = useState(false)
  const [blocks, setBlocks] = useState<BuiltinBlock[]>([])
  const [libraryQuery, setLibraryQuery] = useState('')
  const [libraryOpen, setLibraryOpen] = useState(true)
  const [details, setDetails] = useState<BuiltinBlock | null>(null)
  const [builtinSet, setBuiltinSet] = useState<BuiltinSet | null>(null)
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [preview, setPreview] = useState<string>('')
  const [segments, setSegments] = useState<ComposedSegment[]>([])
  const [previewTokens, setPreviewTokens] = useState(0)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [busy, setBusy] = useState(false)
  /** token totals per set ref (EXPANDED composition), fetched lazily */
  const [setTokens, setSetTokens] = useState<Record<string, number>>({})

  const refresh = useCallback(async (retired: boolean) => {
    try {
      setRows(await api.listInstructions(retired))
    } catch (err) {
      fail(err)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    void refresh(includeRetired)
  }, [refresh, includeRetired])
  useEffect(() => {
    api.listBuiltinBlocks().then(setBlocks).catch(fail)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Token weight per sidebar set — each set previews once (its EXPANDED
  // composition, as the runtime would build it) and the count is cached.
  useEffect(() => {
    let alive = true
    const refs = [
      ...BUILTIN_SETS.map((s) => s.ref),
      ...rows.map((r) => r.ref),
    ].filter((ref) => setTokens[ref] === undefined)
    if (!refs.length) return
    void (async () => {
      for (const ref of refs) {
        try {
          const tokens = await api.tokensForItems([ref])
          if (!alive) return
          setSetTokens((m) => ({ ...m, [ref]: tokens }))
        } catch {
          if (!alive) return
        }
      }
    })()
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows])

  // continuous final rendering (debounced server-side compose)
  const cleanItems = draft.items.map((v) => v.trim()).filter(Boolean)
  const itemsKey = cleanItems.join(' ')
  const previewSeq = useRef(0)
  useEffect(() => {
    if (!cleanItems.length) {
      setPreview('')
      setSegments([])
      setPreviewTokens(0)
      return
    }
    const seq = ++previewSeq.current
    setPreviewBusy(true)
    const timer = window.setTimeout(() => {
      api.previewBody(cleanItems)
        .then((result) => {
          if (previewSeq.current === seq) {
            setPreview(result.body)
            setSegments(result.segments)
            setPreviewTokens(result.tokens)
          }
        })
        .catch((err) => {
          if (previewSeq.current === seq) fail(err)
        })
        .finally(() => {
          if (previewSeq.current === seq) setPreviewBusy(false)
        })
    }, 600)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemsKey])

  const open = async (ref: string) => {
    setBusy(true)
    setNotice(null)
    setBuiltinSet(null)
    try {
      const record = await api.getInstruction(ref)
      setDraft({
        instruction_id: record.instruction_id,
        name: record.name,
        description: record.description,
        tags: (record.tags ?? []).join(', '),
        signals: (record.signals ?? []).join('\n'),
        items: [...record.items],
        loaded: record,
      })
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    setBusy(true)
    setNotice(null)
    try {
      const record = await api.saveVersion({
        instruction_id: draft.instruction_id.trim().toLowerCase(),
        name: draft.name.trim(),
        description: draft.description.trim(),
        tags: draft.tags.split(',').map((t) => t.trim()).filter(Boolean),
        signals: draft.signals.split('\n').map((s) => s.trim()).filter(Boolean),
        items: cleanItems,
      })
      ok(`Saved ${record.ref} (by ${record.created_by})`)
      setDraft((d) => ({ ...d, loaded: record }))
      void refresh(includeRetired)
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const retire = async (ref: string) => {
    setBusy(true)
    setNotice(null)
    try {
      await api.retireInstruction(ref)
      ok(`Retired ${ref}`)
      void refresh(includeRetired)
      if (draft.loaded) void open(`instr:custom:${draft.loaded.instruction_id}:${draft.loaded.version}`)
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const setItem = (index: number, value: string) =>
    setDraft((d) => ({ ...d, items: d.items.map((v, i) => (i === index ? value : v)) }))
  const removeItem = (index: number) =>
    setDraft((d) => ({ ...d, items: d.items.filter((_, i) => i !== index) }))
  const moveItem = (index: number, delta: number) =>
    setDraft((d) => {
      const next = [...d.items]
      const target = index + delta
      if (target < 0 || target >= next.length) return d
      ;[next[index], next[target]] = [next[target], next[index]]
      return { ...d, items: next }
    })
  const appendItem = (value: string) => setDraft((d) => ({ ...d, items: [...d.items, value] }))

  const jumpToSource = (item: string) => {
    const builtin = blocks.find((b) => b.name === item)
    if (builtin) {
      setLibraryOpen(true)
      setDetails(builtin)
      return
    }
    const set = BUILTIN_SETS.find((s) => s.ref === item)
    if (set) {
      setBuiltinSet(set)
      return
    }
    if (item.startsWith('instr:custom:')) void open(item)
  }

  const query = libraryQuery.trim().toLowerCase()
  const blockMatches = (b: BuiltinBlock) =>
    !query ||
    b.name.toLowerCase().includes(query) ||
    b.description.toLowerCase().includes(query) ||
    b.signals.some((s) => s.toLowerCase().includes(query)) ||
    b.tags.some((t) => t.toLowerCase().includes(query))
  const libraryBlocks = blocks.filter(blockMatches)

  // What Assign wires: the selected built-in set, or the loaded stored set.
  const assignSource: AssignSource | null = builtinSet
    ? {
        optionId: builtinSet.optionId,
        label: builtinSet.label,
        description: builtinSet.description,
        ref: builtinSet.ref,
      }
    : draft.loaded
      ? {
          optionId: draft.loaded.instruction_id,
          label: draft.loaded.name || draft.loaded.instruction_id,
          description: draft.loaded.description || undefined,
          ref: `instr:custom:${draft.loaded.instruction_id}:${draft.loaded.version}`,
        }
      : null

  return (
    <div className="agc-root">
      <aside className="agc-side">
        <div className="agc-side-head">
          <h2>Sets</h2>
          <button
            className="agc-btn"
            onClick={() => { setDraft(EMPTY_DRAFT); setBuiltinSet(null); setNotice(null) }}
          >
            New
          </button>
        </div>

        <div className="agc-side-group">Built-in</div>
        {BUILTIN_SETS.map((set) => (
          <div key={set.ref} className="agc-list-wrap">
            <button
              className={'agc-list-row' + (builtinSet?.ref === set.ref ? ' is-active' : '')}
              onClick={() => { setBuiltinSet(set); setDraft(EMPTY_DRAFT); setNotice(null) }}
            >
              <span className="agc-list-name">
                {set.label}
                {setTokens[set.ref] !== undefined ? (
                  <span className="agc-tok">{formatTokens(setTokens[set.ref])}</span>
                ) : null}
              </span>
              <span className="agc-list-ref">{set.ref}</span>
            </button>
            <button className="agc-block-add" title="Insert into items" onClick={() => appendItem(set.ref)}>+</button>
          </div>
        ))}

        <div className="agc-side-group">
          My sets
          <label className="agc-check agc-check-inline">
            <input
              type="checkbox"
              checked={includeRetired}
              onChange={(e) => setIncludeRetired(e.target.checked)}
            />
            retired
          </label>
        </div>
        <div className="agc-list">
          {rows.map((row) => (
            <div key={row.instruction_id} className="agc-list-wrap">
              <button
                className={
                  'agc-list-row' + (draft.loaded?.instruction_id === row.instruction_id ? ' is-active' : '')
                }
                onClick={() => void open(row.ref)}
              >
                <span className="agc-list-name">
                  {row.name}
                  {setTokens[row.ref] !== undefined ? (
                    <span className="agc-tok">{formatTokens(setTokens[row.ref])}</span>
                  ) : null}
                </span>
                <span className="agc-list-ref">{row.ref}</span>
                {(row.tags ?? []).length ? (
                  <span className="agc-chiprow">
                    {(row.tags ?? []).map((t) => <span key={t} className="agc-chip">{t}</span>)}
                  </span>
                ) : null}
                {row.status !== 'active' ? <span className="agc-tag agc-tag-warn">{row.status}</span> : null}
              </button>
              <button className="agc-block-add" title="Insert ref into items" onClick={() => appendItem(row.ref)}>+</button>
            </div>
          ))}
          {!rows.length ? <div className="agc-empty">No stored sets yet.</div> : null}
        </div>
      </aside>

      <main className="agc-main">
        <div className="agc-help">
          Compose a set from the <strong>block library</strong> (click a block to
          see its signals; <strong>+</strong> adds it) or insert whole sets by
          ref from the sidebar. The <strong>composed instruction</strong> renders
          continuously, section by section — click a section label to jump to
          its source block. <em>Save</em> creates the next immutable version;
          <em> Assign</em> wires the selected set (built-in or yours) to an
          application agent as a pickable instruction profile.
        </div>
        {notice ? <div className={`agc-notice agc-notice-${notice.kind}`}>{notice.text}</div> : null}

        {builtinSet ? (
          <section className="agc-block-details agc-builtin-set">
            <div className="agc-block-details-head">
              <span className="agc-block-name">{builtinSet.label} — {builtinSet.ref}</span>
              {setTokens[builtinSet.ref] !== undefined ? (
                <span className="agc-tok">{formatTokens(setTokens[builtinSet.ref])}</span>
              ) : null}
              <button className="agc-btn" onClick={() => setBuiltinSet(null)}>Close</button>
            </div>
            <p className="agc-muted">{builtinSet.description}</p>
            <ul className="agc-signal-list">
              {builtinSet.signals.map((s) => <li key={s}>{s}</li>)}
            </ul>
            <AssignPanel source={assignSource!} busy={busy} onDone={ok} onFail={fail} />
          </section>
        ) : null}

        <div className="agc-columns">
          <section className="agc-editor">
            <div className="agc-field-row">
              <label className="agc-field">
                <span>Id (slug)</span>
                <input
                  value={draft.instruction_id}
                  disabled={Boolean(draft.loaded)}
                  placeholder="support-tone"
                  onChange={(e) => setDraft((d) => ({ ...d, instruction_id: e.target.value }))}
                />
              </label>
              <label className="agc-field">
                <span>Name</span>
                <input
                  value={draft.name}
                  placeholder="Support tone"
                  onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
                />
              </label>
            </div>
            <div className="agc-field-row">
              <label className="agc-field">
                <span>Description</span>
                <input
                  value={draft.description}
                  onChange={(e) => setDraft((d) => ({ ...d, description: e.target.value }))}
                />
              </label>
              <label className="agc-field">
                <span>Tags (comma-separated)</span>
                <input
                  value={draft.tags}
                  placeholder="tone, support"
                  onChange={(e) => setDraft((d) => ({ ...d, tags: e.target.value }))}
                />
              </label>
            </div>
            <label className="agc-field">
              <span>Signals (one per line — the behaviors this unit carries)</span>
              <textarea
                className="agc-signals-input"
                rows={2}
                value={draft.signals}
                placeholder={'Warm, empathetic support voice.\nNever promise refunds without a visible policy source.'}
                onChange={(e) => setDraft((d) => ({ ...d, signals: e.target.value }))}
              />
            </label>

            <div className="agc-items-head">
              <h3>Items (composed in order)</h3>
              <button className="agc-btn" onClick={() => appendItem('')}>Add literal text</button>
              <button className="agc-btn" onClick={() => setLibraryOpen((v) => !v)}>
                {libraryOpen ? 'Hide library' : 'Block library'}
              </button>
            </div>

            {libraryOpen ? (
              <div className="agc-library">
                <input
                  className="agc-library-search"
                  placeholder="Search blocks by name, signal, or tag…"
                  value={libraryQuery}
                  onChange={(e) => setLibraryQuery(e.target.value)}
                />
                {details ? (
                  <div className="agc-block-details">
                    <div className="agc-block-details-head">
                      <span className="agc-block-name">{details.name}</span>
                      {details.tokens !== undefined ? (
                        <span className="agc-tok">{formatTokens(details.tokens)}</span>
                      ) : null}
                      <button className="agc-btn" onClick={() => appendItem(details.name)}>
                        + Add to items
                      </button>
                      <button className="agc-btn" onClick={() => setDetails(null)}>Close</button>
                    </div>
                    {details.signals.length ? (
                      <ul className="agc-signal-list">
                        {details.signals.map((s) => <li key={s}>{s}</li>)}
                      </ul>
                    ) : null}
                    <span className="agc-chiprow">
                      {details.tags.map((t) => <span key={t} className="agc-chip">{t}</span>)}
                      {(details.profiles ?? []).map((pr) => (
                        <span key={pr} className="agc-chip agc-chip-muted">profile: {pr}</span>
                      ))}
                    </span>
                    {details.text ? <pre className="agc-block-text">{details.text}</pre> : null}
                  </div>
                ) : null}
                <div className="agc-library-scroll">
                  {libraryBlocks.map((b) => (
                    <LibraryCard key={b.name} block={b} onShow={() => setDetails(b)} onAdd={() => appendItem(b.name)} />
                  ))}
                  {!libraryBlocks.length ? (
                    <div className="agc-empty">No block matches "{libraryQuery}".</div>
                  ) : null}
                </div>
              </div>
            ) : null}

            <div className="agc-items">
              {draft.items.map((item, index) => (
                <div key={index} className="agc-item">
                  <textarea
                    rows={item.includes('\n') || item.length > 80 ? 4 : 1}
                    value={item}
                    placeholder="token or literal instruction text"
                    onChange={(e) => setItem(index, e.target.value)}
                  />
                  <div className="agc-item-actions">
                    <button title="Move up" onClick={() => moveItem(index, -1)}>↑</button>
                    <button title="Move down" onClick={() => moveItem(index, 1)}>↓</button>
                    <button title="Remove" onClick={() => removeItem(index)}>✕</button>
                  </div>
                </div>
              ))}
              {!draft.items.length ? (
                <div className="agc-empty">Add blocks from the library or literal text.</div>
              ) : null}
            </div>

            <div className="agc-actions">
              <button
                className="agc-btn agc-btn-primary"
                disabled={busy || !draft.instruction_id.trim() || !draft.name.trim() || !cleanItems.length}
                onClick={() => void save()}
              >
                {draft.loaded ? `Save as v${draft.loaded.version + 1}` : 'Save v1'}
              </button>
            </div>

            {draft.loaded && assignSource ? (
              <AssignPanel source={assignSource} busy={busy} onDone={ok} onFail={fail} />
            ) : null}

            {draft.loaded?.versions?.length ? (
              <section className="agc-versions">
                <h3>Versions</h3>
                {draft.loaded.versions.map((v) => (
                  <div key={v.version} className="agc-version-row">
                    <span className="agc-list-ref">instr:custom:{draft.loaded!.instruction_id}:{v.version}</span>
                    <span>{v.status}</span>
                    <span className="agc-muted">{v.created_by}{v.created_at ? ` · ${v.created_at}` : ''}</span>
                    {v.status === 'active' ? (
                      <button
                        className="agc-btn agc-btn-danger"
                        disabled={busy}
                        onClick={() => void retire(`instr:custom:${draft.loaded!.instruction_id}:${v.version}`)}
                      >
                        Retire
                      </button>
                    ) : null}
                  </div>
                ))}
              </section>
            ) : null}
          </section>

          <section className="agc-composed">
            <div className="agc-composed-head">
              <h3>Composed instruction</h3>
              {previewTokens && !previewBusy ? (
                <span className="agc-tok agc-tok-total">{formatTokens(previewTokens)} total</span>
              ) : null}
              {previewBusy ? <span className="agc-muted">rendering…</span> : null}
            </div>
            {segments.length ? (
              <div className="agc-composed-body agc-composed-segments">
                {segments.map((segment, index) => (
                  <div key={index} className="agc-segment">
                    <button
                      className="agc-segment-source"
                      title="Show the source block"
                      onClick={() => jumpToSource(segment.item)}
                    >
                      {segment.item.length > 72 ? `${segment.item.slice(0, 71)}…` : segment.item}
                    </button>
                    {segment.tokens !== undefined ? (
                      <span className="agc-tok">{formatTokens(segment.tokens)}</span>
                    ) : null}
                    <pre>{segment.body || '(empty)'}</pre>
                  </div>
                ))}
              </div>
            ) : (
              <pre className="agc-composed-body">
                {cleanItems.length
                  ? (preview || (previewBusy ? '' : '(empty body)'))
                  : 'Add items to see the final instruction an agent receives.'}
              </pre>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

/** Token counts are cl100k weights, shown compact (1234 → "1.2k tok"). */
function formatTokens(tokens: number): string {
  const n = tokens >= 1000 ? `${(tokens / 1000).toFixed(1).replace(/\.0$/, '')}k` : String(tokens)
  return `${n} tok`
}

function LibraryCard({
  block,
  onShow,
  onAdd,
}: {
  block: BuiltinBlock
  onShow: () => void
  onAdd: () => void
}) {
  return (
    <div className="agc-block-card" onClick={onShow} title="Show block details" role="button" tabIndex={0}>
      <div className="agc-block-card-row">
        <span className="agc-block-name">{block.name}</span>
        {block.tokens !== undefined ? <span className="agc-tok">{formatTokens(block.tokens)}</span> : null}
        <button
          className="agc-block-add"
          title="Add to items"
          onClick={(e) => { e.stopPropagation(); onAdd(); }}
        >
          +
        </button>
      </div>
      <span className="agc-block-desc">{block.description}</span>
      <span className="agc-chiprow">
        {block.tags.map((t) => <span key={t} className="agc-chip">{t}</span>)}
      </span>
    </div>
  )
}

/** App → agent → Assign, reusable for built-in and stored sources. */
function AssignPanel({
  source,
  busy,
  onDone,
  onFail,
}: {
  source: AssignSource
  busy: boolean
  onDone: (text: string) => void
  onFail: (err: unknown) => void
}) {
  const api = useApi()
  const { apps, load } = useApps()
  const [bundleId, setBundleId] = useState('')
  const [slots, setSlots] = useState<AgentSlot[]>([])
  const [slotIndex, setSlotIndex] = useState(-1)
  const [makeDefault, setMakeDefault] = useState(false)
  const [working, setWorking] = useState(false)

  const pickApp = async (id: string) => {
    setBundleId(id)
    setSlots([])
    setSlotIndex(-1)
    if (!id) return
    try {
      const result = await api.getAppAgents(id)
      setSlots(result.agents)
      setSlotIndex(result.agents.length ? 0 : -1)
    } catch (err) {
      onFail(err)
    }
  }

  const assign = async () => {
    const slot = slots[slotIndex]
    if (!bundleId || !slot) return
    setWorking(true)
    try {
      await api.assignInstruction(bundleId, slot, source, { makeDefault })
      onDone(
        `Assigned ${source.ref} to ${bundleId} → ${slotLabel(slot)} as option "${source.optionId}"` +
        (makeDefault ? ' (default)' : ''),
      )
    } catch (err) {
      onFail(err)
    } finally {
      setWorking(false)
    }
  }

  return (
    <section className="agc-assign">
      <h3>Assign to an application agent</h3>
      <p className="agc-assign-hint">
        Adds/updates the instruction-profile option "{source.optionId}" on the
        chosen agent, wiring {source.ref} — users can pick it immediately.
      </p>
      <div className="agc-assign-row">
        <select
          value={bundleId}
          onFocus={() => { void load().catch(onFail) }}
          onChange={(e) => void pickApp(e.target.value)}
        >
          <option value="">Application…</option>
          {apps.map((app) => (
            <option key={app.bundleId} value={app.bundleId}>{app.name}</option>
          ))}
        </select>
        <select
          value={slotIndex}
          disabled={!slots.length}
          onChange={(e) => setSlotIndex(Number(e.target.value))}
        >
          {!slots.length ? <option value={-1}>Agent…</option> : null}
          {slots.map((slot, index) => (
            <option key={`${slot.container}:${slot.key}`} value={index}>{slotLabel(slot)}</option>
          ))}
        </select>
        <label className="agc-check">
          <input type="checkbox" checked={makeDefault} onChange={(e) => setMakeDefault(e.target.checked)} />
          make default
        </label>
        <button
          className="agc-btn agc-btn-primary"
          disabled={busy || working || !bundleId || slotIndex < 0}
          onClick={() => void assign()}
        >
          Assign
        </button>
      </div>
    </section>
  )
}

function slotLabel(slot: AgentSlot): string {
  return slot.container === 'agents' ? `agents · ${slot.key}` : slot.key
}

// ═══════════════════════════════ AGENTS VIEW ═════════════════════════════
// The Agent FORGE: the full per-agent configuration, edited as a STAGED
// DRAFT. Every control and section stages into a local overlay over the
// stored config; nothing reaches the application until one explicit Apply
// (a single merge-write covering both roots). While forging, the projection
// meter shows what the TOTAL system instruction would weigh under the draft
// — reacting to instruction sets, tool catalog detail, the connected tool
// roster, the skill gallery, subagents, and the action protocol.

import type { AgentProfile, AgentRoot, ProjectionResult } from './api.ts'

interface SectionSpec {
  root: AgentRoot
  section: string
  title: string
  hint: string
  summarize: (value: unknown) => string
}

const AGENT_SECTIONS: SectionSpec[] = [
  {
    root: 'react', section: 'instruction_profiles', title: 'Instruction profiles',
    hint: 'The pickable instruction sets (default + options with blocks/facets). The Sets tab assigns stored sets here.',
    summarize: (v) => {
      const p = (v ?? {}) as { default?: string; options?: { id?: string }[] }
      const ids = (p.options ?? []).map((o) => o.id).filter(Boolean)
      return ids.length ? `default: ${p.default ?? ids[0]} · options: ${ids.join(', ')}` : '(none declared)'
    },
  },
  {
    root: 'react', section: 'instructions', title: 'Instructions (always-on)',
    hint: 'The agent-level instruction config: a blocks list (composition tokens) or a body, plus facet defaults.',
    summarize: (v) => {
      if (Array.isArray(v)) return `${v.length} block(s)`
      if (v && typeof v === 'object') return Object.keys(v as object).join(', ') || '(empty)'
      return v ? 'body text' : '(platform default)'
    },
  },
  {
    root: 'react', section: 'supported_models', title: 'Supported models',
    hint: 'The admin-allowed model list users pick from (model/provider/label, optional num_ctx).',
    summarize: (v) => Array.isArray(v)
      ? (v as { model?: string }[]).map((m) => m.model).filter(Boolean).join(', ') || '(none)'
      : '(none — picker hidden)',
  },
  {
    root: 'react', section: 'subagents', title: 'Subagents',
    hint: 'Helper-agent delegation: availability and defaults. Enabling adds the delegation tool to the instruction.',
    summarize: (v) => (v === undefined ? '(not offered)' : JSON.stringify(v)),
  },
  {
    root: 'consumer', section: 'tools', title: 'Tools & traits',
    hint: 'The tool connections (modules, MCP, named services) with allow-lists, runtimes, and tool_traits (parallel-execution strategy: exploration/exploitation). The connected roster sizes the catalog inside the instruction.',
    summarize: (v) => {
      if (!Array.isArray(v)) return '(none)'
      const names = (v as { name?: string; tool_traits?: unknown }[]).map(
        (t) => `${t.name}${t.tool_traits ? '·traits' : ''}`,
      )
      return `${names.length} connection(s): ${names.join(', ')}`
    },
  },
  {
    root: 'consumer', section: 'skills', title: 'Skills',
    hint: 'Per-consumer skill enablement (e.g. public.* plus app skills).',
    summarize: (v) => {
      const consumers = ((v ?? {}) as { consumers?: Record<string, unknown> }).consumers
      return consumers ? `consumers: ${Object.keys(consumers).join(', ')}` : '(default set)'
    },
  },
  {
    root: 'consumer', section: 'event_sources', title: 'Event sources',
    hint: 'Reactive event sources (named-service namespaces with discovery + block-production/pull policies).',
    summarize: (v) => Array.isArray(v)
      ? (v as { namespace?: string; kind?: string; enabled?: boolean }[])
          .map((e) => `${e.namespace ?? e.kind}${e.enabled === false ? ' (off)' : ''}`)
          .join(', ') || '(none)'
      : '(none)',
  },
  {
    root: 'consumer', section: 'model', title: 'Model (serving)',
    hint: 'Per-agent serving parameters (e.g. max_tokens).',
    summarize: (v) => (v && typeof v === 'object' ? JSON.stringify(v) : '(defaults)'),
  },
  {
    root: 'consumer', section: 'capabilities', title: 'Capabilities',
    hint: 'The capability-provider inventory (e.g. simple_model_pick: role, default, supported models).',
    summarize: (v) => {
      const models = ((v ?? {}) as { models?: { role?: string; supported?: unknown[] } }).models
      return models ? `models: role ${models.role ?? '?'} · ${models.supported?.length ?? 0} supported` : '(none)'
    },
  },
]

type Overlay = Record<string, unknown>

function AgentsView() {
  const api = useApi()
  const { notice, setNotice, fail, ok } = useNotice()
  const { apps, load } = useApps()
  const [bundleId, setBundleId] = useState('')
  const [profiles, setProfiles] = useState<AgentProfile[]>([])
  const [config, setConfig] = useState<Record<string, unknown>>({})
  const [selected, setSelected] = useState<AgentProfile | null>(null)
  const [draftReact, setDraftReact] = useState<Overlay>({})
  const [draftConsumer, setDraftConsumer] = useState<Overlay>({})
  const [draftOnly, setDraftOnly] = useState<AgentProfile[]>([])
  const [newAgentKey, setNewAgentKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [projBusy, setProjBusy] = useState(false)

  const dirty = Object.keys(draftReact).length > 0 || Object.keys(draftConsumer).length > 0

  useEffect(() => {
    void load().catch(fail)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const clearDraft = () => {
    setDraftReact({})
    setDraftConsumer({})
    setProjection(null)
  }

  const reload = async (id: string, keepKey?: string) => {
    const result = await api.getAgentProfiles(id)
    setProfiles(result.agents)
    setConfig(result.config)
    if (keepKey) {
      setSelected(result.agents.find((a) => a.key === keepKey) ?? null)
      setDraftOnly((list) => list.filter((a) => !result.agents.some((r) => r.key === a.key)))
    }
  }

  const pickApp = async (id: string) => {
    if (dirty && !window.confirm('Discard the staged draft?')) return
    setBundleId(id)
    setProfiles([])
    setDraftOnly([])
    setConfig({})
    setSelected(null)
    clearDraft()
    setNotice(null)
    if (!id) return
    try {
      await reload(id)
    } catch (err) {
      fail(err)
    }
  }

  const pickAgent = (agent: AgentProfile) => {
    if (agent.key === selected?.key) return
    if (dirty && !window.confirm('Discard the staged draft?')) return
    setSelected(agent)
    clearDraft()
    setNotice(null)
  }

  const storedReact = selected ? (api.agentReactBlock(config, selected) as Overlay) : {}
  const storedConsumer = selected ? (api.agentConsumerBlock(config, selected) as Overlay) : {}
  const effectiveReact = api.deepMerge(storedReact, draftReact) as Overlay
  const effectiveConsumer = api.deepMerge(storedConsumer, draftConsumer) as Overlay

  /** Stage a value: section === null merges a partial object at the root. */
  const stage = (root: AgentRoot, section: string | null, value: unknown) => {
    const update = (prev: Overlay): Overlay =>
      section === null ? { ...prev, ...(value as Overlay) } : { ...prev, [section]: value }
    if (root === 'react') setDraftReact(update)
    else setDraftConsumer(update)
  }
  const unstage = (root: AgentRoot, section: string) => {
    const drop = (prev: Overlay): Overlay => {
      const next = { ...prev }
      delete next[section]
      return next
    }
    if (root === 'react') setDraftReact(drop)
    else setDraftConsumer(drop)
  }

  // The projection meter: debounced server projection of the EFFECTIVE draft.
  const projKey = selected
    ? JSON.stringify({ r: effectiveReact, c: effectiveConsumer })
    : ''
  const projSeq = useRef(0)
  useEffect(() => {
    if (!selected) {
      setProjection(null)
      return
    }
    const seq = ++projSeq.current
    setProjBusy(true)
    const timer = window.setTimeout(() => {
      api.projectAgentConfig({ react: effectiveReact, consumer: effectiveConsumer })
        .then((result) => {
          if (projSeq.current === seq) setProjection(result)
        })
        .catch(() => {
          if (projSeq.current === seq) setProjection(null)
        })
        .finally(() => {
          if (projSeq.current === seq) setProjBusy(false)
        })
    }, 700)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projKey])

  const apply = async () => {
    if (!selected || !dirty) return
    setBusy(true)
    setNotice(null)
    try {
      const props: Record<string, unknown> = {}
      if (Object.keys(draftReact).length) {
        const agentPatch = { [selected.key]: draftReact }
        props.react = selected.reactContainer === 'root' ? agentPatch : { agents: agentPatch }
      }
      if (Object.keys(draftConsumer).length) {
        props.surfaces = { as_consumer: { agents: { [selected.key]: draftConsumer } } }
      }
      await api.writeAppProps(bundleId, props)
      ok(`Applied the draft to ${selected.key}. The change is live.`)
      clearDraft()
      await reload(bundleId, selected.key)
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const addAgent = () => {
    const key = newAgentKey.trim()
    if (!bundleId || !key) return
    if (profiles.some((a) => a.key === key) || draftOnly.some((a) => a.key === key)) return
    if (dirty && !window.confirm('Discard the staged draft?')) return
    const agent: AgentProfile = { key, reactContainer: 'agents', hasConsumer: false }
    setDraftOnly((list) => [...list, agent])
    setSelected(agent)
    clearDraft()
    setNewAgentKey('')
    setNotice(null)
  }

  const allAgents = [...profiles, ...draftOnly.filter((d) => !profiles.some((p) => p.key === d.key))]
  const blockFor = (root: AgentRoot) => (root === 'react' ? effectiveReact : effectiveConsumer)
  const stagedIn = (root: AgentRoot) => (root === 'react' ? draftReact : draftConsumer)

  return (
    <div className="agc-root">
      <aside className="agc-side">
        <div className="agc-side-head"><h2>Agents</h2></div>
        <div className="agc-side-pad">
          <select value={bundleId} onChange={(e) => void pickApp(e.target.value)}>
            <option value="">Application…</option>
            {apps.map((app) => (
              <option key={app.bundleId} value={app.bundleId}>{app.name}</option>
            ))}
          </select>
        </div>
        <div className="agc-list">
          {allAgents.map((agent) => (
            <button
              key={agent.key}
              className={'agc-list-row' + (selected?.key === agent.key ? ' is-active' : '')}
              onClick={() => pickAgent(agent)}
            >
              <span className="agc-list-name">
                {agent.key}
                {draftOnly.some((d) => d.key === agent.key) ? (
                  <span className="agc-chip agc-chip-staged">draft</span>
                ) : null}
              </span>
              <span className="agc-list-ref">
                {[
                  agent.reactContainer ? (agent.reactContainer === 'agents' ? 'react.agents' : 'react') : null,
                  agent.hasConsumer ? 'as_consumer' : null,
                ].filter(Boolean).join(' + ') || 'new'}
              </span>
            </button>
          ))}
          {bundleId && !allAgents.length ? <div className="agc-empty">No agents found.</div> : null}
        </div>
        {bundleId ? (
          <div className="agc-side-pad agc-add-agent">
            <input
              value={newAgentKey}
              placeholder="new-agent-key"
              onChange={(e) => setNewAgentKey(e.target.value)}
            />
            <button className="agc-btn" disabled={!newAgentKey.trim()} onClick={addAgent}>
              Add agent
            </button>
          </div>
        ) : null}
      </aside>

      <main className="agc-main">
        <div className="agc-help">
          The agent forge. Every edit STAGES into a draft — nothing reaches the
          application until <strong>Apply</strong> writes the whole draft in one
          merge to the agent's real config
          (<code>surfaces.as_consumer.agents</code> + <code>react…</code>).
          The projection meter shows what the TOTAL system instruction would
          weigh under the draft: instruction sets, catalog detail, the
          connected tool roster, skill gallery, subagents, protocol.
        </div>
        {notice ? <div className={`agc-notice agc-notice-${notice.kind}`}>{notice.text}</div> : null}
        {selected ? (
          <>
            <ProjectionMeter projection={projection} busy={projBusy} />
            {dirty ? (
              <DraftBar
                draftReact={draftReact}
                draftConsumer={draftConsumer}
                busy={busy}
                onApply={() => void apply()}
                onDiscard={() => { clearDraft(); setNotice(null) }}
              />
            ) : null}
            <QuickControls
              reactBlock={effectiveReact}
              consumerBlock={effectiveConsumer}
              stagedReact={draftReact}
              onStage={stage}
            />
            {AGENT_SECTIONS.map((spec) => (
              <AgentSection
                key={`${spec.root}:${spec.section}`}
                spec={spec}
                value={(blockFor(spec.root))[spec.section]}
                staged={spec.section in stagedIn(spec.root)}
                onStage={(value) => stage(spec.root, spec.section, value)}
                onUnstage={() => unstage(spec.root, spec.section)}
              />
            ))}
          </>
        ) : (
          <p className="agc-empty">Pick an application and an agent.</p>
        )}
      </main>
    </div>
  )
}

/** The forge's cost meter: total + breakdown of the projected instruction. */
function ProjectionMeter({
  projection,
  busy,
}: {
  projection: ProjectionResult | null
  busy: boolean
}) {
  return (
    <section className="agc-projection">
      <span className="agc-projection-title">Projected instruction</span>
      {projection ? (
        <>
          <span className="agc-tok agc-tok-total">{formatTokens(projection.tokens.total)} total</span>
          <span className="agc-tok">base+instructions {formatTokens(projection.tokens.protocol_and_instructions)}</span>
          <span className="agc-tok">tool catalog {formatTokens(projection.tokens.tool_catalog)}</span>
          <span className="agc-tok">skill gallery {formatTokens(projection.tokens.skill_gallery)}</span>
          <span className="agc-muted">
            {projection.tools.included_ids.length} tool(s) projected
            {projection.tools.skipped.length
              ? ` · not projected: ${projection.tools.skipped.join('; ')}`
              : ''}
          </span>
        </>
      ) : (
        <span className="agc-muted">{busy ? '' : 'no projection yet'}</span>
      )}
      {busy ? <span className="agc-muted">projecting…</span> : null}
    </section>
  )
}

/** The staged draft: what Apply will merge-write, per root. */
function DraftBar({
  draftReact,
  draftConsumer,
  busy,
  onApply,
  onDiscard,
}: {
  draftReact: Record<string, unknown>
  draftConsumer: Record<string, unknown>
  busy: boolean
  onApply: () => void
  onDiscard: () => void
}) {
  const [open, setOpen] = useState(false)
  const staged = [
    ...Object.keys(draftReact).map((k) => `react.${k}`),
    ...Object.keys(draftConsumer).map((k) => `as_consumer.${k}`),
  ]
  return (
    <section className="agc-draftbar">
      <div className="agc-draftbar-row">
        <span className="agc-chip agc-chip-staged">staged</span>
        <span className="agc-draftbar-list">{staged.join(' · ')}</span>
        <button className="agc-btn" onClick={() => setOpen((v) => !v)}>
          {open ? 'Hide' : 'Show'} draft
        </button>
        <button className="agc-btn" disabled={busy} onClick={onDiscard}>Discard</button>
        <button className="agc-btn agc-btn-primary" disabled={busy} onClick={onApply}>
          Apply draft
        </button>
      </div>
      {open ? (
        <pre className="agc-block-text">
          {yamlDump(
            {
              ...(Object.keys(draftReact).length ? { react: draftReact } : {}),
              ...(Object.keys(draftConsumer).length ? { as_consumer: draftConsumer } : {}),
            },
            { lineWidth: 100 },
          )}
        </pre>
      ) : null}
    </section>
  )
}

/** The high-value scalars as typed controls. Picking STAGES into the draft;
 *  the blank option shows the current effective (stored ⊕ staged) value. */
function QuickControls({
  reactBlock,
  consumerBlock,
  stagedReact,
  onStage,
}: {
  reactBlock: Record<string, unknown>
  consumerBlock: Record<string, unknown>
  stagedReact: Record<string, unknown>
  onStage: (root: AgentRoot, section: string | null, value: unknown) => void
}) {
  const instructions = (reactBlock.instructions ?? {}) as Record<string, unknown>
  const instructionsObj = typeof instructions === 'object' && !Array.isArray(instructions) ? instructions : {}
  const pipeline = (reactBlock.event_source_pipeline ?? {}) as Record<string, unknown>
  const snapshots = (reactBlock.story_snapshots ?? {}) as Record<string, unknown>
  const model = (consumerBlock.model ?? {}) as Record<string, unknown>
  const [maxTokens, setMaxTokens] = useState('')

  const stageInstructionFacet = (facet: string, value: unknown) => {
    const stagedInstr = (stagedReact.instructions ?? {}) as Record<string, unknown>
    const base = typeof stagedInstr === 'object' && !Array.isArray(stagedInstr) ? stagedInstr : {}
    onStage('react', 'instructions', { ...base, [facet]: value })
  }

  const enumControl = (
    label: string,
    current: unknown,
    values: string[],
    write: (value: string) => void,
  ) => (
    <label className="agc-quick" key={label}>
      <span>{label}</span>
      <select value="" onChange={(e) => { if (e.target.value) write(e.target.value) }}>
        <option value="">{current === undefined ? '(inherit)' : String(current)}</option>
        {values.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
    </label>
  )

  return (
    <section className="agc-quickbar">
      {enumControl('Tool catalog', instructionsObj.tool_catalog_detail, ['full', 'compact'], (v) =>
        stageInstructionFacet('tool_catalog_detail', v))}
      {enumControl('Skills form', instructionsObj.skills_form, ['full', 'compact'], (v) =>
        stageInstructionFacet('skills_form', v))}
      {enumControl('Skill gallery', instructionsObj.include_skill_gallery, ['true', 'false'], (v) =>
        stageInstructionFacet('include_skill_gallery', v === 'true'))}
      {enumControl('Multi-action', reactBlock.multi_action_mode, ['off', 'on'], (v) =>
        onStage('react', null, { multi_action_mode: v }))}
      {enumControl('Event pipeline', pipeline.enabled, ['true', 'false'], (v) =>
        onStage('react', 'event_source_pipeline', { enabled: v === 'true' }))}
      {enumControl('Story snapshots', snapshots.enabled, ['true', 'false'], (v) =>
        onStage('react', 'story_snapshots', { enabled: v === 'true' }))}
      <label className="agc-quick">
        <span>max_tokens</span>
        <span className="agc-quick-pair">
          <input
            value={maxTokens}
            placeholder={model.max_tokens !== undefined ? String(model.max_tokens) : '(inherit)'}
            onChange={(e) => setMaxTokens(e.target.value)}
          />
          <button
            className="agc-btn"
            disabled={!/^\d+$/.test(maxTokens.trim())}
            onClick={() => { onStage('consumer', 'model', { max_tokens: Number(maxTokens) }); setMaxTokens('') }}
          >
            Stage
          </button>
        </span>
      </label>
    </section>
  )
}

/** One configurable area: human summary + YAML editor staging into the draft. */
function AgentSection({
  spec,
  value,
  staged,
  onStage,
  onUnstage,
}: {
  spec: SectionSpec
  value: unknown
  staged: boolean
  onStage: (value: unknown) => void
  onUnstage: () => void
}) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [error, setError] = useState('')

  const openEditor = () => {
    setText(value === undefined ? '' : yamlDump(value, { lineWidth: 100 }))
    setError('')
    setOpen(true)
  }

  const stageText = () => {
    try {
      const parsed: unknown = yamlLoad(text)
      if (parsed === undefined || parsed === null) throw new Error('the section payload is empty')
      onStage(parsed)
      setError('')
      setOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <section className="agc-agent-section">
      <div className="agc-agent-section-head">
        <div className="agc-agent-section-title">
          <h3>{spec.title}</h3>
          <span className="agc-chip agc-chip-muted">
            {spec.root === 'consumer' ? 'as_consumer' : 'react'}
          </span>
          {staged ? <span className="agc-chip agc-chip-staged">staged</span> : null}
        </div>
        <span className="agc-muted">{spec.summarize(value)}</span>
        {staged ? (
          <button className="agc-btn" onClick={onUnstage}>Unstage</button>
        ) : null}
        <button className="agc-btn" onClick={() => (open ? setOpen(false) : openEditor())}>
          {open ? 'Close' : 'Edit'}
        </button>
      </div>
      {open ? (
        <>
          <p className="agc-assign-hint">{spec.hint}</p>
          {error ? <div className="agc-notice agc-notice-error">{error}</div> : null}
          <textarea
            className="agc-yaml-editor agc-yaml-editor-small"
            spellCheck={false}
            value={text}
            placeholder="YAML for this section"
            onChange={(e) => setText(e.target.value)}
          />
          <div className="agc-actions">
            <button className="agc-btn agc-btn-primary" disabled={!text.trim()} onClick={stageText}>
              Stage section
            </button>
          </div>
        </>
      ) : null}
    </section>
  )
}

// ═══════════════════════════ APP SETTINGS VIEW ═══════════════════════════

function AppSettingsView() {
  const api = useApi()
  const { notice, setNotice, fail, ok } = useNotice()
  const { apps, load } = useApps()
  const [bundleId, setBundleId] = useState('')
  const [configText, setConfigText] = useState('')
  const [patchText, setPatchText] = useState('')
  const [secretsView, setSecretsView] = useState('')
  const [secretsText, setSecretsText] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void load().catch(fail)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pickApp = async (id: string) => {
    setBundleId(id)
    setConfigText('')
    setPatchText('')
    setSecretsView('')
    setNotice(null)
    if (!id) return
    try {
      const { config } = await api.getAppAgents(id)
      setConfigText(yamlDump(config, { lineWidth: 110 }))
    } catch (err) {
      fail(err)
    }
    try {
      const redacted = await api.getBundleSecretsRedacted(id)
      setSecretsView(yamlDump(redacted, { lineWidth: 110 }))
    } catch {
      setSecretsView('(secrets not readable on this deployment)')
    }
  }

  const applyPatch = async () => {
    if (!bundleId) return
    setBusy(true)
    setNotice(null)
    try {
      const patch = parseStructured(patchText)
      await api.writeAppProps(bundleId, patch)
      ok('Config merged. Reloading the stored view…')
      setPatchText('')
      const { config } = await api.getAppAgents(bundleId)
      setConfigText(yamlDump(config, { lineWidth: 110 }))
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const applySecrets = async (mode: 'set' | 'clear') => {
    if (!bundleId) return
    setBusy(true)
    setNotice(null)
    try {
      const secrets = parseStructured(secretsText)
      await api.setBundleSecrets(bundleId, secrets, mode)
      ok(mode === 'set' ? 'Secrets stored (values are write-only).' : 'Secrets cleared.')
      setSecretsText('')
      try {
        setSecretsView(yamlDump(await api.getBundleSecretsRedacted(bundleId), { lineWidth: 110 }))
      } catch { /* redacted view optional */ }
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="agc-main agc-appsettings">
      <div className="agc-help">
        The application's FULL effective config (roles and visibility on
        surfaces, agent settings, everything) and its secrets. Patches are
        MERGE-written as <strong>YAML or JSON</strong> through the platform
        admin routes and land live; secret values are write-only (the stored
        view lists keys redacted).
      </div>
      {notice ? <div className={`agc-notice agc-notice-${notice.kind}`}>{notice.text}</div> : null}
      <div className="agc-side-pad">
        <select value={bundleId} onChange={(e) => void pickApp(e.target.value)}>
          <option value="">Application…</option>
          {apps.map((app) => (
            <option key={app.bundleId} value={app.bundleId}>{app.name}</option>
          ))}
        </select>
      </div>
      {bundleId ? (
        <div className="agc-columns">
          <section className="agc-editor">
            <h3>Effective config (read view)</h3>
            <pre className="agc-config-view">{configText}</pre>
            <h3>Merge patch (YAML or JSON)</h3>
            <textarea
              className="agc-yaml-editor agc-yaml-editor-small"
              spellCheck={false}
              placeholder={'surfaces:\n  as_provider:\n    bundle:\n      visibility:\n        allowed_roles: [kdcube:role:registered]'}
              value={patchText}
              onChange={(e) => setPatchText(e.target.value)}
            />
            <div className="agc-actions">
              <button className="agc-btn agc-btn-primary" disabled={busy || !patchText.trim()} onClick={() => void applyPatch()}>
                Apply merge patch
              </button>
            </div>
          </section>
          <section className="agc-editor">
            <h3>Secrets (stored keys, redacted)</h3>
            <pre className="agc-config-view">{secretsView || '(none)'}</pre>
            <h3>Set / clear secrets (YAML or JSON, nested)</h3>
            <textarea
              className="agc-yaml-editor agc-yaml-editor-small"
              spellCheck={false}
              placeholder={'services:\n  llm:\n    custom:\n      api_key: "<value>"'}
              value={secretsText}
              onChange={(e) => setSecretsText(e.target.value)}
            />
            <div className="agc-actions">
              <button className="agc-btn agc-btn-primary" disabled={busy || !secretsText.trim()} onClick={() => void applySecrets('set')}>
                Set secrets
              </button>
              <button className="agc-btn agc-btn-danger" disabled={busy || !secretsText.trim()} onClick={() => void applySecrets('clear')}>
                Clear listed keys
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  )
}
