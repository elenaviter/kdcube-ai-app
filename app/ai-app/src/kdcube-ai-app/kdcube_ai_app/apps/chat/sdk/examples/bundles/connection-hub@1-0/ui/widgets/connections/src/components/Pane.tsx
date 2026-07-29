/**
 * Pane system for Connection Hub tabs — the workspace-scene window idea
 * (ui/scene windows.tsx) scaled down to a widget: each tab shows its areas
 * as PINNED panes stacked in one viewport-high column. A pinned pane takes
 * its content's height; when the column runs out, panes shrink and scroll
 * internally — the page itself never scrolls. A pane can be expanded to
 * fill the column, or unpinned into a floating draggable/resizable window
 * that opens at EXACTLY the geometry it had while pinned.
 */

import { useCallback, useEffect, useRef, useState, type ReactNode, type PointerEvent as ReactPointerEvent } from 'react';

const MIN_W = 320;
const MIN_H = 160;
const BASE_Z = 1000;

interface FloatRect {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
}

interface PaneState {
  floating: boolean;
  expanded: boolean;
  rect: FloatRect;
}

export interface PaneDef {
  id: string;
  title: string;
  content: ReactNode;
  /** The tab's lead pane: spans the full row at generous height while its
   *  siblings stay visible beneath — for a pane whose content is THE pending
   *  action (an agent access request). */
  lead?: boolean;
}

const ICON_UNPIN = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M7 17 17 7M9 7h8v8" />
  </svg>
);
const ICON_DOCK = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12h14M12 5l-7 7 7 7" />
  </svg>
);
const ICON_EXPAND = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
  </svg>
);
const ICON_COLLAPSE = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M8 3v3a2 2 0 0 1-2 2H3M16 3v3a2 2 0 0 0 2 2h3M21 16h-3a2 2 0 0 0-2 2v3M3 16h3a2 2 0 0 1 2 2v3" />
  </svg>
);

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function PaneGroup({ panes }: { panes: PaneDef[] }) {
  const [states, setStates] = useState<Record<string, PaneState>>({});
  const zRef = useRef(BASE_Z);
  const dockedEls = useRef<Record<string, HTMLElement | null>>({});
  const groupRef = useRef<HTMLDivElement | null>(null);

  // Layout diagnostics for the overflow bug: the panes size with viewport units
  // but live inside a modal that can be shorter than the viewport. Log the
  // group vs viewport vs each pane's clip so we can see, from real numbers,
  // whether a pane's body actually scrolls or its content is clipped
  // unreachable. Fires when the pane set or docking changes.
  const paneIds = panes.map((p) => p.id).join(',');
  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    const gr = group.getBoundingClientRect();
    const rows: string[] = [];
    Object.entries(dockedEls.current).forEach(([id, el]) => {
      if (!el) return;
      const body = el.querySelector('.pane-body') as HTMLElement | null;
      rows.push(
        `${id}: pane ${Math.round(el.clientHeight)}h/${Math.round(el.scrollHeight)}sc` +
        (body ? ` · body ${Math.round(body.clientHeight)}h/${Math.round(body.scrollHeight)}sc ${body.scrollHeight > body.clientHeight ? 'SCROLLS' : 'fits'}` : ''),
      );
    });
    // eslint-disable-next-line no-console
    console.log(
      `[connections.PaneGroup] viewport=${window.innerHeight} · group ${Math.round(gr.height)}h/${Math.round(group.scrollHeight)}sc ` +
      `${group.scrollHeight > group.clientHeight ? 'GROUP-SCROLLS' : 'group-fits'} · panes[${rows.join(' | ')}]`,
    );
  }, [paneIds, states]);

  const stateOf = (id: string): PaneState =>
    states[id] ?? { floating: false, expanded: false, rect: { x: 24, y: 24, w: MIN_W, h: MIN_H, z: BASE_Z } };

  const update = useCallback((id: string, patch: Partial<PaneState> | ((prev: PaneState) => PaneState)) => {
    setStates((current) => {
      const prev = current[id] ?? { floating: false, expanded: false, rect: { x: 24, y: 24, w: MIN_W, h: MIN_H, z: BASE_Z } };
      const next = typeof patch === 'function' ? patch(prev) : { ...prev, ...patch };
      return { ...current, [id]: next };
    });
  }, []);

  const front = useCallback((id: string) => {
    zRef.current += 2;
    const z = zRef.current;
    update(id, (prev) => ({ ...prev, rect: { ...prev.rect, z } }));
  }, [update]);

  // Unpin in place: the floating window opens at the pinned pane's exact
  // viewport geometry — the user resizes from there if needed.
  const unpin = useCallback((id: string) => {
    zRef.current += 2;
    const z = zRef.current;
    const el = dockedEls.current[id];
    const rect = el?.getBoundingClientRect();
    const w = clamp(Math.round(rect?.width ?? MIN_W), MIN_W, window.innerWidth - 8);
    const h = clamp(Math.round(rect?.height ?? MIN_H), MIN_H, window.innerHeight - 8);
    update(id, (prev) => ({
      ...prev,
      floating: true,
      expanded: false,
      rect: {
        x: clamp(Math.round(rect?.left ?? 24), 4, Math.max(4, window.innerWidth - w - 4)),
        y: clamp(Math.round(rect?.top ?? 24), 4, Math.max(4, window.innerHeight - h - 4)),
        w,
        h,
        z,
      },
    }));
  }, [update]);

  const startDrag = useCallback((id: string, event: ReactPointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest('button')) return;
    event.preventDefault();
    front(id);
    const startX = event.clientX;
    const startY = event.clientY;
    const origin = stateOf(id).rect;
    const onMove = (move: PointerEvent) => {
      update(id, (prev) => ({
        ...prev,
        rect: {
          ...prev.rect,
          x: clamp(origin.x + move.clientX - startX, 4, Math.max(4, window.innerWidth - 80)),
          y: clamp(origin.y + move.clientY - startY, 4, Math.max(4, window.innerHeight - 48)),
        },
      }));
    };
    const finish = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', finish);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', finish);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [front, states, update]);

  const startResize = useCallback((id: string, event: ReactPointerEvent<HTMLElement>) => {
    event.preventDefault();
    front(id);
    const startX = event.clientX;
    const startY = event.clientY;
    const origin = stateOf(id).rect;
    const onMove = (move: PointerEvent) => {
      update(id, (prev) => ({
        ...prev,
        rect: {
          ...prev.rect,
          w: clamp(origin.w + move.clientX - startX, MIN_W, window.innerWidth - 8),
          h: clamp(origin.h + move.clientY - startY, MIN_H, window.innerHeight - 8),
        },
      }));
    };
    const finish = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', finish);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', finish);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [front, states, update]);

  const resolved = panes.map((pane) => ({ pane, state: stateOf(pane.id) }));
  const docked = resolved.filter((item) => !item.state.floating);
  const expandedDocked = docked.find((item) => item.state.expanded);
  const visibleDocked = expandedDocked ? [expandedDocked] : docked;

  const renderBar = (item: (typeof resolved)[number]) => {
    const { pane, state } = item;
    return (
      <header
        className="pane-bar"
        onPointerDown={state.floating ? (event) => startDrag(pane.id, event) : undefined}
      >
        <span className="pane-title">{pane.title}</span>
        <span className="pane-controls">
          {!state.floating ? (
            <button
              type="button"
              className="pane-btn"
              title={state.expanded ? 'Restore' : 'Expand'}
              aria-label={state.expanded ? 'Restore' : 'Expand'}
              onClick={() => update(pane.id, { expanded: !state.expanded })}
            >
              {state.expanded ? ICON_COLLAPSE : ICON_EXPAND}
            </button>
          ) : null}
          <button
            type="button"
            className="pane-btn"
            title={state.floating ? 'Pin back into the page' : `Pop out ${pane.title}`}
            aria-label={state.floating ? 'Pin back into the page' : `Pop out ${pane.title}`}
            onClick={() => {
              if (state.floating) update(pane.id, { floating: false, expanded: false });
              else unpin(pane.id);
            }}
          >
            {state.floating ? ICON_DOCK : ICON_UNPIN}
          </button>
        </span>
      </header>
    );
  };

  // A lead pane spans the whole row, so the panes that share the two-column
  // grid are only the non-lead ones. When exactly one of those remains it has
  // no partner to sit beside — it must span too, instead of rendering as a
  // lonely half-width column next to empty space.
  const soloDocked = visibleDocked.filter((item) => !item.pane.lead).length === 1;

  return (
    <div className="pane-group" ref={groupRef}>
      {visibleDocked.map((item) => (
        <div
          key={item.pane.id}
          className={`pane${item.state.expanded ? ' pane--fill' : ''}${item.pane.lead ? ' pane--lead' : ''}${
            soloDocked && !item.pane.lead ? ' pane--solo' : ''}`}
          ref={(el) => {
            dockedEls.current[item.pane.id] = el;
          }}
        >
          {renderBar(item)}
          <div className="pane-body">{item.pane.content}</div>
        </div>
      ))}
      {resolved.filter((item) => item.state.floating).map((item) => (
        <section
          key={item.pane.id}
          className="pane pane--floating"
          style={{
            left: item.state.rect.x,
            top: item.state.rect.y,
            width: item.state.rect.w,
            height: item.state.rect.h,
            zIndex: item.state.rect.z,
          }}
          aria-label={item.pane.title}
          onPointerDownCapture={() => front(item.pane.id)}
        >
          {renderBar(item)}
          <div className="pane-body">{item.pane.content}</div>
          <button
            type="button"
            className="pane-grip"
            title="Resize"
            aria-label="Resize"
            onPointerDown={(event) => startResize(item.pane.id, event)}
          />
        </section>
      ))}
    </div>
  );
}
