import assert from 'node:assert/strict'
import { test } from 'node:test'
import { resolveComponentSpecs, normalizeExternalPanelConfig } from '../dist/scene/index.js'

const defaults = [
  {
    alias: 'pinboard', bundleId: '', widgetAlias: 'pinboard', title: 'Pin Board',
    accent: 'pink', gated: true, views: false, size: { w: 720, h: 560 },
    targetSurfaces: ['sdk.canvas.pinboard'], drop: { effect: 'pin', patterns: ['*'] },
    placement: 'docked', rail: true, defaultOpen: false, enabled: true, order: 10,
  },
  {
    alias: 'usage', bundleId: '', widgetAlias: 'usage_card', title: 'Usage',
    accent: 'gold', gated: true, views: false, size: { w: 380, h: 520 },
    targetSurfaces: ['sdk.usage.card'], placement: 'floating', rail: true,
    defaultOpen: false, enabled: true, order: 60,
  },
]

test('config entries override defaults by alias and sort by order', () => {
  const specs = resolveComponentSpecs({
    pinboard: { accent: 'teal', order: 70 },
    stats: {
      bundle_id: 'kdcube.stats@2026-05-20-12-05', widget_alias: 'usage',
      route: 'public/widgets/usage', title: 'Stats', accent: 'orange',
      gated: false, views: true, size: { w: 720, h: 520 }, order: 40,
    },
  }, defaults)
  assert.deepEqual(specs.map((s) => s.alias), ['stats', 'usage', 'pinboard'])
  const pin = specs.find((s) => s.alias === 'pinboard')
  assert.equal(pin.accent, 'teal')
  assert.equal(pin.placement, 'docked')
  const stats = specs.find((s) => s.alias === 'stats')
  assert.equal(stats.route, 'public/widgets/usage')
  assert.equal(stats.gated, false)
})

test('enabled:false removes a default; docked flag maps to placement', () => {
  const specs = resolveComponentSpecs({
    usage: { enabled: false },
    board2: { widget_alias: 'pinboard', docked: true, drop: { effect: 'open', patterns: ['mem:*'], target_surface: 'sdk.memory.viewer' } },
  }, defaults)
  assert.equal(specs.some((s) => s.alias === 'usage'), false)
  const board2 = specs.find((s) => s.alias === 'board2')
  assert.equal(board2.placement, 'docked')
  assert.equal(board2.drop.targetSurface, 'sdk.memory.viewer')
})

test('normalizes explicit receiver readiness without changing components that omit it', () => {
  const specs = resolveComponentSpecs({
    pinboard: {},
    usage: {
      ready: {
        type: 'message',
        message_type: 'kdcube.surface.ready',
        fallback_delay_ms: 6000,
      },
    },
  }, defaults)

  assert.equal(specs.find((s) => s.alias === 'pinboard').ready, undefined)
  assert.deepEqual(specs.find((s) => s.alias === 'usage').ready, {
    type: 'message',
    messageType: 'kdcube.surface.ready',
    fallbackDelayMs: 6000,
  })
})

test('external panel config normalizes surfaces and message types', () => {
  const panel = normalizeExternalPanelConfig({
    id: 'task_panel', label: 'Tasks', bundle_id: 'task-tracker@1-0',
    widget_alias: 'task_tracker_tasks',
    open_message_types: ['kdcube-task-tracker-open-issue'],
    surfaces: { 'task_tracker.issue_list': { label: 'task list', command: { action: 'refresh' } } },
  })
  assert.equal(panel.title, 'Tasks')
  assert.deepEqual(panel.open_message_types, ['kdcube-task-tracker-open-issue'])
  assert.deepEqual(panel.surfaces['task_tracker.issue_list'].command, { action: 'refresh' })
  assert.equal(normalizeExternalPanelConfig({ id: 'x' }), null)
})
