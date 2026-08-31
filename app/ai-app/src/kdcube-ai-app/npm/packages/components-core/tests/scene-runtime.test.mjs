import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  createContextDragBroker,
  createSceneRuntime,
  normalizeContextDragMessage,
  sceneMatchObjectSelector,
} from '../dist/scene/index.js'

test('keeps the first surface command queued until a cold target reports ready', () => {
  let ready = false
  let opened = 0
  const posted = []
  const runtime = createSceneRuntime({ flushDelayMs: -1 })
  runtime.registerSurface('connection_hub.connections', {
    label: 'Connection Hub',
    ensureOpen: () => { opened += 1 },
    isReady: () => ready,
    postCommand: (command) => {
      posted.push(command)
      return true
    },
    commandFromOpen: () => null,
  })

  const command = {
    type: 'kdcube.surface.command',
    target_surface: 'connection_hub.connections',
    command_id: 'first-click',
    ui_event: {
      tab: 'delegated_by_kdcube',
      pending_agent_grant: '1',
    },
  }
  const queued = runtime.queueSurfaceCommand('connection_hub.connections', command)

  assert.equal(queued.ok, true)
  assert.equal(queued.code, 'queued')
  assert.equal(opened, 1)
  assert.equal(posted.length, 0)
  assert.deepEqual(runtime.getPendingCommand('connection_hub.connections'), command)

  ready = true
  assert.equal(runtime.flushSurface('connection_hub.connections'), true)
  assert.deepEqual(posted, [command])
  assert.equal(runtime.getPendingCommand('connection_hub.connections'), undefined)
})

test('normalizes canonical context drag messages', () => {
  const active = normalizeContextDragMessage({
    type: 'kdcube-context-drag-start',
    source_surface_ref: 'chat',
    contexts: [
      {
        ref: 'task:issue:T-1',
        label: 'Task one',
      },
    ],
  })

  assert.equal(active.sourceSurfaceRef, 'chat')
  assert.equal(active.contexts[0].ref, 'task:issue:T-1')
  assert.equal(active.contexts[0].object_ref, 'task:issue:T-1')
})

test('rejects non-canonical drag start messages', () => {
  assert.equal(normalizeContextDragMessage({ type: 'not-canonical-drag-start', contexts: [{ ref: 'task:issue:T-1' }] }), null)
})

test('matches drop targets by selector policy', () => {
  const broker = createContextDragBroker({
    objectAction: async () => ({ ok: true }),
    dispatchOpenResponse: () => ({ ok: true, code: 'dispatched', targetSurface: 'alpha.viewer', message: 'ok' }),
  })
  broker.handleDragStart({
    type: 'kdcube-context-drag-start',
    contexts: [{ ref: 'alpha:item:1/file', namespace: 'alpha:item' }],
  })

  assert.equal(sceneMatchObjectSelector('alpha:item:1/file', 'alpha:*'), true)
  assert.equal(sceneMatchObjectSelector('alpha:item:1/file', 'alpha:other:*'), false)
  assert.equal(broker.accepts({ surfaceRef: 'alpha', targetSurface: 'alpha.viewer', accepts: { open: ['alpha:*'] } }), true)
  assert.equal(broker.accepts({ surfaceRef: 'beta', targetSurface: 'beta.viewer', accepts: { open: ['beta:*'] } }), false)
})

test('opens owning surfaces through provider object action', async () => {
  const calls = []
  const broker = createContextDragBroker({
    objectAction: async (request) => {
      calls.push(request)
      return {
        ok: true,
        object_ref: request.object_ref,
        ui_event: {
          target_surface: request.target_surface,
          object_ref: request.object_ref,
        },
      }
    },
    dispatchOpenResponse: (response, source) => ({
      ok: true,
      code: 'dispatched',
      targetSurface: response.ui_event.target_surface,
      message: `opened ${source.ref}`,
    }),
  })
  broker.handleDragStart({
    type: 'kdcube-context-drag-start',
    contexts: [{ ref: 'alpha:record:1', label: 'Alpha item' }],
  })

  const result = await broker.dropOnTarget({
    surfaceRef: 'alpha',
    targetSurface: 'alpha.viewer',
    accepts: { open: ['alpha:*'] },
    dropEffect: 'open',
  })

  assert.equal(result.ok, true)
  assert.equal(result.code, 'opened')
  assert.equal(calls[0].object_ref, 'alpha:record:1')
  assert.equal(calls[0].target_surface, 'alpha.viewer')
})

test('delivers pin and attach targets locally without provider open', async () => {
  const delivered = []
  const broker = createContextDragBroker({
    objectAction: async () => {
      throw new Error('provider should not be called for pin target')
    },
    dispatchOpenResponse: () => ({ ok: true, code: 'dispatched', targetSurface: 'unused', message: 'unused' }),
  })
  broker.handleDragStart({
    type: 'kdcube-context-drag-start',
    contexts: [{ ref: 'task:issue:T-1' }],
  })

  const result = await broker.dropOnTarget({
    surfaceRef: 'pinboard',
    targetSurface: 'canvas.main',
    accepts: { pin: ['*'] },
    dropEffect: 'pin',
    deliverContext: ({ context, target }) => delivered.push([context.ref, target.surfaceRef]),
  })

  assert.equal(result.ok, true)
  assert.deepEqual(delivered, [['task:issue:T-1', 'pinboard']])
})
