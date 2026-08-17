import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  hydrateHistoricalConversation,
} from '../dist/chat/index.js'

test('historical assistant-only turns keep their stored timestamp order', () => {
  const turns = hydrateHistoricalConversation({
    conversation_id: 'conv-1',
    turns: [
      {
        turn_id: 'turn-recursion',
        artifacts: [
          {
            type: 'chat:assistant',
            ts: '2026-08-17T10:14:45.211504Z',
            data: {
              text: 'Recursion limit of 25 reached without hitting a stop condition.',
            },
          },
        ],
      },
      {
        turn_id: 'turn-after',
        artifacts: [
          {
            type: 'chat:user',
            ts: '2026-08-17T11:53:08.261983Z',
            data: {
              text: 'try again',
              event_type: 'event.user.prompt',
            },
          },
          {
            type: 'chat:assistant',
            ts: '2026-08-17T11:53:08.261983Z',
            data: {
              text: 'Let me pull the test image that was created earlier.',
            },
          },
        ],
      },
    ],
  })

  assert.deepEqual(turns.map((turn) => turn.id), ['turn-recursion', 'turn-after'])
  assert.equal(turns[0].answer, 'Recursion limit of 25 reached without hitting a stop condition.')
  assert.equal(turns[1].userMessage, 'try again')
})
