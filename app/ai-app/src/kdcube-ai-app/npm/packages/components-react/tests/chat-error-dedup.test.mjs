import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

const CHAT_TURN_SOURCE = readFileSync(
  new URL('../src/chat/ui/features/chat/ChatTurnView.tsx', import.meta.url),
  'utf8',
)

test('Chat view renders one detailed notice for a chat.error event', () => {
  assert.match(
    CHAT_TURN_SOURCE,
    /event\.kind === 'artifact' && event\.artifact\.kind === 'service_error'/,
  )
  assert.match(
    CHAT_TURN_SOURCE,
    /turn\.state === 'error' && !hasInlineServiceError/,
  )
})
