import assert from 'node:assert/strict'
import { test } from 'node:test'
import { objectActionExternalUrl as canvasObjectActionExternalUrl } from '../dist/canvas/index.js'
import { objectActionExternalUrl } from '../dist/chat/index.js'

test('uses only the resolver-declared external URL for browser opens', () => {
  const response = {
    object: { web_url: 'https://untrusted.example/from-object' },
    ui_event: { external_url: 'https://docs.google.com/document/d/doc-1/edit' },
  }

  assert.equal(
    objectActionExternalUrl(response),
    'https://docs.google.com/document/d/doc-1/edit',
  )
  assert.equal(
    objectActionExternalUrl({ object: response.object }),
    '',
  )
  assert.equal(
    canvasObjectActionExternalUrl(response),
    'https://docs.google.com/document/d/doc-1/edit',
  )
})

test('resolves relative provider URLs against the runtime origin', () => {
  assert.equal(
    objectActionExternalUrl(
      { external_url: '/api/files/report' },
      'https://demo.kdcube.tech/chat',
    ),
    'https://demo.kdcube.tech/api/files/report',
  )
})

test('rejects non-web schemes from object-open responses', () => {
  assert.equal(objectActionExternalUrl({ external_url: 'javascript:alert(1)' }), '')
  assert.equal(objectActionExternalUrl({ external_url: 'data:text/html,test' }), '')
})
