import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  applyChatStep,
  chatActions,
  chatReducer,
  createEmptyTurn,
  initialState,
} from '../dist/chat/index.js'

// Surfaced live: two stacked consent banners (a stale connect_required from
// before the account existed + the current claim_upgrade_required), re-raised
// every turn, with a close button that looked dead because the next envelope
// re-added the banner. One banner per provider, stable while the signature
// repeats, dismissal remembered per conversation.

function consentEnvelope(turnId, { provider = 'slack', claims, message, tools = [] } = {}) {
  return {
    type: 'chat.step',
    timestamp: '2026-07-07T12:00:00.000Z',
    service: { request_id: `req:${turnId}` },
    conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: turnId },
    event: { step: 'tool.preflight', status: 'completed', title: 'Preflight', agent: null },
    data: {
      error: { code: 'needs_connected_account_consent', message },
      consent: {
        kind: 'delegated_to_kdcube.connected_account',
        provider_id: provider,
        claims,
        url: `/widgets/connections_settings?tab=delegated_to_kdcube&provider_id=${provider}&claims=${claims.join(',')}`,
        action_label: 'Approve access',
        tools,
      },
    },
  }
}

function baseState() {
  return {
    ...initialState,
    conversationId: 'conv-1',
    turns: [createEmptyTurn('turn-1', 1_000, 'hello')],
  }
}

const SEVEN = ['slack:search', 'slack:channels', 'slack:history', 'slack:files:read', 'slack:files:write', 'slack:assistant:search', 'slack:post']

test('the banner carries the claims as chips and lifts the inline enumeration from the text', () => {
  const shown = applyChatStep(baseState(), consentEnvelope('turn-1', {
    claims: ['slack:files:write', 'slack:post'],
    message: 'Your Slack account is connected but has not approved the required access (needs: slack:files:write, slack:post). Approve it in Connection Hub, then retry.',
  }))
  assert.equal(shown.banners.length, 1)
  const banner = shown.banners[0]
  assert.deepEqual(banner.consentClaims, ['slack:files:write', 'slack:post'])
  assert.doesNotMatch(banner.text, /needs:/)
  assert.match(banner.text, /has not approved the required access\. Approve it in Connection Hub, then retry\./)
})

test('a repeated consent state keeps ONE banner with a stable id', () => {
  const first = applyChatStep(baseState(), consentEnvelope('turn-1', { claims: SEVEN, message: 'Connect your Slack account.' }))
  assert.equal(first.banners.length, 1)
  const id = first.banners[0].id
  const second = applyChatStep(first, consentEnvelope('turn-1', { claims: SEVEN, message: 'Connect your Slack account.' }))
  assert.equal(second.banners.length, 1)
  assert.equal(second.banners[0].id, id)
})

test('a NEW consent state for the same provider supersedes the older banner', () => {
  const stale = applyChatStep(baseState(), consentEnvelope('turn-1', { claims: SEVEN, message: 'Connect your Slack account.' }))
  const upgraded = applyChatStep(
    stale,
    consentEnvelope('turn-2', {
      claims: ['slack:files:write', 'slack:post'],
      message: 'Your Slack account is connected but has not approved the required access.',
      tools: ['slack.upload_slack_file', 'slack.post_slack_message'],
    }),
  )
  assert.equal(upgraded.banners.length, 1)
  assert.match(upgraded.banners[0].text, /has not approved/)
  assert.deepEqual(upgraded.banners[0].consentTools, ['slack.upload_slack_file', 'slack.post_slack_message'])
})

test('dismiss removes the banner AND keeps the identical state quiet; a changed claims set shows again', () => {
  const shown = applyChatStep(baseState(), consentEnvelope('turn-1', { claims: ['slack:post'], message: 'Approve slack:post.' }))
  assert.equal(shown.banners.length, 1)
  const dismissed = chatReducer(shown, chatActions.dismissBanner(shown.banners[0].id))
  assert.equal(dismissed.banners.length, 0)

  const reRaised = applyChatStep(dismissed, consentEnvelope('turn-2', { claims: ['slack:post'], message: 'Approve slack:post.' }))
  assert.equal(reRaised.banners.length, 0)

  const changed = applyChatStep(dismissed, consentEnvelope('turn-3', { claims: ['slack:post', 'slack:files:write'], message: 'Approve slack:post and slack:files:write.' }))
  assert.equal(changed.banners.length, 1)
})

test('a new conversation clears the dismissal memory', () => {
  const shown = applyChatStep(baseState(), consentEnvelope('turn-1', { claims: ['slack:post'], message: 'Approve slack:post.' }))
  const dismissed = chatReducer(shown, chatActions.dismissBanner(shown.banners[0].id))
  assert.equal(dismissed.dismissedConsentSignatures.length, 1)
  const fresh = chatReducer(dismissed, chatActions.startNewConversation())
  assert.equal(fresh.dismissedConsentSignatures.length, 0)
})

test('spotlightTools sets and clearToolSpotlight clears the menu request', () => {
  const lit = chatReducer({ ...initialState }, chatActions.spotlightTools(['slack.post_slack_message', ' ', 'slack.upload_slack_file']))
  assert.deepEqual(lit.toolSpotlight.tools, ['slack.post_slack_message', 'slack.upload_slack_file'])
  assert.ok(lit.toolSpotlight.nonce > 0)
  const cleared = chatReducer(lit, chatActions.clearToolSpotlight())
  assert.equal(cleared.toolSpotlight, null)
})

test('a per-agent grant demand routes the banner to the Delegated-by-KDCube tab with a one-click grant', () => {
  const env = {
    type: 'chat.step',
    timestamp: '2026-07-07T12:00:00.000Z',
    service: { request_id: 'req:turn-1' },
    conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
    event: { step: 'delegated_to_kdcube.consent', status: 'completed', title: 'Consent', agent: null },
    data: {
      error: { code: 'needs_connected_account_consent', message: 'memory_search needs your consent to memories:read.' },
      consent: {
        kind: 'delegated_agent_grant',
        claims: ['memories:read'],
        resource: 'https://h/api/mcp/mem',
        url: '/widgets/connections?tab=delegated_to_kdcube',
        action_label: 'Grant access',
        agent_client_id: 'kdcube-agent:app:lg-react',
        grant: {
          operation: 'delegated_agent_grant_create',
          payload: { client_id: 'kdcube-agent:app:lg-react', resource: 'https://h/api/mcp/mem', claims: ['memories:read'] },
        },
      },
    },
  }
  const shown = applyChatStep(baseState(), env)
  assert.equal(shown.banners.length, 1)
  const banner = shown.banners[0]
  assert.equal(banner.actionLabel, 'Grant access')
  // The AUTOMATION tab (Delegated by KDCube) — an agent grant is a kind of
  // automation access; the connected-accounts tab is the wrong destination.
  assert.equal(banner.consent.tab, 'delegated_by_kdcube')
  assert.equal(banner.consent.params.agent_client_id, 'kdcube-agent:app:lg-react')
  assert.equal(banner.consent.params.pending_agent_grant, '1')
  assert.equal(banner.consent.params.resource, 'https://h/api/mcp/mem')
  assert.deepEqual(banner.consentClaims, ['memories:read'])
})

test('a per-account agent grant deep link preserves account_id/account_claim from the served URL', () => {
  const env = {
    type: 'chat.step',
    timestamp: '2026-07-29T12:00:00.000Z',
    service: { request_id: 'req:turn-1' },
    conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
    event: { step: 'delegated_to_kdcube.consent', status: 'completed', title: 'Consent', agent: 'connection-hub' },
    data: {
      error: { code: 'needs_connected_account_consent' },
      consent: {
        kind: 'delegated_to_kdcube.connected_account',
        reason: 'agent_grant_required',
        provider_id: 'google',
        claims: ['gmail:read'],
        account_id: 'google_cc4e6c3955b5f55d',
        candidates: [{
          account_id: 'google_cc4e6c3955b5f55d',
          label: 'KDCube Demo Reader',
          email: 'kdcube.demo.reader@gmail.com',
          claims: ['gmail:read'],
        }],
        url: '/widgets/connections_settings'
          + '?tab=delegated_by_kdcube&pending_agent_grant=1'
          + '&agent_client_id=kdcube-agent%3Aworkspace%402026-03-31-13-36%3Amain'
          + '&resource=%2A%2Fapi%2Fintegrations%2Fbundles%2F%2A%2F%2A%2Fkdcube-services%401-0%2Fpublic%2Fmcp%2Fnamed_services%2A'
          + '&claims=&account_id=google_cc4e6c3955b5f55d&account_claim=gmail%3Aread',
        action_label: 'Grant this agent access',
        tools: ['mail'],
      },
    },
  }
  const shown = applyChatStep(baseState(), env)
  assert.equal(shown.banners.length, 1)
  const banner = shown.banners[0]
  assert.equal(banner.text, 'Grant this agent gmail:read on KDCube Demo Reader.')
  assert.equal(banner.actionLabel, 'Grant this agent access')
  assert.equal(banner.consent.tab, 'delegated_by_kdcube')
  assert.equal(banner.consent.params.pending_agent_grant, '1')
  assert.equal(banner.consent.params.agent_client_id, 'kdcube-agent:workspace@2026-03-31-13-36:main')
  assert.equal(banner.consent.params.account_id, 'google_cc4e6c3955b5f55d')
  assert.equal(banner.consent.params.account_claim, 'gmail:read')
  assert.equal(banner.consent.params.claims, undefined)
  assert.deepEqual(banner.consentClaims, ['gmail:read'])
  assert.deepEqual(banner.consentTools, ['mail'])
})

test('nested tool-result details still raise the precise per-account agent grant banner', () => {
  const env = {
    type: 'chat.step',
    timestamp: '2026-07-29T12:00:00.000Z',
    service: { request_id: 'req:turn-1' },
    conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
    event: { step: 'react.tool', status: 'error', title: 'Tool result', agent: 'react.tool' },
    data: {
      result: [{
        artifact_id: 'named_services.search_objects',
        error: {
          code: 'needs_connected_account_consent',
          message: 'Grant this agent gmail:read on account KDCube Demo Reader.',
          where: 'named_services.search_objects',
          details: {
            reason: 'agent_grant_required',
            provider_id: 'google',
            claims: ['gmail:read'],
            account_id: 'google_cc4e6c3955b5f55d',
            connection_hub_url: '/widgets/connections_settings'
              + '?tab=delegated_by_kdcube&pending_agent_grant=1'
              + '&agent_client_id=kdcube-agent%3Aworkspace%402026-03-31-13-36%3Amain'
              + '&resource=%2A%2Fapi%2Fintegrations%2Fbundles%2F%2A%2F%2A%2Fkdcube-services%401-0%2Fpublic%2Fmcp%2Fnamed_services%2A'
              + '&claims=&account_id=google_cc4e6c3955b5f55d&account_claim=gmail%3Aread',
            action_label: 'Grant this agent access',
            consent: {
              kind: 'delegated_to_kdcube.connected_account',
              reason: 'agent_grant_required',
              provider_id: 'google',
              claims: ['gmail:read'],
              account_id: 'google_cc4e6c3955b5f55d',
              candidates: [{
                account_id: 'google_cc4e6c3955b5f55d',
                label: 'KDCube Demo Reader',
                email: 'kdcube.demo.reader@gmail.com',
                claims: ['gmail:read'],
              }],
            },
          },
        },
      }],
    },
  }
  const shown = applyChatStep(baseState(), env)
  assert.equal(shown.banners.length, 1)
  const banner = shown.banners[0]
  assert.equal(banner.text, 'Grant this agent gmail:read on account KDCube Demo Reader.')
  assert.equal(banner.actionLabel, 'Grant this agent access')
  assert.equal(banner.consent.tab, 'delegated_by_kdcube')
  assert.equal(banner.consent.params.account_id, 'google_cc4e6c3955b5f55d')
  assert.equal(banner.consent.params.account_claim, 'gmail:read')
  assert.equal(banner.consent.params.claims, undefined)
  assert.deepEqual(banner.consentClaims, ['gmail:read'])
})

test('a more specific per-account agent grant demand supersedes the previous account demand', () => {
  function accountDemand(accountId) {
    return {
      type: 'chat.step',
      timestamp: '2026-07-29T12:00:00.000Z',
      service: { request_id: `req:${accountId}` },
      conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
      event: { step: 'delegated_to_kdcube.consent', status: 'completed', title: 'Consent', agent: 'connection-hub' },
      data: {
        error: { code: 'needs_connected_account_consent' },
        consent: {
          kind: 'delegated_to_kdcube.connected_account',
          reason: 'agent_grant_required',
          provider_id: 'google',
          claims: ['gmail:read'],
          account_id: accountId,
          url: '/widgets/connections_settings'
            + '?tab=delegated_by_kdcube&pending_agent_grant=1'
            + '&agent_client_id=kdcube-agent%3Aworkspace%402026-03-31-13-36%3Amain'
            + '&resource=%2A%2Fapi%2Fintegrations%2Fbundles%2F%2A%2F%2A%2Fkdcube-services%401-0%2Fpublic%2Fmcp%2Fnamed_services%2A'
            + `&account_id=${accountId}&account_claim=gmail%3Aread`,
          action_label: 'Grant this agent access',
        },
      },
    }
  }
  let state = applyChatStep(baseState(), accountDemand('google_610de3d5454c4c5e'))
  state = applyChatStep(state, accountDemand('google_cc4e6c3955b5f55d'))
  assert.equal(state.banners.length, 1)
  assert.equal(state.banners[0].consent.params.account_id, 'google_cc4e6c3955b5f55d')
})

test('one agent, two pending resources -> TWO coexisting banners (surfaced live: slack swallowed memories)', () => {
  function agentDemand(resource, claims, toolName) {
    return {
      type: 'chat.step',
      timestamp: '2026-07-07T12:00:00.000Z',
      service: { request_id: 'req:turn-1' },
      conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
      event: { step: 'delegated_to_kdcube.consent', status: 'completed', title: 'Consent', agent: null },
      data: {
        error: { code: 'needs_connected_account_consent', message: `${toolName} needs your consent to ${claims.join(', ')}.` },
        consent: {
          kind: 'delegated_agent_grant',
          claims,
          resource,
          url: '',
          action_label: 'Grant access',
          agent_client_id: 'kdcube-agent:app:lg-react',
          grant: {
            operation: 'delegated_agent_grant_create',
            payload: { client_id: 'kdcube-agent:app:lg-react', resource, claims },
          },
        },
      },
    }
  }
  let state = applyChatStep(baseState(), agentDemand('*/user-memories@2026-06-26/public/mcp/memories*', ['memories:read'], 'memories'))
  state = applyChatStep(state, agentDemand('*/kdcube-services@1-0/public/mcp/named_services*', ['named_services:use', 'slack:read'], 'slack'))
  // Both demands stay visible — the second must NOT supersede the first.
  assert.equal(state.banners.length, 2)
  const resources = state.banners.map((b) => b.consent.params.resource).sort()
  assert.ok(resources[0].includes('kdcube-services') && resources[1].includes('user-memories'))
  // Re-emitting the SAME demand (the per-turn re-announce) keeps ONE banner.
  const again = applyChatStep(state, agentDemand('*/user-memories@2026-06-26/public/mcp/memories*', ['memories:read'], 'memories'))
  assert.equal(again.banners.length, 2)
})

// A named-service demand names ONE inner operation. The panel seeds its picker
// from namespace/operation, so a banner that drops them opens the grant with
// nothing selected and the user has to find the operation themselves.
test('a named-service grant demand carries its namespace and operation to the panel', () => {
  const env = {
    type: 'chat.step',
    timestamp: '2026-07-07T12:00:00.000Z',
    service: { request_id: 'req:turn-1' },
    conversation: { session_id: 'session-1', conversation_id: 'conv-1', turn_id: 'turn-1' },
    event: { step: 'delegated_to_kdcube.consent', status: 'completed', title: 'Consent', agent: null },
    data: {
      error: { code: 'delegated_consent_required', message: 'named_services_search needs consent.' },
      consent: {
        kind: 'delegated_agent_grant',
        claims: ['named_services:use'],
        resource: 'https://h/api/mcp/named_services',
        agent_client_id: 'kdcube-agent:app:lg-react',
        namespace: 'task',
        operation: 'object.search',
        grant: {
          operation: 'delegated_agent_grant_create',
          payload: {
            client_id: 'kdcube-agent:app:lg-react',
            resource: 'https://h/api/mcp/named_services',
            claims: ['named_services:use'],
            named_service_operations: { task: ['object.search'] },
          },
        },
      },
    },
  }
  const banner = applyChatStep(baseState(), env).banners[0]
  assert.equal(banner.consent.params.namespace, 'task')
  assert.equal(banner.consent.params.operation, 'object.search')
})
