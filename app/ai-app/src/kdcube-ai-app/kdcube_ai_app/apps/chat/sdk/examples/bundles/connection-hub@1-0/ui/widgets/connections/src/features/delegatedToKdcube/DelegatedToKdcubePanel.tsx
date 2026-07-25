import { useEffect, useMemo, useRef, useState } from 'react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { AccountRow, type AccountStatusTone } from '../../components/AccountRow';
import { PaneGroup } from '../../components/Pane';
import { ConsentPlan, type ConsentPlanAction } from './ConsentPlan';
import type { DelegatedToKdcubeAccount, DelegatedToKdcubeClaim, DelegatedToKdcubeProvider } from '../../api/types';
import {
  connectDelegatedToKdcubeCredential,
  disconnectDelegatedToKdcube,
  loadDelegatedToKdcube,
  startDelegatedToKdcubeOAuth,
  type ConnectCredentialArgs,
} from './delegatedToKdcubeSlice';

// Consent payloads deep-link here: ?tab=delegated_to_kdcube&provider_id=…
// &connector_app_id=…&claims=a,b&account_id=… — preselect what the tool
// asked for and highlight the affected account.
interface ConsentDeepLink {
  providerId: string;
  connectorAppId: string;
  claims: string[];
  accountId: string;
  // Present when the connect step was reached because an AGENT needed a
  // per-account claim: after the provider step, the panel offers a hand-off to
  // grant it to this agent.
  agentClientId: string;
  agentResource: string;
  // The DOOR claims the hand-off agent grant must cover (can differ from the
  // provider connect claims: mail connects gmail:*, the grant names mail:*).
  agentClaims: string[];
}

function consentDeepLink(openParams?: Record<string, string>): ConsentDeepLink {
  // Host-passed openParams win over the URL — an embedded host may not be able
  // to mutate this frame's URL, so the props ARE the deep link on first summon.
  const url = new URLSearchParams(window.location.search);
  const get = (key: string): string => {
    const fromProps = openParams && typeof openParams[key] === 'string' ? openParams[key] : '';
    return (fromProps || url.get(key) || '').trim();
  };
  return {
    providerId: get('provider_id'),
    connectorAppId: get('connector_app_id'),
    claims: get('claims').split(',').map((item) => item.trim()).filter(Boolean),
    accountId: get('account_id'),
    agentClientId: get('agent_client_id'),
    agentResource: get('agent_resource'),
    agentClaims: get('agent_claims').split(',').map((item) => item.trim()).filter(Boolean),
  };
}

function providerLabel(provider: DelegatedToKdcubeProvider): string {
  return provider.label || provider.provider_id;
}

function claimLabel(claim: DelegatedToKdcubeClaim | undefined, claimId: string): string {
  // The REAL consent is the claim token; that is what renders. The friendly
  // label stays descriptive metadata, shown as a hint where a surface adds one.
  void claim;
  return claimId;
}

function accountTitle(account: DelegatedToKdcubeAccount): string {
  return account.display_name || account.email || account.workspace || account.external_subject || account.account_id;
}

function accountSubtitle(account: DelegatedToKdcubeAccount, provider?: DelegatedToKdcubeProvider): string | undefined {
  // Granted claims render as prominent chips on the row; the subtitle keeps
  // only the account's identity facts.
  const bits = [
    provider ? providerLabel(provider) : account.provider_id,
    account.email,
    account.workspace,
  ].filter(Boolean);
  return bits.length ? bits.join(' · ') : undefined;
}

function accountGrantedChips(account: DelegatedToKdcubeAccount, provider?: DelegatedToKdcubeProvider): string[] {
  return (account.claims || []).map((claimId) => claimLabel(provider?.claims?.[claimId], claimId));
}

function formatCredentialDate(value?: number): string {
  if (!value) return '';
  try {
    return new Date(value * 1000).toLocaleString();
  } catch {
    return '';
  }
}

function formatIsoDate(value?: string): string {
  if (!value) return '';
  try {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? '' : parsed.toLocaleString();
  } catch {
    return '';
  }
}

function accountLastError(account: DelegatedToKdcubeAccount): string {
  if (!account.last_error) return '';
  const at = formatIsoDate(account.last_error_at);
  return at ? `Last error (${at}): ${account.last_error}` : `Last error: ${account.last_error}`;
}

function accountStatus(account: DelegatedToKdcubeAccount): { label: string; tone: AccountStatusTone; detail: string } {
  const status = account.credential_status || account.status || '';
  const expires = formatCredentialDate(account.credential_expires_at);
  if (status === 'revoked') {
    return {
      label: 'revoked',
      tone: 'error',
      detail: account.credential_message || 'Access was revoked. Reconnect the account if it is still needed.',
    };
  }
  if (account.reconnect_required || status === 'reconnect_required' || status === 'missing') {
    return {
      label: 'reconnect required',
      tone: 'error',
      detail: account.credential_message || 'Reconnect this account before KDCube can use it.',
    };
  }
  if (status === 'refreshable') {
    return {
      label: 'refreshes automatically',
      tone: 'warn',
      detail: account.credential_message || (expires ? `Access expired ${expires}; KDCube will refresh on next use.` : ''),
    };
  }
  if (status === 'expires_soon') {
    return {
      label: 'expires soon',
      tone: 'warn',
      detail: expires ? `Access expires ${expires}.` : (account.credential_message || ''),
    };
  }
  return {
    label: 'connected',
    tone: 'ok',
    detail: expires ? `Access valid until ${expires}.` : (account.credential_message || ''),
  };
}

function firstProviderId(providers: DelegatedToKdcubeProvider[]): string {
  return providers[0]?.provider_id || '';
}

function firstConnectorAppId(provider?: DelegatedToKdcubeProvider): string {
  return Object.values(provider?.connector_apps || {}).find((app) => app.enabled !== false)?.connector_app_id || '';
}

function defaultClaims(provider?: DelegatedToKdcubeProvider, connectorAppId?: string): string[] {
  const app = connectorAppId ? provider?.connector_apps?.[connectorAppId] : undefined;
  const ceiling = app?.allowed_claims || [];
  return ceiling.length ? ceiling : Object.keys(provider?.claims || {});
}

function oauthEnabled(provider?: DelegatedToKdcubeProvider, connectorAppId?: string): boolean {
  const app = connectorAppId ? provider?.connector_apps?.[connectorAppId] : undefined;
  return Boolean(provider?.adapter?.includes('oauth') && app?.client_id);
}

export function DelegatedToKdcubePanel({ openParams }: { openParams?: Record<string, string> } = {}) {
  const dispatch = useAppDispatch();
  const { enabled, providers, accounts, busy } = useAppSelector((s) => s.delegatedToKdcube);
  const providerList = useMemo(
    () => Object.values(providers).filter((provider) => provider.enabled !== false).sort((a, b) => providerLabel(a).localeCompare(providerLabel(b))),
    [providers],
  );
  const [deepLink] = useState<ConsentDeepLink>(() => consentDeepLink(openParams));
  const [providerId, setProviderId] = useState(deepLink.providerId);
  const selectedProviderId = providerId || firstProviderId(providerList);
  const selectedProvider = providers[selectedProviderId];
  const [connectorAppId, setConnectorAppId] = useState(deepLink.providerId ? deepLink.connectorAppId : '');
  const selectedConnectorAppId = connectorAppId || firstConnectorAppId(selectedProvider);
  const claimIds = Object.keys(selectedProvider?.claims || {});
  const suggestedClaims = defaultClaims(selectedProvider, selectedConnectorAppId);
  const [claims, setClaims] = useState<string[]>(deepLink.providerId ? deepLink.claims : []);
  // Deep-links and prefills may carry claims of another provider; only
  // claims this provider declares may reach the OAuth start. If nothing
  // survives the filter, fall back to the provider's suggested claims.
  const rawSelectedClaims = claims.length ? claims : suggestedClaims;
  const providerScopedClaims = claimIds.length
    ? rawSelectedClaims.filter((claimId) => claimIds.includes(claimId))
    : rawSelectedClaims;
  const selectedClaims = providerScopedClaims.length ? providerScopedClaims : suggestedClaims;
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [workspace, setWorkspace] = useState('');
  const [secretKind, setSecretKind] = useState<ConnectCredentialArgs['secretKind']>('app_password');
  const [secretValue, setSecretValue] = useState('');
  const [formNotice, setFormNotice] = useState('');
  // Account whose access the form is managing; its checked set replaces the
  // account's claims on approval. Empty = connecting a new account.
  const [managedAccountId, setManagedAccountId] = useState('');
  const formRef = useRef<HTMLFormElement | null>(null);
  const canStartOAuth = oauthEnabled(selectedProvider, selectedConnectorAppId) && selectedClaims.length > 0;

  const launchOAuth = async (
    targetProviderId: string,
    targetConnectorAppId: string,
    targetClaims: string[],
    opts?: {accountId?: string; claimsMode?: 'add' | 'replace'},
  ) => {
    console.info('[connect-route] launchOAuth', {
      provider: targetProviderId, connectorApp: targetConnectorAppId,
      claims: targetClaims, accountId: opts?.accountId || '', claimsMode: opts?.claimsMode || '',
    });
    const result = await dispatch(startDelegatedToKdcubeOAuth({
      providerId: targetProviderId,
      connectorAppId: targetConnectorAppId,
      claims: targetClaims,
      returnHint: window.location.href,
      accountId: opts?.accountId,
      claimsMode: opts?.claimsMode,
    })).unwrap().catch((err) => {
      console.warn('[connect-route] startOAuth dispatch FAILED', err);
      return undefined;
    });
    if (result?.authorize_url) {
      console.info('[connect-route] launchOAuth -> opening provider tab', result.authorize_url);
      // Arms the one-shot focus refresh in App: when the user returns from
      // the provider tab, the widget re-fetches once (no standing polling).
      try {
        sessionStorage.setItem('kdc-oauth-pending', '1');
      } catch {
        // Storage unavailable: the BroadcastChannel push still covers it.
      }
      window.open(result.authorize_url, '_blank', 'noopener,noreferrer');
    } else {
      console.warn('[connect-route] launchOAuth got NO authorize_url — nothing opened', {
        provider: targetProviderId, connectorApp: targetConnectorAppId, result,
      });
    }
  };

  // Re-run provider approval for an existing account: same connector app,
  // same claims. OAuth accounts go straight to the provider; credential
  // accounts get the form prefilled so the user pastes a fresh secret.
  const reconnect = (account: DelegatedToKdcubeAccount) => {
    const provider = providers[account.provider_id];
    const appId = account.connector_app_id || firstConnectorAppId(provider);
    const accountClaims = account.claims?.length ? account.claims : defaultClaims(provider, appId);
    if (oauthEnabled(provider, appId)) {
      void launchOAuth(account.provider_id, appId, accountClaims, {
        accountId: account.account_id,
        claimsMode: 'replace',
      });
      return;
    }
    prefillForAccount(account, accountClaims, 'Enter a fresh credential to reconnect this account.');
  };

  // Manage access: prefill the form with the account's current claims
  // checked. The checked set is authoritative on re-approval — ticking adds
  // access, unticking removes it.
  const manageAccess = (account: DelegatedToKdcubeAccount) => {
    const provider = providers[account.provider_id];
    const appId = account.connector_app_id || firstConnectorAppId(provider);
    const merged = Array.from(new Set([
      ...(account.claims || []),
      ...(deepLink.accountId === account.account_id ? deepLink.claims : []),
    ]));
    prefillForAccount(
      account,
      merged.length ? merged : defaultClaims(provider, appId),
      'This account keeps exactly the checked access: tick to add, untick to remove, then reconnect.',
    );
  };

  const prefillForAccount = (account: DelegatedToKdcubeAccount, targetClaims: string[], notice: string) => {
    setProviderId(account.provider_id);
    setConnectorAppId(account.connector_app_id || '');
    setManagedAccountId(account.account_id);
    setClaims(targetClaims);
    setEmail(account.email || '');
    setDisplayName(account.display_name || '');
    setWorkspace(account.workspace || '');
    setFormNotice(notice);
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // ── consent-plan (deep-link) wiring ────────────────────────────────────
  const [planDismissed, setPlanDismissed] = useState(false);
  const planProvider = deepLink.providerId ? providers[deepLink.providerId] : undefined;
  useEffect(() => {
    console.info('[connect-route] delegated-to-kdcube panel mount', {
      deepLink,
      planProviderResolved: Boolean(planProvider),
      knownProviders: Object.keys(providers || {}),
      location: window.location.search,
    });
    // Mount-time snapshot only — the summon remounts the panel by key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const planRequestedClaims = useMemo(() => {
    if (!planProvider) return [] as string[];
    const declared = Object.keys(planProvider.claims || {});
    const requested = deepLink.claims.filter((claimId) => declared.includes(claimId));
    return requested.length ? requested : defaultClaims(planProvider, deepLink.connectorAppId || firstConnectorAppId(planProvider));
  }, [planProvider, deepLink]);
  const planAccount = useMemo(() => {
    if (!planProvider) return undefined;
    const candidates = accounts.filter((account) =>
      account.provider_id === deepLink.providerId
      && (!deepLink.connectorAppId || !account.connector_app_id || account.connector_app_id === deepLink.connectorAppId));
    if (deepLink.accountId) {
      const exact = candidates.find((account) => account.account_id === deepLink.accountId);
      if (exact) return exact;
    }
    // Prefer the account that already approved most of what was requested.
    return [...candidates].sort((a, b) => {
      const approvedIn = (account: DelegatedToKdcubeAccount) =>
        planRequestedClaims.filter((claimId) => (account.claims || []).includes(claimId)).length;
      return approvedIn(b) - approvedIn(a);
    })[0];
  }, [planProvider, accounts, deepLink, planRequestedClaims]);

  // The plan hands over the claims to submit: the account's held claims plus
  // whatever the user kept ticked in step 3 (they may untick any requested
  // claim before connecting).
  const runPlanAction = (action: ConsentPlanAction, planClaims: string[]) => {
    if (!planProvider) {
      console.warn('[connect-route] runPlanAction with NO planProvider — button did nothing', { action });
      return;
    }
    const appId = deepLink.connectorAppId || firstConnectorAppId(planProvider);
    const targetClaims = planClaims.length ? planClaims : planRequestedClaims;
    console.info('[connect-route] runPlanAction', {
      action, provider: planProvider.provider_id, appId,
      oauthEnabled: oauthEnabled(planProvider, appId), targetClaims,
      hasPlanAccount: Boolean(planAccount),
    });
    if (action === 'connect') {
      if (oauthEnabled(planProvider, appId)) {
        void launchOAuth(planProvider.provider_id, appId, targetClaims);
        return;
      }
      console.info('[connect-route] runPlanAction connect: OAuth NOT enabled -> prefill credential form (no direct connect)');
      setProviderId(planProvider.provider_id);
      setConnectorAppId(appId);
      setClaims(targetClaims);
      setFormNotice(`Enter the ${providerLabel(planProvider)} account details and its credential below.`);
      formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (!planAccount) return;
    const accountAppId = planAccount.connector_app_id || appId;
    if (oauthEnabled(planProvider, accountAppId)) {
      // A consent-plan approval ADDS to the account's held access; the
      // server unions the claims so nothing already approved is dropped.
      void launchOAuth(planProvider.provider_id, accountAppId, targetClaims, {
        accountId: planAccount.account_id,
        claimsMode: action === 'reconnect' ? 'replace' : 'add',
      });
      return;
    }
    prefillForAccount(
      planAccount,
      targetClaims,
      action === 'reconnect'
        ? 'Enter a fresh credential to reconnect this account.'
        : 'Check the additional access this account should approve, then reconnect it.',
    );
  };

  const toggleClaim = (claimId: string) => {
    setClaims((current) => {
      const base = current.length ? current : suggestedClaims;
      return base.includes(claimId)
        ? base.filter((item) => item !== claimId)
        : [...base, claimId];
    });
  };

  const changeProvider = (nextProviderId: string) => {
    setProviderId(nextProviderId);
    setConnectorAppId('');
    setClaims([]);
    setFormNotice('');
    setManagedAccountId('');
  };

  const changeConnectorApp = (nextConnectorAppId: string) => {
    setConnectorAppId(nextConnectorAppId);
    setClaims([]);
    setFormNotice('');
    setManagedAccountId('');
  };

  const submit = async () => {
    if (!selectedProviderId || !selectedConnectorAppId || !secretValue || selectedClaims.length === 0) return;
    await dispatch(connectDelegatedToKdcubeCredential({
      providerId: selectedProviderId,
      connectorAppId: selectedConnectorAppId,
      // The account email is the user-facing identity of a pasted-in
      // credential; a separate provider subject is an OAuth-callback concern.
      externalSubject: email,
      email,
      displayName,
      workspace,
      claims: selectedClaims,
      secretKind,
      secretValue,
    })).unwrap().catch(() => undefined);
    setEmail('');
    setDisplayName('');
    setWorkspace('');
    setSecretValue('');
    void dispatch(loadDelegatedToKdcube());
  };

  const startOAuth = async () => {
    console.info('[connect-route] startOAuth (connect-pane button)', {
      selectedProviderId, selectedConnectorAppId, canStartOAuth,
      selectedClaims, managedAccountId,
    });
    if (!selectedProviderId || !selectedConnectorAppId || !canStartOAuth) {
      console.warn('[connect-route] startOAuth guard blocked — nothing launched', {
        selectedProviderId, selectedConnectorAppId, canStartOAuth,
      });
      return;
    }
    await launchOAuth(
      selectedProviderId,
      selectedConnectorAppId,
      selectedClaims,
      managedAccountId ? {accountId: managedAccountId, claimsMode: 'replace'} : undefined,
    );
  };

  const disconnect = (accountId: string) => {
    void dispatch(disconnectDelegatedToKdcube({ accountId })).then(() => dispatch(loadDelegatedToKdcube()));
  };

  if (!enabled) {
    return (
      <section className="card">
        <div className="card-head">
          <h2>Delegated to KDCube</h2>
        </div>
        <p className="muted">No external account delegation providers are enabled in this environment.</p>
      </section>
    );
  }

  const existingPane = (
    <section className="card">
      <div className="card-head">
        <p className="muted" style={{ margin: 0 }}>
          External accounts this user allows KDCube applications or automation
          to use. Connecting an account is only step one: each agent or app
          still needs its own grant, per account, for what it may do with it
          (approved under Delegated by KDCube).
        </p>
        <span className="badge badge-ok">{providerList.length} providers</span>
      </div>

      {planProvider && !planDismissed ? (
        <ConsentPlan
          request={{
            provider: planProvider,
            providerLabel: providerLabel(planProvider),
            requestedClaims: planRequestedClaims,
            account: planAccount,
          }}
          claimLabel={(claimId) => claimLabel(planProvider.claims?.[claimId], claimId)}
          busy={busy}
          onAction={runPlanAction}
          onDismiss={() => setPlanDismissed(true)}
          agentHandoff={deepLink.agentClientId ? {
            clientId: deepLink.agentClientId,
            resource: deepLink.agentResource,
            accountId: planAccount?.account_id || deepLink.accountId,
            claim: planRequestedClaims[0] || '',
            claims: deepLink.agentClaims,
          } : undefined}
        />
      ) : null}

      <div className="integration-provider-list">
        {providerList.map((provider) => {
          const providerAccounts = accounts.filter((account) => account.provider_id === provider.provider_id);
          const caps = Object.entries(provider.claims || {});
          return (
            <div className="integration-provider" key={provider.provider_id}>
              <div className="integration-provider-head">
                <div>
                  <div className="account-title">{providerLabel(provider)}</div>
                  {provider.adapter ? <div className="account-sub">{provider.adapter}</div> : null}
                </div>
                {caps.length ? (
                  <div className="account-sub claim-catalog">
                    Can delegate: {caps.map(([claimId, claim]) => claimLabel(claim, claimId)).join(' · ')}
                  </div>
                ) : null}
              </div>
              {providerAccounts.length ? (
                <ul className="accounts">
                  {providerAccounts.map((account) => {
                    const status = accountStatus(account);
                    const needsReconnect = status.tone === 'error';
                    return (
                      <AccountRow
                        key={account.account_id}
                        title={accountTitle(account)}
                        subtitle={accountSubtitle(account, provider)}
                        grantedChips={accountGrantedChips(account, provider)}
                        statusLabel={status.label}
                        statusTone={status.tone}
                        detail={status.detail}
                        lastError={needsReconnect ? accountLastError(account) : ''}
                        highlighted={deepLink.accountId === account.account_id}
                        busy={busy}
                        actions={(
                          <>
                            <button
                              className={needsReconnect ? 'btn' : 'btn btn-ghost'}
                              type="button"
                              disabled={busy}
                              onClick={() => reconnect(account)}
                            >
                              Reconnect
                            </button>
                            <button
                              className="btn btn-ghost"
                              type="button"
                              disabled={busy}
                              onClick={() => manageAccess(account)}
                            >
                              Manage access
                            </button>
                          </>
                        )}
                        onDisconnect={() => disconnect(account.account_id)}
                      />
                    );
                  })}
                </ul>
              ) : (
                <p className="muted">No accounts delegated to KDCube.</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );

  const managedAccount = managedAccountId
    ? accounts.find((account) => account.account_id === managedAccountId)
    : undefined;

  const connectPane = (
    <section className="card">
      <form
        ref={formRef}
        className="form form-flush"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy) void submit();
        }}
      >
        {managedAccount ? (
          <p className="muted" style={{ margin: 0 }}>
            Managing <strong>{accountTitle(managedAccount)}</strong> — the checked
            access is what this account keeps after re-approval.{' '}
            <button
              type="button"
              className="btn btn-ghost"
              disabled={busy}
              onClick={() => {
                setManagedAccountId('');
                setClaims([]);
                setFormNotice('');
              }}
            >
              Connect a different account instead
            </button>
          </p>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            Pick the provider, choose what KDCube may do with the account, then
            approve. The result appears in Connected accounts.
          </p>
        )}
        {formNotice ? <p className="notice success">{formNotice}</p> : null}
        <div className="inline-fields">
          <select className="input" value={selectedProviderId} onChange={(event) => changeProvider(event.target.value)}>
            {providerList.map((provider) => (
              <option key={provider.provider_id} value={provider.provider_id}>
                {providerLabel(provider)}
              </option>
            ))}
          </select>
          <select className="input" value={selectedConnectorAppId} onChange={(event) => changeConnectorApp(event.target.value)}>
            {Object.values(selectedProvider?.connector_apps || {}).map((app) => (
              <option key={app.connector_app_id} value={app.connector_app_id}>
                {app.label || app.connector_app_id}
              </option>
            ))}
          </select>
        </div>

        {claimIds.length ? (
          <div className="scope-list">
            {claimIds.map((claimId) => (
              <label className="scope-item" key={claimId}>
                <input
                  type="checkbox"
                  checked={selectedClaims.includes(claimId)}
                  onChange={() => toggleClaim(claimId)}
                  disabled={busy}
                />
                <span>{claimLabel(selectedProvider?.claims?.[claimId], claimId)}</span>
              </label>
            ))}
          </div>
        ) : null}

        {canStartOAuth ? (
          <div className="oauth-connect">
            <button className="btn" type="button" disabled={busy} onClick={() => void startOAuth()}>
              {managedAccount
                ? `Re-approve with ${providerLabel(selectedProvider)}`
                : `Connect with ${providerLabel(selectedProvider)}`}
            </button>
            <span className="small">
              Opens {providerLabel(selectedProvider)}'s approval page in a new tab; the account
              details come back from the provider.
            </span>
          </div>
        ) : (
          <>
            <p className="muted">
              This provider uses a credential you paste in (for example an app
              password created in the provider's security settings).
            </p>
            <div className="inline-fields">
              <input className="input" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="account email" />
              <input className="input" value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="label shown in KDCube (optional)" />
            </div>
            <div className="inline-fields">
              <input className="input" value={workspace} onChange={(event) => setWorkspace(event.target.value)} placeholder="mailbox / workspace (optional)" />
            </div>
            <div className="inline-fields">
              <select className="input" value={secretKind} onChange={(event) => setSecretKind(event.target.value as ConnectCredentialArgs['secretKind'])}>
                <option value="app_password">app password</option>
                <option value="access_token">access token</option>
                <option value="api_key">API key</option>
                <option value="secret">secret</option>
              </select>
              <input
                className="input"
                type="password"
                value={secretValue}
                onChange={(event) => setSecretValue(event.target.value)}
                placeholder="credential"
                autoComplete="new-password"
              />
            </div>
            <button className="btn" type="submit" disabled={busy || !selectedProviderId || !selectedConnectorAppId || !secretValue || selectedClaims.length === 0}>
              {managedAccount ? 'Reconnect' : 'Connect'}
            </button>
          </>
        )}
      </form>
    </section>
  );

  return (
    <PaneGroup
      panes={[
        { id: 'accounts', title: 'Connected accounts', content: existingPane },
        { id: 'connect', title: managedAccount ? 'Manage account access' : 'Connect a new account', content: connectPane },
      ]}
    />
  );
}
