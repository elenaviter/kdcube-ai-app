import { createAsyncThunk, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { getOp, postOp } from '../../api/client';
import type {
  DelegatedToKdcubeAccount,
  DelegatedToKdcubeProvider,
  DelegatedToKdcubeCatalogResult,
  DelegatedToKdcubeMutationResult,
  DelegatedToKdcubeOAuthStartResult,
} from '../../api/types';

export interface DelegatedToKdcubeState {
  enabled: boolean;
  providers: Record<string, DelegatedToKdcubeProvider>;
  accounts: DelegatedToKdcubeAccount[];
  loading: boolean;
  busy: boolean;
  error: string;
}

const initialState: DelegatedToKdcubeState = {
  enabled: false,
  providers: {},
  accounts: [],
  loading: true,
  busy: false,
  error: '',
};

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function resultError(result: { error?: string; message?: string } | null | undefined, fallback: string): string {
  return result?.message || result?.error || fallback;
}

export const loadDelegatedToKdcube = createAsyncThunk<DelegatedToKdcubeCatalogResult, void, { rejectValue: string }>(
  'delegatedToKdcube/load',
  async (_arg, { rejectWithValue }) => {
    try {
      const res = await getOp<DelegatedToKdcubeCatalogResult>('delegated_to_kdcube_catalog');
      if (res?.ok === false) return rejectWithValue(resultError(res, 'Failed to load delegated to KDCube'));
      return res || {};
    } catch (e) {
      return rejectWithValue(message(e));
    }
  },
);

export interface ConnectCredentialArgs {
  providerId: string;
  connectorAppId: string;
  accountId?: string;
  claimsMode?: 'add' | 'replace';
  externalSubject?: string;
  email?: string;
  displayName?: string;
  workspace?: string;
  claims: string[];
  secretKind: 'app_password' | 'access_token' | 'api_key' | 'secret';
  secretValue: string;
}

export const connectDelegatedToKdcubeCredential = createAsyncThunk<
  DelegatedToKdcubeMutationResult,
  ConnectCredentialArgs,
  { rejectValue: string }
>(
  'delegatedToKdcube/connectCredential',
  async (args, { rejectWithValue }) => {
    try {
      const res = await postOp<DelegatedToKdcubeMutationResult>('delegated_to_kdcube_connect_credential', {
        provider_id: args.providerId,
        connector_app_id: args.connectorAppId,
        ...(args.accountId ? {account_id: args.accountId} : {}),
        ...(args.claimsMode ? {claims_mode: args.claimsMode} : {}),
        external_subject: args.externalSubject || undefined,
        email: args.email || undefined,
        display_name: args.displayName || undefined,
        workspace: args.workspace || undefined,
        claims: args.claims,
        [args.secretKind]: args.secretValue,
      });
      if (res?.ok === false) return rejectWithValue(resultError(res, 'Failed to connect integration'));
      return res || {};
    } catch (e) {
      return rejectWithValue(message(e));
    }
  },
);

export interface StartOAuthArgs {
  providerId: string;
  connectorAppId: string;
  claims: string[];
  returnHint?: string;
  // Existing account this approval belongs to. With claimsMode "add" (the
  // default) the server unions the claims with the account's held set; with
  // "replace" the passed claims become the account's exact set.
  accountId?: string;
  claimsMode?: 'add' | 'replace';
}

export const startDelegatedToKdcubeOAuth = createAsyncThunk<
  DelegatedToKdcubeOAuthStartResult,
  StartOAuthArgs,
  { rejectValue: string }
>(
  'delegatedToKdcube/startOAuth',
  async (args, { rejectWithValue }) => {
    try {
      const res = await postOp<DelegatedToKdcubeOAuthStartResult>('delegated_to_kdcube_start_oauth', {
        provider_id: args.providerId,
        connector_app_id: args.connectorAppId,
        claims: args.claims,
        return_hint: args.returnHint || window.location.href,
        ...(args.accountId ? {account_id: args.accountId} : {}),
        ...(args.claimsMode ? {claims_mode: args.claimsMode} : {}),
      });
      if (res?.ok === false) return rejectWithValue(resultError(res, 'Failed to start OAuth connection'));
      return res || {};
    } catch (e) {
      return rejectWithValue(message(e));
    }
  },
);

export const disconnectDelegatedToKdcube = createAsyncThunk<
  DelegatedToKdcubeMutationResult,
  { accountId: string },
  { rejectValue: string }
>(
  'delegatedToKdcube/disconnect',
  async ({ accountId }, { rejectWithValue }) => {
    try {
      const res = await postOp<DelegatedToKdcubeMutationResult>('delegated_to_kdcube_disconnect', {
        account_id: accountId,
      });
      if (res?.ok === false) return rejectWithValue(resultError(res, 'Failed to disconnect integration'));
      return res || {};
    } catch (e) {
      return rejectWithValue(message(e));
    }
  },
);

const delegatedToKdcubeSlice = createSlice({
  name: 'delegatedToKdcube',
  initialState,
  reducers: {
    clearDelegatedToKdcubeError(state) {
      state.error = '';
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(loadDelegatedToKdcube.fulfilled, (state, action: PayloadAction<DelegatedToKdcubeCatalogResult>) => {
        state.loading = false;
        state.enabled = Boolean(action.payload.enabled);
        state.providers = action.payload.providers || {};
        state.accounts = action.payload.accounts || [];
      })
      .addCase(loadDelegatedToKdcube.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload ?? 'Failed to load delegated to KDCube';
      });

    builder
      .addCase(connectDelegatedToKdcubeCredential.pending, (state) => {
        state.busy = true;
        state.error = '';
      })
      .addCase(connectDelegatedToKdcubeCredential.fulfilled, (state, action) => {
        state.busy = false;
        if (action.payload.account) {
          const next = action.payload.account;
          state.accounts = [next, ...state.accounts.filter((account) => account.account_id !== next.account_id)];
        }
      })
      .addCase(connectDelegatedToKdcubeCredential.rejected, (state, action) => {
        state.busy = false;
        state.error = action.payload ?? 'Failed to connect integration';
      })
      .addCase(startDelegatedToKdcubeOAuth.pending, (state) => {
        state.busy = true;
        state.error = '';
      })
      .addCase(startDelegatedToKdcubeOAuth.fulfilled, (state) => {
        state.busy = false;
      })
      .addCase(startDelegatedToKdcubeOAuth.rejected, (state, action) => {
        state.busy = false;
        state.error = action.payload ?? 'Failed to start OAuth connection';
      })
      .addCase(disconnectDelegatedToKdcube.pending, (state) => {
        state.busy = true;
        state.error = '';
      })
      .addCase(disconnectDelegatedToKdcube.fulfilled, (state, action) => {
        state.busy = false;
        const accountId = action.payload.account_id || action.meta.arg.accountId;
        state.accounts = state.accounts.filter((account) => account.account_id !== accountId);
      })
      .addCase(disconnectDelegatedToKdcube.rejected, (state, action) => {
        state.busy = false;
        state.error = action.payload ?? 'Failed to disconnect integration';
      });
  },
});

export const { clearDelegatedToKdcubeError } = delegatedToKdcubeSlice.actions;
export default delegatedToKdcubeSlice.reducer;
