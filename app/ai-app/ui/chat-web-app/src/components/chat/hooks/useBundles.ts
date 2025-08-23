/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 Elena Viter
 */

import {useCallback, useEffect, useState} from 'react';
import {BundleInfo} from '../types/types';

export interface UseBundlesResult {
    bundles: Record<string, BundleInfo>;
    defaultId?: string;
    loading: boolean;
    error?: string;
    reload: () => Promise<void>;
}

export function useBundles(baseUrl: string, authContext: any): UseBundlesResult {
    const [bundles, setBundles] = useState<Record<string, BundleInfo>>({});
    const [defaultId, setDefaultId] = useState<string | undefined>();
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>();

    const reload = useCallback(async () => {
        setLoading(true); setError(undefined);
        try {
            const headers: HeadersInit = [['Content-Type','application/json']];
            authContext?.appendAuthHeader?.(headers);
            const res = await fetch(`${baseUrl}/bundles`, {method: 'GET', headers});
            const data = await res.json();
            if (!res.ok) throw new Error((data?.detail as string) || 'Failed to load bundles');
            setBundles(data.available_bundles || {});
            setDefaultId(data.default_bundle_id);
        } catch (e:any) {
            setError(e.message || 'Failed to load bundles');
        } finally {
            setLoading(false);
        }
    }, [baseUrl, authContext]);

    useEffect(() => { reload(); }, [reload]);

    return {bundles, defaultId, loading, error, reload};
}
