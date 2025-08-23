import {BundleInfo} from "../types/chat.ts";
import {cloneElement, useEffect, useId, useMemo, useState} from "react";

type BundlesListAdminProps = {
    bundles: Record<string, BundleInfo>;
    defaultId?: string;
    loading?: boolean;
    onReload?: () => void | Promise<void>;
    onSave: (b: BundleInfo) => void | Promise<void>;
    onDelete: (id: string) => void | Promise<void>;
    onSetDefault: (id: string) => void | Promise<void>;
};

export function BundlesListAdmin({
                              bundles, defaultId, loading, onReload, onSave, onDelete, onSetDefault
                          }: BundlesListAdminProps) {
    const [q, setQ] = useState("");
    const [editing, setEditing] = useState<BundleInfo | null>(null);
    const [adding, setAdding] = useState<boolean>(false);

    const rows = useMemo(() => {
        const arr = Object.values(bundles || {});
        const term = q.trim().toLowerCase();
        const filtered = term
            ? arr.filter(b =>
                b.id.toLowerCase().includes(term) ||
                (b.name || "").toLowerCase().includes(term)
            )
            : arr;
        return filtered.sort((a,b) => (a.name || a.id).localeCompare(b.name || b.id));
    }, [bundles, q]);

    return (
        <div className="space-y-3">
            {/* Toolbar - thin controls */}
            <div className="flex items-center gap-2">
                <input
                    className="h-8 flex-1 rounded-md border border-gray-300 px-2 text-sm outline-none focus:ring-2 focus:ring-gray-300"
                    placeholder="Search ID or name…"
                    value={q}
                    onChange={(e)=>setQ(e.target.value)}
                />
                <button
                    className="h-8 rounded-md border border-gray-300 px-3 text-sm hover:bg-gray-50"
                    onClick={()=>setAdding(true)}
                >
                    Add
                </button>
                <button
                    className="h-8 rounded-md border border-gray-300 px-3 text-sm hover:bg-gray-50 disabled:opacity-50"
                    disabled={loading}
                    onClick={()=>onReload?.()}
                >
                    Reload
                </button>
            </div>

            {/* Lines (table) */}
            <div className="border border-gray-200 rounded-md overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-600">
                    <tr>
                        <th className="text-left px-2 py-2 w-[42%]">ID</th>
                        <th className="text-left px-2 py-2 w-[42%]">Name</th>
                        <th className="text-left px-2 py-2">Default</th>
                        <th className="px-2 py-2"></th>
                    </tr>
                    </thead>
                    <tbody>
                    {rows.map(b => (
                        <tr key={b.id} className="border-t border-gray-200">
                            <td className="px-2 py-1.5">
                                <button
                                    className="text-blue-600 hover:underline underline-offset-2"
                                    onClick={()=>{ setEditing(b); setAdding(false); }}
                                    title="Edit"
                                >
                                    {b.id}
                                </button>
                            </td>
                            <td className="px-2 py-1.5">
                                <button
                                    className="text-blue-700 hover:underline underline-offset-2"
                                    onClick={()=>{ setEditing(b); setAdding(false); }}
                                    title="Edit"
                                >
                                    {b.name || b.id}
                                </button>
                            </td>
                            <td className="px-2 py-1.5">
                                <button
                                    className={`text-xs px-2 py-0.5 rounded ${defaultId===b.id ? 'bg-yellow-200 text-yellow-900' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                                    onClick={()=>onSetDefault(b.id)}
                                >
                                    {defaultId===b.id ? 'Default' : 'Set default'}
                                </button>
                            </td>
                            <td className="px-2 py-1.5 text-right">
                                <button
                                    className="text-xs px-2 py-0.5 rounded bg-red-50 text-red-700 hover:bg-red-100"
                                    onClick={()=>onDelete(b.id)}
                                >
                                    Delete
                                </button>
                            </td>
                        </tr>
                    ))}
                    {!loading && rows.length === 0 && (
                        <tr><td className="px-2 py-3 text-gray-500" colSpan={4}>No bundles match your search.</td></tr>
                    )}
                    </tbody>
                </table>
            </div>

            {/* Inline editor below the table */}
            {(editing || adding) && (
                <EditorCard
                    mode={adding ? 'create' : 'edit'}
                    initial={adding ? ({
                        id: "", name: "", path: "", module: "", singleton: false, description: ""
                    } as BundleInfo) : (editing as BundleInfo)}
                    readOnlyId={!adding}
                    onCancel={()=>{ setEditing(null); setAdding(false); }}
                    onSave={async (b) => {
                        await onSave(b);
                        setEditing(null);
                        setAdding(false);
                    }}
                    onDelete={adding ? undefined : async (id) => {
                        await onDelete?.(id);
                        setEditing(null);
                    }}
                />
            )}
        </div>
    );
}

function EditorCard({
                        mode, initial, readOnlyId=false, onCancel, onSave, onDelete
                    }: {
    mode: 'create' | 'edit';
    initial: BundleInfo;
    readOnlyId?: boolean;
    onCancel: () => void;
    onSave: (b: BundleInfo) => void | Promise<void>;
    onDelete?: (id: string) => void | Promise<void>;
}) {
    const [draft, setDraft] = useState<BundleInfo>(initial);
    useEffect(()=> setDraft(initial), [initial]);

    return (
        <div className="border border-gray-200 rounded-lg p-3 bg-white">
            <h4 className="text-sm font-semibold mb-2">
                {mode === 'create' ? 'New bundle' : 'Edit bundle'}
            </h4>
            <div className="grid grid-cols-1 gap-2">
                <Field label={`ID${readOnlyId ? '' : ' *'}`}>
                    <input
                        className="h-8 w-full rounded-md border border-gray-300 px-2 text-sm disabled:bg-gray-50 disabled:text-gray-500"
                        value={draft.id}
                        disabled={readOnlyId}
                        onChange={(e)=>setDraft({...draft, id: e.target.value})}
                        placeholder="unique-id"
                    />
                </Field>

                <Field label="Name">
                    <input
                        className="h-8 w-full rounded-md border border-gray-300 px-2 text-sm"
                        value={draft.name || ""}
                        onChange={(e)=>setDraft({...draft, name: e.target.value})}
                    />
                </Field>

                <Field label="Path *">
                    <input
                        className="h-8 w-full rounded-md border border-gray-300 px-2 text-sm"
                        value={draft.path || ""}
                        onChange={(e)=>setDraft({...draft, path: e.target.value})}
                        placeholder="/path/to/bundle(.py|/dir|.whl|.zip)"
                    />
                </Field>

                <Field label={`Module${draft.path && (/\.whl$|\.zip$/i.test(draft.path)) ? ' *' : ''}`}>
                    <input
                        className="h-8 w-full rounded-md border border-gray-300 px-2 text-sm"
                        value={draft.module || ""}
                        onChange={(e)=>setDraft({...draft, module: e.target.value})}
                        placeholder="package.module"
                    />
                </Field>

                <div className="flex items-center">
                    <input
                        id="singleton"
                        type="checkbox"
                        className="mr-2"
                        checked={!!draft.singleton}
                        onChange={(e)=>setDraft({...draft, singleton: e.target.checked})}
                    />
                    <label htmlFor="singleton" className="text-sm text-gray-700">Singleton</label>
                </div>

                <Field label="Description">
          <textarea
              className="min-h-[72px] w-full rounded-md border border-gray-300 p-2 text-sm"
              value={draft.description || ""}
              onChange={(e)=>setDraft({...draft, description: e.target.value})}
              placeholder="Optional"
          />
                </Field>
            </div>

            <div className="flex items-center justify-between gap-2 mt-3">
                <div className="flex gap-2">
                    {onDelete && mode === 'edit' && (
                        <button
                            className="h-8 rounded-md border border-red-300 text-red-700 px-3 text-sm hover:bg-red-50"
                            onClick={()=>onDelete(draft.id)}
                        >
                            Delete
                        </button>
                    )}
                </div>
                <div className="flex gap-2">
                    <button
                        className="h-8 rounded-md border border-gray-300 px-3 text-sm hover:bg-gray-50"
                        onClick={onCancel}
                    >
                        Cancel
                    </button>
                    <button
                        className="h-8 rounded-md bg-blue-600 text-white px-3 text-sm hover:bg-blue-700 disabled:opacity-50"
                        disabled={!draft.id?.trim() || !draft.path?.trim()}
                        onClick={()=>onSave(draft)}
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    const id = useId();
    return (
        <label className="block">
            <div className="mb-1 text-xs font-medium text-gray-600">{label}</div>
            {cloneElement(children as React.ReactElement, { id })}
        </label>
    );
}
