import React, { useRef, useState } from "react";
import StatusBadge from "./StatusBadge";
import { LeaseVersionOut, VersionStatusResponse } from "../lib/api";

type Props = {
  versions: LeaseVersionOut[];
  statuses: Record<string, VersionStatusResponse | undefined>;
  currentVersionId?: string | null;
  onSelect: (versionId: string) => void;
  onUpdated?: () => Promise<void> | void;
};

export default function VersionTimeline({ versions, statuses, currentVersionId, onSelect, onUpdated }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newLabel, setNewLabel] = useState<string>("");
  const fileRef = useRef<HTMLInputElement | null>(null);
  const onRename = async (id: string) => {
    const fd = new FormData();
    fd.append("label", newLabel);
    await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"}/v1/versions/${id}`, { method: "PATCH", body: fd });
    setEditingId(null);
    try { await onUpdated?.(); } catch {}
  };
  const onReupload = async (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"}/v1/versions/${id}/reupload`, { method: "POST", body: fd });
    if (fileRef.current) fileRef.current.value = "";
    setEditingId(null);
    try { await onUpdated?.(); } catch {}
  };
  return (
    <div className="space-y-2">
      {versions.map((v) => {
        const s = statuses[v.id];
        const isCurrent = currentVersionId === v.id;
        return (
          <div
            key={v.id}
            onClick={() => onSelect(v.id)}
            role="button"
            className={`w-full border rounded px-3 py-2 bg-gray-800/60 hover:bg-gray-800 transition cursor-pointer ${
              isCurrent ? "ring-2 ring-blue-500" : ""
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                <span className="font-medium">{v.label || "Untitled Version"}</span>
                <span className="text-xs text-gray-400">{v.created_at ? new Date(v.created_at).toLocaleString() : ""}</span>
              </div>
              <StatusBadge status={s?.status || v.status} stage={s?.stage} progress={s?.progress} />
            </div>
            <div className="mt-2 flex gap-2">
              {editingId === v.id ? (
                <>
                  <input
                    type="text"
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.target.value)}
                    placeholder="New name"
                    className="bg-gray-900 border border-gray-700 rounded p-1 text-xs"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <button onClick={(e) => { e.stopPropagation(); onRename(v.id); }} className="text-xs px-2 py-1 rounded bg-blue-600 hover:bg-blue-700">Save</button>
                  <input ref={fileRef} type="file" className="text-xs" onClick={(e) => e.stopPropagation()} onChange={(e) => e.target.files && onReupload(v.id, e.target.files[0])} />
                  <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600">Cancel</button>
                </>
              ) : (
                <button onClick={(e) => { e.stopPropagation(); setEditingId(v.id); setNewLabel(v.label || ""); }} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600">Edit</button>
              )}
            </div>
          </div>
        );
      })}
      {versions.length === 0 && (
        <div className="text-sm text-gray-400">No versions yet. Upload a PDF to create your first version.</div>
      )}
    </div>
  );
}



