import React, { useEffect, useMemo, useState } from "react";
import { api, DiffResponse, LeaseVersionOut, VersionStatusResponse } from "../lib/api";

type Props = {
  projectId: string;
  versions: LeaseVersionOut[];
  statuses: Record<string, VersionStatusResponse | undefined>;
};

const impactColor = (imp?: string) =>
  imp === "beneficial" ? "text-green-400" : imp === "neutral" ? "text-yellow-400" : "text-red-400";

export default function DiffPanel({ projectId, versions, statuses }: Props) {
  const [baseId, setBaseId] = useState<string | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const [res, setRes] = useState<DiffResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!baseId && versions.length >= 2) setBaseId(versions[1].id);
    if (!compareId && versions.length >= 1) setCompareId(versions[0].id);
  }, [versions]);

  const bothProcessed = useMemo(() => {
    if (!baseId || !compareId) return false;
    const bs = statuses[baseId];
    const cs = statuses[compareId];
    return bs?.status === "processed" && cs?.status === "processed";
  }, [baseId, compareId, statuses]);

  const canRun = useMemo(() => baseId && compareId && baseId !== compareId && bothProcessed, [baseId, compareId, bothProcessed]);

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("base_version_id", baseId as string);
    fd.append("compare_version_id", compareId as string);
    const out = await api.post<DiffResponse>("/v1/diff", fd);
    if (out.status < 400) setRes(out.data);
    setLoading(false);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <label className="block text-xs text-gray-400">Base version</label>
          <select className="w-full bg-gray-900 border border-gray-700 rounded p-2" value={baseId || ""} onChange={(e) => setBaseId(e.target.value)}>
            {versions.map((v) => (
              <option key={v.id} value={v.id}>{v.label || v.id}</option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-xs text-gray-400">Compare version</label>
          <select className="w-full bg-gray-900 border border-gray-700 rounded p-2" value={compareId || ""} onChange={(e) => setCompareId(e.target.value)}>
            {versions.map((v) => (
              <option key={v.id} value={v.id}>{v.label || v.id}</option>
            ))}
          </select>
        </div>
        <button onClick={run} disabled={!canRun || loading} className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50">
          {loading ? "Comparing…" : bothProcessed ? "Compare" : "Waiting for processing…"}
        </button>
      </div>

      {res?.changes?.length ? (
        <ul className="space-y-3">
          {res.changes.map((c, i) => (
            <li key={i} className="border border-gray-700 rounded p-3">
              <div className={`font-medium ${impactColor(c.impact)}`}>{c.summary || "Change"}</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm mt-2">
                <div>
                  <div className="text-gray-400 mb-1">Before</div>
                  <div className="whitespace-pre-wrap break-words">{c.before || ""}</div>
                </div>
                <div>
                  <div className="text-gray-400 mb-1">After</div>
                  <div className="whitespace-pre-wrap break-words">{c.after || ""}</div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="text-sm text-gray-400">
          {bothProcessed
            ? "No changes detected for the selected versions."
            : "Select two different versions. The Compare button will enable once both are processed."}
        </div>
      )}
    </div>
  );
}


