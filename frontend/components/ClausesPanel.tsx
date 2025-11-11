import React, { useState } from "react";
import { api, ClausesResponse } from "../lib/api";

const PRESETS = [
  "tenant improvements",
  "SNDA",
  "defaults",
  "insurance",
  "use and exclusivity",
  "renewal options",
  "rent escalations",
];

type Props = {
  versionId: string;
};

export default function ClausesPanel({ versionId }: Props) {
  const [topic, setTopic] = useState("");
  const [items, setItems] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const run = async (t: string) => {
    if (!versionId) return;
    setLoading(true);
    setTopic(t);
    const fd = new FormData();
    fd.append("topic", t);
    const res = await api.post<ClausesResponse>(`/v1/versions/${versionId}/clauses`, fd);
    if (res.status < 400) setItems(res.data?.clauses || []);
    setLoading(false);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button key={p} onClick={() => run(p)} className={`px-3 py-1 rounded text-sm border border-gray-700 ${topic === p ? "bg-blue-600" : "bg-gray-800 hover:bg-gray-700"}`}>
            {p}
          </button>
        ))}
      </div>
      {loading && <div className="text-sm text-gray-400">Gathering clauses…</div>}
      <ul className="list-disc pl-5 space-y-2 text-sm">
        {items.map((t, i) => (
          <li key={i}>
            <div className="whitespace-pre-wrap break-words">{(t || "").replace(/\n  /g, "\n      ")}</div>
          </li>
        ))}
      </ul>
      {!items.length && !loading && <div className="text-sm text-gray-400">No clauses yet. Choose a topic above.</div>}
    </div>
  );
}





