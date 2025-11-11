import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import { api, ProjectOut, LeaseVersionOut, VersionStatusResponse, RiskOut, AbnormalitiesOut } from "../../../lib/api";
import VersionTimeline from "../../../components/VersionTimeline";
import StatusBadge from "../../../components/StatusBadge";
import AnalysisTabs from "../../../components/AnalysisTabs";
import ClausesPanel from "../../../components/ClausesPanel";
import DiffPanel from "../../../components/DiffPanel";

export default function ProjectWorkspacePage() {
  const router = useRouter();
  const { projectId } = router.query as { projectId: string };

  const [project, setProject] = useState<ProjectOut | null>(null);
  const [versions, setVersions] = useState<LeaseVersionOut[]>([]);
  const [statuses, setStatuses] = useState<Record<string, VersionStatusResponse>>({});
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [risk, setRisk] = useState<RiskOut | null>(null);
  const [abn, setAbn] = useState<AbnormalitiesOut | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [tab, setTab] = useState<"overview" | "risk" | "abnormalities" | "qa" | "clauses" | "diff">("overview");
  const [uploading, setUploading] = useState(false);
  const [versionLabel, setVersionLabel] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selectedVersionId = useMemo(() => currentId || project?.current_version_id || versions[0]?.id || null, [currentId, project, versions]);

  useEffect(() => {
    if (!projectId) return;
    (async () => {
      const [p, v] = await Promise.all([
        api.get<ProjectOut>(`/v1/projects/${projectId}`),
        api.get<LeaseVersionOut[]>(`/v1/projects/${projectId}/versions`),
      ]);
      if (p.status < 400) setProject(p.data);
      if (v.status < 400) setVersions(v.data || []);
    })();
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return;
    let active = true;
    const tick = async () => {
      const res = await api.get<VersionStatusResponse[]>(`/v1/projects/${projectId}/versions/status`);
      if (res.status < 400 && active) {
        const map: Record<string, VersionStatusResponse> = {};
        for (const r of res.data || []) map[r.id] = r;
        setStatuses(map);
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [projectId]);

  useEffect(() => {
    const vid = selectedVersionId;
    if (!vid) {
      setRisk(null);
      setAbn(null);
      return;
    }
    (async () => {
      const [r, a] = await Promise.all([
        api.get<RiskOut>(`/v1/versions/${vid}/risk`),
        api.get<AbnormalitiesOut>(`/v1/versions/${vid}/abnormalities`),
      ]);
      if (r.status < 400) setRisk(r.data);
      if (a.status < 400) setAbn(a.data);
    })();
  }, [selectedVersionId]);

  const onUpload = async (file: File) => {
    if (!projectId || !file) return;
    // Show a name input prompt UI inline after selection: we already have a field below; if empty, default to filename
    const fd = new FormData();
    fd.append("file", file);
    const labelToUse = (versionLabel && versionLabel.trim()) || file.name.replace(/\.[^.]+$/, "");
    fd.append("label", labelToUse);
    setUploading(true);
    try {
      const res = await api.post<LeaseVersionOut>(`/v1/projects/${projectId}/versions/upload`, fd);
      if (res.status < 400) {
        const uploaded = res.data as LeaseVersionOut;
        // Prompt user to optionally rename after upload
        try {
          const proposed = window.prompt("Name this version", labelToUse);
          const finalName = (proposed || "").trim();
          if (finalName && finalName !== labelToUse) {
            const upd = new FormData();
            upd.append("label", finalName);
            await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"}/v1/versions/${uploaded.id}`, { method: "PATCH", body: upd });
          }
        } catch {}
        const v = await api.get<LeaseVersionOut[]>(`/v1/projects/${projectId}/versions`);
        if (v.status < 400) {
          setVersions(v.data || []);
        }
        // Start polling status immediately after upload so UI leaves "Uploading…"
        setTimeout(async () => {
          await api.get(`/v1/projects/${projectId}/versions/status`);
          // ignore result; existing poller will update state
        }, 200);
        // Auto-select the newly uploaded version so queries use its index
        setCurrentId(uploaded.id);
      }
    } catch (e) {
      console.error("upload failed", e);
    } finally {
      setUploading(false);
      setVersionLabel("");
    }
  };

  const riskColor = (score?: number | null) => {
    if (score == null) return "bg-gray-700 text-gray-200 border-gray-600";
    if (score <= 4) return "bg-red-600/20 text-red-300 border-red-600/40";
    if (score <= 6) return "bg-yellow-600/20 text-yellow-300 border-yellow-600/40";
    if (score <= 8) return "bg-green-600/20 text-green-300 border-green-600/40";
    return "bg-blue-600/20 text-blue-300 border-blue-600/40";
  };

  const riskLeft = (score?: number | null) => {
    if (score == null) return "border-gray-700";
    if (score <= 4) return "border-red-500";
    if (score <= 6) return "border-yellow-500";
    if (score <= 8) return "border-green-500";
    return "border-blue-500";
  };

  const ask = async () => {
    if (!selectedVersionId || !question) return;
    const fd = new FormData();
    fd.append("question", question);
    const res = await api.post<{ answer: string }>(`/v1/versions/${selectedVersionId}/ask`, fd);
    if (res.status < 400) setAnswer(res.data?.answer || null);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 px-6 py-6">
      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6">
        <aside className="col-span-12 md:col-span-3 space-y-4">
          <div className="border border-gray-800 rounded p-4 bg-gray-800/60">
            <div className="font-semibold mb-1">{project?.name || "Project"}</div>
            {project?.description && <div className="text-sm text-gray-400">{project.description}</div>}
          </div>

          <div className="border border-gray-800 rounded p-4 bg-gray-800/60">
            <div className="font-semibold mb-3">Versions</div>
            <VersionTimeline
              versions={versions}
              statuses={statuses}
              currentVersionId={selectedVersionId}
              onSelect={(id) => setCurrentId(id)}
              onUpdated={async () => {
                const v = await api.get<LeaseVersionOut[]>(`/v1/projects/${projectId}/versions`);
                if (v.status < 400) setVersions(v.data || []);
              }}
            />
          </div>

          <div className="border border-gray-800 rounded p-4 bg-gray-800/60 text-center">
            <div className="font-semibold mb-2">Upload new version</div>
            <div className="flex items-center justify-center gap-3">
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    // Reveal a transient inline name input by pre-filling with file stem
                    setVersionLabel(f.name.replace(/\.[^.]+$/, ""));
                    // Kick off upload immediately; user can rename via Versions → Edit afterwards if they skip here
                    onUpload(f);
                  }
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
              />
              <button
                title="Upload PDF"
                onClick={() => fileInputRef.current?.click()}
                className="h-10 w-10 rounded-full bg-green-600 hover:bg-green-700 text-white flex items-center justify-center"
              >
                +
              </button>
              {uploading === false && versionLabel !== undefined && null}
            </div>
            {uploading && <div className="text-xs text-gray-400 mt-2">Uploading and processing…</div>}
          </div>
        </aside>

        <main className="col-span-12 md:col-span-9 space-y-6">
          <div className="border border-gray-800 rounded p-4 bg-gray-800/60">
            <div className="flex items-center justify-between">
              <AnalysisTabs value={tab} onChange={setTab} />
              {selectedVersionId && (
                <StatusBadge
                  status={statuses[selectedVersionId]?.status || "uploaded"}
                  stage={statuses[selectedVersionId]?.stage}
                  progress={statuses[selectedVersionId]?.progress}
                />
              )}
            </div>

            {tab === "overview" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                <div className="border border-gray-700 rounded p-3">
                  <div className="font-medium mb-2">Risk</div>
                  {!risk?.payload && <div className="text-sm text-gray-400">No risk results yet.</div>}
                  {risk?.payload && (
                    <ul className="space-y-2 text-sm">
                      {Object.entries(risk.payload).map(([k, v]) => (
                        <li key={k} className={`flex items-start justify-between gap-4 pl-3 border-l-4 ${riskLeft(v.score)}`}>
                          <div className="flex-1">
                            <div className="font-medium capitalize">{k.replace(/_/g, " ")}</div>
                            <div className="text-gray-400">{v.explanation}</div>
                          </div>
                          <div className={`text-lg font-bold border rounded px-2 py-0.5 ${riskColor(v.score)}`}>{v.score ?? "?"}</div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="border border-gray-700 rounded p-3">
                  <div className="font-medium mb-2">Abnormalities</div>
                  {!abn?.payload?.length && <div className="text-sm text-gray-400">No abnormalities yet.</div>}
                  {abn?.payload?.length ? (
                    <ul className="space-y-2 text-sm">
                      {abn.payload.map((a, i) => (
                        <li key={i} className={a.impact === "beneficial" ? "text-green-400" : a.impact === "neutral" ? "text-yellow-400" : "text-red-400"}>
                          {a.text}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              </div>
            )}

            {tab === "risk" && (
              <div className="mt-2 space-y-3">
                {!risk?.payload && <div className="text-sm text-gray-400">No risk results yet.</div>}
                {risk?.payload && (
                  <ul className="space-y-3">
                    {Object.entries(risk.payload).map(([k, v]) => (
                      <li key={k} className={`border border-gray-700 rounded p-3 pl-3 border-l-4 ${riskLeft(v.score)}`}>
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <div className="font-medium capitalize">{k.replace(/_/g, " ")}</div>
                            <div className="text-gray-400 text-sm">{v.explanation}</div>
                          </div>
                          <div className={`text-xl font-bold border rounded px-2 py-0.5 ${riskColor(v.score)}`}>{v.score ?? "?"}</div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {tab === "abnormalities" && (
              <div className="mt-2">
                {!abn?.payload?.length && <div className="text-sm text-gray-400">No abnormalities yet.</div>}
                {abn?.payload?.length ? (
                  <ul className="space-y-2 text-sm">
                    {abn.payload.map((a, i) => (
                      <li key={i} className={a.impact === "beneficial" ? "text-green-400" : a.impact === "neutral" ? "text-yellow-400" : "text-red-400"}>
                        {a.text}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )}

            {tab === "qa" && (
              <div className="mt-2 space-y-2">
                <textarea className="w-full bg-gray-900 border border-gray-700 rounded p-2" rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about this version…" />
                <button onClick={ask} disabled={!selectedVersionId || !question} className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">
                  Ask
                </button>
                {answer && <div className="text-sm whitespace-pre-wrap text-gray-200 border border-gray-700 rounded p-3">{answer}</div>}
              </div>
            )}

            {tab === "clauses" && selectedVersionId && (
              <div className="mt-2">
                <ClausesPanel versionId={selectedVersionId} />
              </div>
            )}

            {tab === "diff" && (
              <div className="mt-2">
                <DiffPanel versions={versions} statuses={statuses} />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}


