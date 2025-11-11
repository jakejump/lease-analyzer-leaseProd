import { useEffect, useState } from "react";
import Link from "next/link";
import { api, ProjectOut } from "../../lib/api";

export default function ProjectsIndexPage() {
  const [projects, setProjects] = useState<ProjectOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const fetchProjects = async () => {
    setLoading(true);
    const res = await api.get<ProjectOut[]>("/v1/projects");
    if (res.status < 400) setProjects(res.data || []);
    setLoading(false);
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const createProject = async () => {
    const res = await api.post<ProjectOut>("/v1/projects", { name, description });
    if (res.status < 400 && res.data?.id) {
      setName("");
      setDescription("");
      setProjects([res.data, ...projects]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 px-6 py-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-sm text-gray-400">Project name</label>
            <input className="w-full bg-gray-800 border border-gray-700 rounded p-2" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="flex-1">
            <label className="block text-sm text-gray-400">Description</label>
            <input className="w-full bg-gray-800 border border-gray-700 rounded p-2" value={description} onChange={(e) => setDescription(e.target.value)} />
          </div>
          <button onClick={createProject} disabled={!name} className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">
            Create
          </button>
        </div>

        <div className="border-t border-gray-800 pt-6">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold">Projects</h2>
            {loading && <span className="text-sm text-gray-400">Loading...</span>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {projects.map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`} className="block border border-gray-800 rounded p-4 bg-gray-800/60 hover:bg-gray-800">
                <div className="font-medium">{p.name}</div>
                {p.description && <div className="text-sm text-gray-400 mt-1 line-clamp-2">{p.description}</div>}
              </Link>
            ))}
            {projects.length === 0 && !loading && <div className="text-gray-400">No projects yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}



