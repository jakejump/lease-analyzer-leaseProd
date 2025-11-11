import React from "react";

type TabKey = "overview" | "risk" | "abnormalities" | "qa" | "clauses" | "diff";

type Props = {
  value: TabKey;
  onChange: (v: TabKey) => void;
};

const TABS: { key: TabKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "risk", label: "Risk" },
  { key: "abnormalities", label: "Abnormalities" },
  { key: "qa", label: "Q&A" },
  { key: "clauses", label: "Clauses" },
  { key: "diff", label: "Diff" },
];

export default function AnalysisTabs({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-2 border-b border-gray-800 mb-3">
      {TABS.map((t) => {
        const active = t.key === value;
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={`px-3 py-2 text-sm rounded-t ${
              active ? "bg-gray-800 text-white border-x border-t border-gray-800" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}





