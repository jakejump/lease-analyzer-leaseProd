import React from "react";

type Props = {
  status: "uploaded" | "processed" | "failed";
  stage?: string | null;
  progress?: number | null;
};

const colorFor = (status: Props["status"]) => {
  switch (status) {
    case "processed":
      return "bg-green-600/20 text-green-300 border-green-600/40";
    case "failed":
      return "bg-red-600/20 text-red-300 border-red-600/40";
    default:
      return "bg-yellow-600/20 text-yellow-300 border-yellow-600/40";
  }
};

export default function StatusBadge({ status, stage, progress }: Props) {
  return (
    <span className={`inline-flex items-center gap-2 border px-2 py-1 rounded text-xs ${colorFor(status)}`}>
      <span className="capitalize">{status}</span>
      {status !== "processed" && (
        <span className="opacity-80">
          {stage ? `${stage}` : null}
          {typeof progress === "number" ? ` • ${progress}%` : null}
        </span>
      )}
    </span>
  );
}





