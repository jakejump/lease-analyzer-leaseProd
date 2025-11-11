import axios, { AxiosInstance } from "axios";

/**
 * Central API client used across the app.
 * Reads base URL from NEXT_PUBLIC_API_BASE with a sensible default for dev.
 */
export const api: AxiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000",
  timeout: 360000,
  validateStatus: () => true,
});

export type Id = string;

export interface ProjectOut {
  id: Id;
  name: string;
  description?: string | null;
  current_version_id?: Id | null;
}

export interface LeaseVersionOut {
  id: Id;
  project_id: Id;
  label?: string | null;
  status: "uploaded" | "processed" | "failed";
  created_at?: string | null;
}

export interface VersionStatusResponse {
  id: Id;
  status: "uploaded" | "processed" | "failed";
  created_at?: string | null;
  updated_at?: string | null;
  stage?: string | null;
  progress?: number | null;
}

export interface RiskOut {
  payload: Record<string, { score: number | null; explanation: string }>;
  model?: string;
  created_at?: string | null;
}

export type Abnormality = { text: string; impact: "beneficial" | "harmful" | "neutral" };

export interface AbnormalitiesOut {
  payload: Abnormality[];
  model?: string;
  created_at?: string | null;
}

export interface ClausesResponse {
  clauses: string[];
}

export interface DiffChange {
  summary?: string;
  before?: string;
  after?: string;
  impact?: "beneficial" | "harmful" | "neutral";
}

export interface DiffResponse {
  base_version_id: Id;
  compare_version_id: Id;
  changes: DiffChange[];
}


