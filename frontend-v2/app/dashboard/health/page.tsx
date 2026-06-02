"use client";

import { useState, useEffect } from "react";
import { CheckCircle2, XCircle, Loader2, RefreshCw } from "lucide-react";
import { apiUrl } from "@/lib/api";

type EndpointStatus = {
  name: string;
  url: string;
  method: "GET" | "POST";
  body?: object;
  status: "ok" | "error" | "loading" | "idle";
  latency?: number;
  message?: string;
};

const ENDPOINTS: Omit<EndpointStatus, "status" | "latency" | "message">[] = [
  { name: "Health Check",         url: apiUrl("/api/health"),        method: "GET" },
  { name: "Visual Detect",        url: apiUrl("/api/detect"),        method: "POST" },
  { name: "Text Detection",       url: apiUrl("/api/detect-text"),   method: "POST", body: { text: "ping" } },
  { name: "Claim Verification",   url: apiUrl("/api/claim-verify"),  method: "POST", body: { claim: "ping" } },
  { name: "Research Agent",       url: apiUrl("/api/research-claim"), method: "POST", body: { claim: "ping" } },
];

export default function HealthPage() {
  const [endpoints, setEndpoints] = useState<EndpointStatus[]>(
    ENDPOINTS.map((e) => ({ ...e, status: "idle" }))
  );
  const [checking, setChecking] = useState(false);

  const checkAll = async () => {
    setChecking(true);
    setEndpoints((prev) => prev.map((e) => ({ ...e, status: "loading" })));

    const results = await Promise.all(
      ENDPOINTS.map(async (ep) => {
        const t0 = performance.now();
        try {
          const res = await fetch(ep.url, {
            method: ep.method,
            headers: ep.body ? { "Content-Type": "application/json" } : undefined,
            body: ep.body ? JSON.stringify(ep.body) : undefined,
          });
          const latency = Math.round(performance.now() - t0);
          return { ...ep, status: res.ok ? "ok" : "error", latency, message: res.ok ? undefined : `HTTP ${res.status}` } as EndpointStatus;
        } catch {
          const latency = Math.round(performance.now() - t0);
          return { ...ep, status: "error", latency, message: "Connection refused" } as EndpointStatus;
        }
      })
    );

    setEndpoints(results);
    setChecking(false);
  };

  useEffect(() => { checkAll(); }, []);

  const allOk = endpoints.every((e) => e.status === "ok");
  const anyError = endpoints.some((e) => e.status === "error");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-zinc-900">API Health</h2>
          <p className="text-sm text-zinc-400 mt-1">Next.js proxies /api/* to the Flask backend on port 5000</p>
        </div>
        <button
          onClick={checkAll}
          disabled={checking}
          className="flex items-center gap-1.5 px-4 py-2 border border-zinc-200 text-zinc-600 text-sm font-medium rounded-lg hover:bg-zinc-50 disabled:opacity-40 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${checking ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Overall status */}
      <div className={`flex items-center gap-3 p-4 rounded-xl border ${
        anyError ? "bg-red-50 border-red-100" : allOk ? "bg-emerald-50 border-emerald-100" : "bg-zinc-50 border-zinc-200"
      }`}>
        {anyError
          ? <XCircle className="w-5 h-5 text-red-500 shrink-0" />
          : allOk
          ? <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
          : <Loader2 className="w-5 h-5 text-zinc-400 animate-spin shrink-0" />
        }
        <p className="text-sm font-medium text-zinc-700">
          {anyError ? "One or more endpoints are unreachable" : allOk ? "All endpoints healthy" : "Checking…"}
        </p>
      </div>

      {/* Endpoint list */}
      <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
        <div className="px-5 py-3 border-b border-zinc-100">
          <h3 className="text-sm font-semibold text-zinc-800">Endpoints</h3>
        </div>
        <div className="divide-y divide-zinc-100">
          {endpoints.map((ep) => (
            <div key={ep.name} className="flex items-center gap-4 px-5 py-3.5">
              <div className="shrink-0">
                {ep.status === "loading" && <Loader2 className="w-4 h-4 text-zinc-400 animate-spin" />}
                {ep.status === "ok"      && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                {ep.status === "error"   && <XCircle className="w-4 h-4 text-red-400" />}
                {ep.status === "idle"    && <div className="w-4 h-4 rounded-full bg-zinc-200" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-800">{ep.name}</p>
                <p className="text-xs text-zinc-400 font-mono truncate">{ep.method} {new URL(ep.url, "http://localhost").pathname}</p>
                {ep.message && <p className="text-xs text-red-400 mt-0.5">{ep.message}</p>}
              </div>
              {ep.latency != null && (
                <span className="text-xs text-zinc-400 tabular-nums shrink-0">{ep.latency} ms</span>
              )}
              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0 ${
                ep.status === "ok"      ? "bg-emerald-50 text-emerald-600" :
                ep.status === "error"   ? "bg-red-50 text-red-500" :
                ep.status === "loading" ? "bg-zinc-100 text-zinc-400" :
                "bg-zinc-100 text-zinc-400"
              }`}>
                {ep.status === "loading" ? "checking" : ep.status === "idle" ? "—" : ep.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
