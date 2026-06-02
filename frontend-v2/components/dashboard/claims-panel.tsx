"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle2, MinusCircle, ChevronDown, ChevronUp, Loader2, ExternalLink, Copy, Check } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { apiUrl } from "@/lib/api";
import { TextResultSkeleton } from "./skeleton";
import { historyStore } from "./session-history";

const MAX_CLAIM = 5000;
const MAX_EVIDENCE = 2000;

type ClaimResult = { label: "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO"; confidence: number; p_support: number; p_refute: number; p_nei: number; elapsed_ms: number };
type ResearchResult = { claim: string; summary: string; report: string; sources: string[]; elapsed_seconds: number };

const labelConfig = {
  SUPPORTED:         { icon: CheckCircle2, color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-100" },
  REFUTED:           { icon: AlertTriangle, color: "text-red-600",    bg: "bg-red-50 border-red-100" },
  "NOT ENOUGH INFO": { icon: MinusCircle,  color: "text-zinc-600", bg: "bg-zinc-50 border-zinc-200" },
};

export function ClaimsPanel() {
  const [claim, setClaim] = useState("");
  const [evidence, setEvidence] = useState("");
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [researchLoading, setResearchLoading] = useState(false);
  const [result, setResult] = useState<ClaimResult | null>(null);
  const [research, setResearch] = useState<ResearchResult | null>(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // ⌘+Enter to verify
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && claim.trim().length >= 10 && !loading) verify();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const verify = async () => {
    setLoading(true); setError(null); setResult(null); setResearch(null);
    try {
      const body: Record<string, string> = { claim };
      if (evidence.trim()) body.evidence = evidence;
      const res = await fetch(apiUrl("/api/claim-verify"), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.message ?? `Server error ${res.status}`);
      setResult(j);
      historyStore.push({
        id: crypto.randomUUID(), type: "claim",
        label: claim.slice(0, 60) + (claim.length > 60 ? "…" : ""),
        is_fake_or_refuted: j.label === "REFUTED",
        confidence: j.confidence, timestamp: new Date(),
      });
      const toastFn = j.label === "SUPPORTED" ? toast.success : j.label === "REFUTED" ? toast.error : toast.warning;
      toastFn(j.label, { description: `${(j.confidence * 100).toFixed(1)}% confidence · ${j.elapsed_ms}ms` });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg); toast.error("Verification failed", { description: msg });
    } finally { setLoading(false); }
  };

  const deepResearch = async () => {
    setResearchLoading(true); setError(null);
    toast.info("Research started", { description: "Live web research in progress — takes 1–3 minutes…" });
    try {
      const res = await fetch(apiUrl("/api/research-claim"), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.message ?? `Server error ${res.status}`);
      setResearch(j); setReportOpen(true);
      toast.success("Research complete", { description: `${j.sources.length} sources found in ${j.elapsed_seconds.toFixed(0)}s` });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg); toast.error("Research failed", { description: msg });
    } finally { setResearchLoading(false); }
  };

  const copyResult = () => {
    const data = research ?? result;
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true); toast.success("Result copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }} className="space-y-4">
      {/* Input card */}
      <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
          <h3 className="text-sm font-semibold text-zinc-800">Claim</h3>
          <span className="text-xs text-zinc-400">{claim.length} / {MAX_CLAIM.toLocaleString()}</span>
        </div>
        <div className="p-5 space-y-3">
          <textarea value={claim}
            onChange={(e) => { setClaim(e.target.value.slice(0, MAX_CLAIM)); setResult(null); setResearch(null); setError(null); }}
            rows={5} placeholder="Enter a factual claim to verify..."
            className="w-full text-sm text-zinc-700 placeholder:text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-brand/30 focus:border-brand/40 transition-colors" />

          <button onClick={() => setEvidenceOpen(!evidenceOpen)}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
            {evidenceOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            {evidenceOpen ? "Hide evidence" : "Add supporting evidence (optional)"}
          </button>

          <AnimatePresence>
            {evidenceOpen && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="space-y-1">
                <div className="flex justify-between text-xs text-zinc-400"><span>Evidence</span><span>{evidence.length} / {MAX_EVIDENCE}</span></div>
                <textarea value={evidence} onChange={(e) => setEvidence(e.target.value.slice(0, MAX_EVIDENCE))}
                  rows={4} placeholder="Paste relevant context here..."
                  className="w-full text-sm text-zinc-700 placeholder:text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-brand/30 focus:border-brand/40 transition-colors" />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex items-center gap-3 flex-wrap">
            <button onClick={verify} disabled={claim.trim().length < 10 || loading}
              className="px-5 py-2 bg-brand text-white text-sm font-medium rounded-lg hover:bg-brand/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              {loading ? "Verifying…" : "Verify Claim"}
            </button>
            <button onClick={deepResearch} disabled={claim.trim().length < 15 || researchLoading}
              className="flex items-center gap-1.5 px-5 py-2 border border-zinc-200 text-zinc-600 text-sm font-medium rounded-lg hover:bg-zinc-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              {researchLoading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Researching…</> : "Deep Research"}
            </button>
            <p className="text-[11px] text-zinc-400"><kbd className="bg-zinc-100 px-1.5 py-0.5 rounded font-mono text-[10px]">⌘ Enter</kbd> to verify</p>
          </div>
          {researchLoading && <p className="text-xs text-zinc-400 animate-pulse">Live web research in progress — this takes 1–3 minutes…</p>}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 bg-red-50 border border-red-100 rounded-xl text-sm text-red-500">
          <AlertTriangle className="w-4 h-4 shrink-0" />{error}
        </div>
      )}

      {loading && <TextResultSkeleton />}

      <AnimatePresence>
        {result && !loading && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
              <h3 className="text-sm font-semibold text-zinc-800">Verdict</h3>
              <div className="flex items-center gap-2">
                <button onClick={copyResult} className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? "Copied" : "Copy"}
                </button>
                <span className="text-xs text-zinc-400">{result.elapsed_ms} ms</span>
              </div>
            </div>
            <div className="p-5 space-y-4">
              {(() => {
                const cfg = labelConfig[result.label];
                const Icon = cfg.icon;
                return (
                  <div className={cn("flex items-center gap-3 p-4 rounded-xl border", cfg.bg)}>
                    <Icon className={cn("w-6 h-6 shrink-0", cfg.color)} />
                    <div>
                      <p className={cn("text-lg font-bold", cfg.color)}>{result.label}</p>
                      <p className="text-xs text-zinc-500">{(result.confidence * 100).toFixed(1)}% confidence</p>
                    </div>
                  </div>
                );
              })()}
              <ProbBar label="Supported" value={result.p_support} color="bg-emerald-400" />
              <ProbBar label="Refuted"   value={result.p_refute}  color="bg-red-400" />
              <ProbBar label="Unclear"   value={result.p_nei}     color="bg-zinc-300" />
            </div>
          </motion.div>
        )}

        {research && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
            <button onClick={() => setReportOpen(!reportOpen)}
              className="w-full flex items-center justify-between px-5 py-3 border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-zinc-800">Research Report</h3>
                <span className="text-[10px] bg-brand/10 text-brand px-1.5 py-0.5 rounded font-medium">
                  {research.sources.length} sources · {research.elapsed_seconds.toFixed(0)}s
                </span>
              </div>
              {reportOpen ? <ChevronUp className="w-4 h-4 text-zinc-400" /> : <ChevronDown className="w-4 h-4 text-zinc-400" />}
            </button>
            <AnimatePresence>
              {reportOpen && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="p-5 space-y-4">
                  <p className="text-sm text-zinc-700 leading-relaxed">{research.summary}</p>
                  {research.report && (
                    <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-4 text-xs text-zinc-600 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                      {research.report}
                    </div>
                  )}
                  {research.sources.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-zinc-500 mb-2">Sources</p>
                      <ul className="space-y-1">
                        {research.sources.map((src, i) => (
                          <li key={i}>
                            <a href={src} target="_blank" rel="noopener noreferrer"
                              className="flex items-center gap-1 text-xs text-brand hover:underline truncate">
                              <ExternalLink className="w-3 h-3 shrink-0" />{src}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-zinc-500 w-16">{label}</span>
      <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-700", color)} style={{ width: `${(value * 100).toFixed(1)}%` }} />
      </div>
      <span className="text-xs font-medium text-zinc-700 w-10 text-right tabular-nums">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}
