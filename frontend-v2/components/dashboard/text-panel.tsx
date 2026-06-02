"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle2, Copy, Check, Layers } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { apiUrl } from "@/lib/api";
import { TextResultSkeleton } from "./skeleton";
import { historyStore } from "./session-history";

const MAX = 10000;

type TextResult = { is_fake: boolean; label: string; confidence: number; p_fake: number; p_real: number; elapsed_ms: number };
type BatchResult = { index: number; status: string; is_fake?: boolean; label?: string; confidence?: number; p_fake?: number; p_real?: number; message?: string };

export function TextPanel() {
  const [text, setText] = useState("");
  const [batchMode, setBatchMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TextResult | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // ⌘+Enter to analyze
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && text.trim().length >= 20 && !loading) analyze();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const analyze = async () => {
    setLoading(true); setError(null); setResult(null); setBatchResults(null);
    try {
      if (batchMode) {
        const texts = text.split("---").map((t) => t.trim()).filter((t) => t.length >= 20);
        if (texts.length === 0) throw new Error("No valid segments. Separate texts with ---");
        const res = await fetch(apiUrl("/api/detect-text/batch"), {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texts }),
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.message ?? `Server error ${res.status}`);
        setBatchResults(j.results);
        toast.success(`Batch complete`, { description: `${j.fake_count} fake, ${j.real_count} real out of ${j.total}` });
      } else {
        const res = await fetch(apiUrl("/api/detect-text"), {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.message ?? `Server error ${res.status}`);
        setResult(j);
        historyStore.push({
          id: crypto.randomUUID(), type: "text", label: text.slice(0, 60) + (text.length > 60 ? "…" : ""),
          is_fake_or_refuted: j.is_fake, confidence: j.confidence, timestamp: new Date(),
        });
        toast[j.is_fake ? "error" : "success"](j.is_fake ? "Fake news detected" : "Appears legitimate", {
          description: `${(j.confidence * 100).toFixed(1)}% confidence · ${j.elapsed_ms}ms`,
        });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg); toast.error("Analysis failed", { description: msg });
    } finally { setLoading(false); }
  };

  const copyResult = () => {
    const data = batchResults ?? result;
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true); toast.success("Result copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const placeholder = batchMode
    ? "Paste first article or headline here...\n\n---\n\nPaste second article here...\n\n---\n\nContinue for each item."
    : "Paste a headline or full article here...";

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }} className="space-y-4">
      <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
          <h3 className="text-sm font-semibold text-zinc-800">Text Input</h3>
          <div className="flex items-center gap-3">
            <button onClick={() => { setBatchMode((v) => !v); setResult(null); setBatchResults(null); }}
              className={cn("flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border transition-colors",
                batchMode ? "border-brand/40 bg-brand/5 text-brand" : "border-zinc-200 text-zinc-400 hover:text-zinc-600"
              )}>
              <Layers className="w-3 h-3" /> Batch
            </button>
            <span className="text-xs text-zinc-400">{text.length} / {MAX.toLocaleString()}</span>
          </div>
        </div>
        <div className="p-5 space-y-4">
          {batchMode && (
            <p className="text-xs text-zinc-400 bg-zinc-50 rounded-lg px-3 py-2">
              Separate each article or headline with <code className="font-mono bg-zinc-200 px-1 rounded">---</code> on its own line. Up to 20 items.
            </p>
          )}
          <textarea value={text} onChange={(e) => { setText(e.target.value.slice(0, MAX)); setResult(null); setBatchResults(null); setError(null); }}
            rows={batchMode ? 12 : 8} placeholder={placeholder}
            className="w-full text-sm text-zinc-700 placeholder:text-zinc-400 bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-brand/30 focus:border-brand/40 transition-colors" />
          <div className="flex items-center gap-3">
            <button onClick={analyze} disabled={text.trim().length < 20 || loading}
              className="px-5 py-2 bg-brand text-white text-sm font-medium rounded-lg hover:bg-brand/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              {loading ? "Analysing…" : batchMode ? "Analyze All" : "Analyze Text"}
            </button>
            <p className="text-[11px] text-zinc-400"><kbd className="bg-zinc-100 px-1.5 py-0.5 rounded font-mono text-[10px]">⌘ Enter</kbd></p>
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 bg-red-50 border border-red-100 rounded-xl text-sm text-red-500">
          <AlertTriangle className="w-4 h-4 shrink-0" />{error}
        </div>
      )}

      {loading && <TextResultSkeleton />}

      {/* Single result */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
              <h3 className="text-sm font-semibold text-zinc-800">Result</h3>
              <div className="flex items-center gap-2">
                <button onClick={copyResult} className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? "Copied" : "Copy"}
                </button>
                <span className="text-xs text-zinc-400">{result.elapsed_ms} ms</span>
              </div>
            </div>
            <div className="p-5 space-y-4">
              <div className={cn("flex items-center gap-3 p-4 rounded-xl",
                result.is_fake ? "bg-red-50 border border-red-100"
                               : "bg-emerald-50 border border-emerald-100")}>
                {result.is_fake ? <AlertTriangle className="w-6 h-6 text-red-500 shrink-0" /> : <CheckCircle2 className="w-6 h-6 text-emerald-500 shrink-0" />}
                <div>
                  <p className={cn("text-lg font-bold", result.is_fake ? "text-red-600" : "text-emerald-600")}>{result.label}</p>
                  <p className="text-xs text-zinc-500">{(result.confidence * 100).toFixed(1)}% confidence</p>
                </div>
              </div>
              <ProbBar label="Real" value={result.p_real} color="bg-emerald-400" />
              <ProbBar label="Fake" value={result.p_fake} color="bg-red-400" />
            </div>
          </motion.div>
        )}

        {/* Batch results */}
        {batchResults && !loading && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
              <h3 className="text-sm font-semibold text-zinc-800">Batch Results</h3>
              <button onClick={copyResult} className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? "Copied" : "Copy JSON"}
              </button>
            </div>
            <div className="divide-y divide-zinc-100">
              {batchResults.map((r) => (
                <div key={r.index} className="flex items-center gap-3 px-5 py-3">
                  <span className="text-[10px] font-mono text-zinc-400 w-6">#{r.index + 1}</span>
                  {r.status === "ok" ? (
                    <>
                      <div className={cn("w-1.5 h-1.5 rounded-full shrink-0", r.is_fake ? "bg-red-400" : "bg-emerald-400")} />
                      <span className={cn("text-xs font-medium", r.is_fake ? "text-red-600" : "text-emerald-600")}>{r.label}</span>
                      <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden mx-2">
                        <div className={cn("h-full rounded-full", r.is_fake ? "bg-red-400" : "bg-emerald-400")}
                          style={{ width: `${((r.p_fake ?? 0) * 100).toFixed(0)}%` }} />
                      </div>
                      <span className="text-xs text-zinc-400 tabular-nums w-10 text-right">{((r.confidence ?? 0) * 100).toFixed(0)}%</span>
                    </>
                  ) : (
                    <span className="text-xs text-red-400">{r.message}</span>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-zinc-500 w-8">{label}</span>
      <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-700", color)} style={{ width: `${(value * 100).toFixed(1)}%` }} />
      </div>
      <span className="text-xs font-medium text-zinc-700 w-10 text-right tabular-nums">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}
