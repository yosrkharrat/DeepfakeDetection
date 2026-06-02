"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, RefreshCw, AlertTriangle, CheckCircle2, Copy, Check, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ConfidenceGauge } from "./confidence-gauge";
import { ResultSkeleton } from "./skeleton";
import { historyStore } from "./session-history";

type FrameResult = { frame_index: number; is_fake: boolean; confidence: number; best_fake_prob: number; num_faces: number };
type DetectResult = {
  media_type: "image" | "video"; is_fake: boolean; confidence: number; num_faces?: number;
  avg_fake_probability?: number; fake_frame_count?: number; num_frames_analyzed?: number;
  frame_results?: FrameResult[]; elapsed_seconds: number;
};

const MAX_VIDEO_FRAMES = 16;

export function VisualPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [showThreshold, setShowThreshold] = useState(false);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ⌘+Enter to analyze
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && file && !loading && !result) analyze();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const handleFile = useCallback((f: File) => {
    setFile(f); setResult(null); setError(null);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); if (inputRef.current) inputRef.current.value = ""; };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("http://localhost:5001/api/detect", { method: "POST", body: fd });
      if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.error ?? `Server error ${res.status}`); }
      const data: DetectResult = await res.json();

      // Apply custom threshold
      const fakePob = data.avg_fake_probability ?? (data.is_fake ? data.confidence : 1 - data.confidence);
      const overridden = { ...data, is_fake: fakePob >= threshold, confidence: fakePob >= threshold ? fakePob : 1 - fakePob };
      setResult(overridden);

      historyStore.push({
        id: crypto.randomUUID(), type: "visual", label: file.name,
        is_fake_or_refuted: overridden.is_fake, confidence: overridden.confidence, timestamp: new Date(),
      });

      toast[overridden.is_fake ? "error" : "success"](
        overridden.is_fake ? "Deepfake detected" : "Authentic media",
        { description: `${(overridden.confidence * 100).toFixed(1)}% confidence · ${data.elapsed_seconds}s` }
      );
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg);
      toast.error("Analysis failed", { description: msg });
    } finally { setLoading(false); }
  };

  const copyResult = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    toast.success("Result copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  // Recompute verdict when threshold slider changes on existing result
  const fakePob = result
    ? result.avg_fake_probability ?? (result.is_fake ? result.confidence : 1 - result.confidence)
    : null;
  const thresholdedResult = fakePob !== null && result
    ? { ...result, is_fake: fakePob >= threshold, confidence: fakePob >= threshold ? fakePob : 1 - fakePob }
    : result;

  const isVideo = file?.type.startsWith("video/");

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }} className="space-y-6">

      {/* Threshold toggle */}
      <div className="flex items-center justify-end">
        <button onClick={() => setShowThreshold((v) => !v)}
          className={cn("flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors",
            showThreshold ? "border-brand/40 bg-brand/5 text-brand" : "border-zinc-200 text-zinc-400 hover:text-zinc-600"
          )}>
          <SlidersHorizontal className="w-3.5 h-3.5" /> Threshold: {threshold.toFixed(2)}
        </button>
      </div>

      <AnimatePresence>
        {showThreshold && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
            className="bg-zinc-50 border border-zinc-200 rounded-xl px-5 py-4">
            <div className="flex items-center gap-4">
              <span className="text-xs text-zinc-500 w-24">Fake threshold</span>
              <input type="range" min="0.1" max="0.9" step="0.01" value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="flex-1 accent-[#0ea5e9]" />
              <span className="text-sm font-mono font-semibold text-zinc-700 w-10 text-right">{threshold.toFixed(2)}</span>
            </div>
            <p className="text-[11px] text-zinc-400 mt-2">Scores above this threshold are classified as FAKE. Default: 0.50</p>
          </motion.div>
        )}
      </AnimatePresence>

      {!file ? (
        /* Upload zone */
        <div
          className={cn("border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-3 py-16 px-6 cursor-pointer transition-colors",
            dragging ? "border-brand bg-brand/5" : "border-zinc-200 hover:border-brand/40 hover:bg-zinc-50"
          )}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="w-12 h-12 rounded-xl bg-zinc-100 flex items-center justify-center">
            <Upload className="w-5 h-5 text-zinc-400" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-zinc-700">Drop your file here</p>
            <p className="text-xs text-zinc-400 mt-1">Images: JPG, PNG · Max 10 MB<br />Videos: MP4, AVI, MOV · Max 200 MB</p>
          </div>
          <button className="px-4 py-1.5 bg-brand text-white text-xs font-medium rounded-lg hover:bg-brand/90 transition-colors">
            Browse files
          </button>
          <p className="text-[11px] text-zinc-400">or press <kbd className="bg-zinc-100 px-1.5 py-0.5 rounded text-[10px] font-mono">⌘ Enter</kbd> after selecting</p>
          <input ref={inputRef} type="file" hidden accept=".jpg,.jpeg,.png,.mp4,.avi,.mov"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
        </div>
      ) : loading ? (
        <ResultSkeleton />
      ) : (
        /* Analysis grid */
        <AnimatePresence>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Preview */}
            <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden">
              <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
                <h3 className="text-sm font-semibold text-zinc-800">Input</h3>
                <span className="text-xs text-zinc-400 truncate max-w-[140px]">{file.name}</span>
              </div>
              <div className="p-4 flex items-center justify-center min-h-[200px] bg-zinc-50">
                {isVideo
                  ? <video src={preview!} controls className="max-h-48 rounded-lg w-full object-contain" />
                  : <img src={preview!} alt="preview" className="max-h-48 rounded-lg object-contain" />
                }
              </div>
            </div>

            {/* Result */}
            <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
                <h3 className="text-sm font-semibold text-zinc-800">Result</h3>
                <div className="flex items-center gap-2">
                  {thresholdedResult && (
                    <button onClick={copyResult}
                      className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                      {copied ? "Copied" : "Copy"}
                    </button>
                  )}
                  {thresholdedResult && <span className="text-xs text-zinc-400">{result?.elapsed_seconds}s</span>}
                </div>
              </div>

              <div className="flex-1 flex flex-col items-center justify-center p-6 gap-4">
                {!thresholdedResult && !error && (
                  <div className="flex flex-col items-center gap-3">
                    <button onClick={analyze}
                      className="px-6 py-2.5 bg-brand text-white text-sm font-medium rounded-lg hover:bg-brand/90 transition-colors">
                      Analyze
                    </button>
                    <p className="text-[11px] text-zinc-400">or <kbd className="bg-zinc-100 px-1.5 py-0.5 rounded font-mono text-[10px]">⌘ Enter</kbd></p>
                  </div>
                )}

                {error && (
                  <div className="flex flex-col items-center gap-2 text-center">
                    <AlertTriangle className="w-8 h-8 text-red-400" />
                    <p className="text-sm text-red-500">{error}</p>
                    <button onClick={analyze} className="text-xs text-brand hover:underline">Retry</button>
                  </div>
                )}

                {thresholdedResult && (
                  <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full space-y-4">
                    {/* Gauge */}
                    <div className="flex justify-center">
                      <ConfidenceGauge value={thresholdedResult.confidence} isFake={thresholdedResult.is_fake} />
                    </div>

                    {/* Probability bar */}
                    <div>
                      <div className="flex justify-between text-xs text-zinc-500 mb-1"><span>Real</span><span>Fake</span></div>
                      <div className="h-2 bg-zinc-100 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-emerald-400 to-red-400 rounded-full transition-all duration-700"
                          style={{ width: `${((fakePob ?? 0) * 100).toFixed(1)}%` }} />
                      </div>
                      <div className="flex justify-between text-[11px] text-zinc-400 mt-1">
                        <span>{(100 - (fakePob ?? 0) * 100).toFixed(1)}%</span>
                        <span>{((fakePob ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {thresholdedResult.num_faces != null && (
                        <div className="bg-zinc-50 rounded-lg px-3 py-2">
                          <p className="text-zinc-400">Faces</p>
                          <p className="font-semibold text-zinc-800">{thresholdedResult.num_faces}</p>
                        </div>
                      )}
                      {thresholdedResult.num_frames_analyzed != null && (
                        <div className="bg-zinc-50 rounded-lg px-3 py-2">
                          <p className="text-zinc-400">Frames</p>
                          <p className="font-semibold text-zinc-800">
                            {thresholdedResult.fake_frame_count}/{thresholdedResult.num_frames_analyzed} fake
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Frame timeline */}
                    {thresholdedResult.frame_results && thresholdedResult.frame_results.length > 0 && (
                      <div>
                        <p className="text-xs text-zinc-400 mb-2">Frame timeline</p>
                        <div className="flex gap-0.5 h-8 items-end">
                          {thresholdedResult.frame_results.map((fr) => (
                            <div key={fr.frame_index}
                              title={`Frame ${fr.frame_index} · ${(fr.best_fake_prob * 100).toFixed(0)}%`}
                              className={cn("flex-1 rounded-sm", fr.best_fake_prob >= threshold ? "bg-red-400" : "bg-emerald-400")}
                              style={{ height: `${Math.max(20, fr.best_fake_prob * 100)}%` }} />
                          ))}
                        </div>
                        <p className="text-[10px] text-zinc-400 mt-1">
                          Each bar = one frame · <span className="text-red-400">■</span> Fake · <span className="text-emerald-400">■</span> Real
                        </p>
                      </div>
                    )}
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      )}

      {file && (
        <button onClick={reset} className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
          <RefreshCw className="w-3.5 h-3.5" /> Analyse another file
        </button>
      )}
    </motion.div>
  );
}
