"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { historyStore, type HistoryEntry } from "./session-history";

export function SessionStats() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    const id = setInterval(() => setEntries([...historyStore]), 500);
    return () => clearInterval(id);
  }, []);

  if (entries.length === 0) return null;

  const fakeCount = entries.filter((e) => e.is_fake_or_refuted).length;
  const realCount = entries.length - fakeCount;
  const byType = {
    visual: entries.filter((e) => e.type === "visual").length,
    text:   entries.filter((e) => e.type === "text").length,
    claim:  entries.filter((e) => e.type === "claim").length,
  };
  const total = entries.length;

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8 border-t border-zinc-200 pt-6"
    >
      {/* Fake vs Real */}
      <div className="bg-white border border-zinc-200 rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-zinc-800 mb-4">Session verdicts</h3>
        <div className="space-y-3">
          <BarRow label="Real" count={realCount} total={total} color="bg-emerald-400" />
          <BarRow label="Fake" count={fakeCount} total={total} color="bg-red-400" />
        </div>
        <p className="text-[11px] text-zinc-400 mt-4">{total} total analysis{total !== 1 ? "es" : ""} this session</p>
      </div>

      {/* By type donut */}
      <div className="bg-white border border-zinc-200 rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-zinc-800 mb-4">By detection type</h3>
        <div className="flex items-center gap-6">
          <DonutChart visual={byType.visual} text={byType.text} claim={byType.claim} total={total} />
          <div className="space-y-2 flex-1">
            <Legend color="bg-brand"       label="Visual" count={byType.visual} total={total} />
            <Legend color="bg-violet-400"  label="Text"   count={byType.text}   total={total} />
            <Legend color="bg-amber-400"   label="Claims" count={byType.claim}  total={total} />
          </div>
        </div>
      </div>
    </motion.section>
  );
}

function BarRow({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-zinc-500 w-8">{label}</span>
      <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
        <motion.div className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }} animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }} />
      </div>
      <span className="text-xs font-semibold text-zinc-700 tabular-nums w-6 text-right">{count}</span>
    </div>
  );
}

function Legend({ color, label, count, total }: { color: string; label: string; count: number; total: number }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2.5 h-2.5 rounded-sm ${color} shrink-0`} />
      <span className="text-xs text-zinc-500 flex-1">{label}</span>
      <span className="text-xs font-semibold text-zinc-700 tabular-nums">{count}</span>
      <span className="text-[10px] text-zinc-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

function DonutChart({ visual, text, claim, total }: { visual: number; text: number; claim: number; total: number }) {
  const R = 28; const cx = 36; const cy = 36;
  const circ = 2 * Math.PI * R;
  const segments = [
    { count: visual, color: "#0ea5e9" },
    { count: text,   color: "#a78bfa" },
    { count: claim,  color: "#fbbf24" },
  ];
  let offset = 0;
  const arcs = segments.map((s) => {
    const dash = total > 0 ? (s.count / total) * circ : 0;
    const arc = { dash, offset, color: s.color };
    offset += dash;
    return arc;
  });
  return (
    <svg width="72" height="72" viewBox="0 0 72 72" className="shrink-0">
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="#e4e4e7" strokeWidth="10" />
      {arcs.map((a, i) => a.dash > 0 ? (
        <circle key={i} cx={cx} cy={cy} r={R} fill="none" stroke={a.color} strokeWidth="10"
          strokeDasharray={`${a.dash} ${circ}`} strokeDashoffset={-a.offset + circ / 4}
          style={{ transition: "stroke-dasharray 0.6s ease" }} />
      ) : null)}
      <text x={cx} y={cy + 5} textAnchor="middle" fontSize="14" fontWeight="700" fill="#71717a">{total}</text>
    </svg>
  );
}
