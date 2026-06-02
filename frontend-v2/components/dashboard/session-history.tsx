"use client";

import { useState, useEffect } from "react";
import { Video, FileText, CheckSquare, AlertTriangle, CheckCircle2, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type HistoryEntry = {
  id: string;
  type: "visual" | "text" | "claim";
  label: string;
  is_fake_or_refuted: boolean;
  confidence: number;
  timestamp: Date;
};

const TYPE_META = {
  visual: { icon: Video,       label: "Visual" },
  text:   { icon: FileText,    label: "Text" },
  claim:  { icon: CheckSquare, label: "Claim" },
};

// Simple in-memory store exposed as a module-level array so panels can push to it.
// In a real app, lift state or use context.
export const historyStore: HistoryEntry[] = [];

export function SessionHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    const id = setInterval(() => {
      setEntries([...historyStore].reverse());
    }, 500);
    return () => clearInterval(id);
  }, []);

  if (entries.length === 0) return null;

  return (
    <section className="mt-8 border-t border-zinc-200 pt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-bold text-zinc-900">Session History</h3>
        <button
          onClick={() => { historyStore.length = 0; setEntries([]); }}
          className="flex items-center gap-1 text-xs text-zinc-400 hover:text-red-400 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" /> Clear
        </button>
      </div>
      <div className="space-y-1">
        {entries.map((e) => {
          const meta = TYPE_META[e.type];
          const Icon = meta.icon;
          return (
            <div key={e.id} className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-zinc-50 transition-colors">
              <div className="w-7 h-7 rounded-md bg-zinc-100 flex items-center justify-center shrink-0">
                <Icon className="w-3.5 h-3.5 text-zinc-500" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-700 truncate">{e.label}</p>
                <p className="text-[10px] text-zinc-400">{meta.label} · {e.timestamp.toLocaleTimeString()}</p>
              </div>
              <div className="flex items-center gap-1.5">
                {e.is_fake_or_refuted
                  ? <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                  : <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                }
                <span className={cn(
                  "text-[11px] font-medium px-1.5 py-0.5 rounded",
                  e.is_fake_or_refuted
                    ? "bg-red-50 text-red-500"
                    : "bg-emerald-50 text-emerald-600"
                )}>
                  {(e.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
