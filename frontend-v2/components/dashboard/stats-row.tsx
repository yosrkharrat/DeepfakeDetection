"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const stats = [
  { label: "Accuracy",  value: "95.87%", sub: "4,937 test samples" },
  { label: "Precision", value: "99.04%", sub: "Low false-positive rate" },
  { label: "F1 Score",  value: "96.90%", sub: "Balanced performance" },
  { label: "AUC-ROC",   value: "0.9951", sub: "Near-perfect separation" },
];

export function StatsRow() {
  return (
    <section className="grid grid-cols-2 lg:grid-cols-4 gap-0 mb-8">
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: i * 0.07 }}
          className={cn(
            "flex flex-col py-1",
            i < stats.length - 1 && "pr-6 border-r border-zinc-200",
            i > 0 && "pl-6"
          )}
        >
          <span className="text-sm text-zinc-400 font-medium mb-1">{s.label}</span>
          <span className="text-2xl font-bold text-zinc-900 tabular-nums">{s.value}</span>
          <span className="text-[11px] text-zinc-400 mt-0.5">{s.sub}</span>
        </motion.div>
      ))}
    </section>
  );
}
