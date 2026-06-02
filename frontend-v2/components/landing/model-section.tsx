"use client"

import { motion } from "framer-motion"
import { ChevronRight } from "lucide-react"

const metrics = [
  { label: "Accuracy",           value: "95.87%", sub: "4,937 test samples" },
  { label: "AUC-ROC",            value: "0.9951", sub: "Near-perfect separation" },
  { label: "False Positive Rate", value: "1.97%",  sub: "Minimal false alarms" },
  { label: "Training samples",   value: "32,898", sub: "FaceForensics++ C23" },
]

export function ModelSection() {
  return (
    <div id="model" className="relative py-20 md:py-40 bg-white">
      <div className="max-w-5xl mx-auto px-6">
        {/* Label */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}
          className="flex items-center gap-2 mb-6"
        >
          <div className="w-2 h-2 rounded-full bg-brand" />
          <span className="text-zinc-400 text-sm">Dual-stream architecture</span>
          <ChevronRight className="w-4 h-4 text-zinc-400" />
        </motion.div>

        {/* Heading */}
        <motion.h2
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.1 }}
          className="text-3xl sm:text-4xl md:text-5xl lg:text-[54px] text-zinc-900 max-w-3xl mb-8"
          style={{ letterSpacing: "-0.032em", fontWeight: 538, lineHeight: 1.1 }}
        >
          Two streams see what one misses
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.2 }}
          className="text-zinc-500 max-w-md mb-12"
        >
          <span className="text-zinc-900 font-medium">RGB stream catches visual artifacts.</span>{" "}
          The FFT stream catches frequency-domain anomalies that the eye can't see.
          Both feed into a fusion layer for the final verdict.
        </motion.p>

        {/* Architecture diagram */}
        <motion.div
          initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <div>
            <div>
              <div className="bg-zinc-50 border border-zinc-200 rounded-2xl p-8">
                <div className="flex items-center gap-4 justify-center flex-wrap">

                  {/* Input */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-2xl bg-white border border-zinc-200 shadow-sm flex items-center justify-center">
                      <svg viewBox="0 0 24 24" fill="none" stroke="#71717a" strokeWidth="1.5" className="w-7 h-7">
                        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                      </svg>
                    </div>
                    <span className="text-xs text-zinc-500 font-medium">Input frame</span>
                  </div>

                  <Arrow />

                  {/* Two streams */}
                  <div className="flex flex-col gap-3">
                    <StreamBox label="RGB Stream" sub="EfficientNet-B4" color="bg-brand/10 border-brand/30 text-brand" />
                    <StreamBox label="FFT Stream" sub="Frequency CNN" color="bg-violet-50 border-violet-200 text-violet-600" />
                  </div>

                  <Arrow />

                  {/* Fusion */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-20 h-16 rounded-2xl bg-white border-2 border-brand/30 shadow-sm flex flex-col items-center justify-center gap-0.5">
                      <span className="text-[10px] font-bold text-brand">FUSION</span>
                      <span className="text-[9px] text-zinc-400">FC + Dropout</span>
                    </div>
                    <span className="text-xs text-zinc-500 font-medium">Fusion layer</span>
                  </div>

                  <Arrow />

                  {/* Output */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="flex flex-col gap-1.5">
                      <div className="px-4 py-2 rounded-xl bg-red-50 border border-red-200 text-xs font-bold text-red-600 text-center">FAKE</div>
                      <div className="px-4 py-2 rounded-xl bg-emerald-50 border border-emerald-200 text-xs font-bold text-emerald-600 text-center">REAL</div>
                    </div>
                    <span className="text-xs text-zinc-500 font-medium">Verdict</span>
                  </div>
                </div>

                {/* Feature dims */}
                <div className="mt-6 pt-5 border-t border-zinc-200 grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-xs text-zinc-400">RGB feature dim</p>
                    <p className="text-sm font-semibold text-zinc-700 tabular-nums">1,792</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-400">FFT feature dim</p>
                    <p className="text-sm font-semibold text-zinc-700 tabular-nums">256</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-400">Input resolution</p>
                    <p className="text-sm font-semibold text-zinc-700 tabular-nums">224 × 224</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Metric grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.4 }}
          className="border-t border-b border-zinc-200"
        >
          <div className="grid grid-cols-2 md:grid-cols-4">
            {metrics.map((m, i) => (
              <div key={m.label} className={`py-10 ${i < metrics.length - 1 ? "md:border-r border-zinc-200" : ""} ${i > 0 ? "md:pl-8" : ""} ${i < metrics.length - 1 ? "md:pr-8" : ""}`}>
                <p className="text-xs text-zinc-400 font-medium mb-2">{m.label}</p>
                <p className="text-3xl font-bold text-zinc-900 tabular-nums">{m.value}</p>
                <p className="text-[11px] text-zinc-400 mt-1">{m.sub}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

function Arrow() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="#d4d4d8" strokeWidth="2" className="w-5 h-5 shrink-0">
      <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
    </svg>
  )
}

function StreamBox({ label, sub, color }: { label: string; sub: string; color: string }) {
  return (
    <div className={`flex flex-col items-center justify-center px-5 py-3 rounded-xl border ${color}`} style={{ minWidth: 140 }}>
      <span className="text-xs font-bold">{label}</span>
      <span className="text-[10px] opacity-70 mt-0.5">{sub}</span>
    </div>
  )
}
