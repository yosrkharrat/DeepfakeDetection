"use client"

import { motion } from "framer-motion"

const steps = [
  {
    step: "01",
    title: "Upload or paste",
    desc: "Drop an image, video, news article, or factual claim into Reality Check.",
    visual: (
      <div className="bg-white border border-zinc-200 rounded-xl p-4">
        <div className="border-2 border-dashed border-zinc-200 rounded-lg py-5 flex flex-col items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="#a1a1aa" strokeWidth="1.5" className="w-7 h-7">
            <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
            <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
          </svg>
          <p className="text-xs text-zinc-400">Image · Video · Text · Claim</p>
        </div>
      </div>
    ),
  },
  {
    step: "02",
    title: "AI analysis",
    desc: "The dual-stream CNN (EfficientNet-B4 + FFT) analyses visual frames. NLP models process text. A research agent fact-checks claims live.",
    visual: (
      <div className="bg-white border border-zinc-200 rounded-xl p-4 space-y-2">
        {[
          { label: "RGB stream",   pct: 85, color: "bg-brand" },
          { label: "FFT stream",   pct: 70, color: "bg-violet-400" },
          { label: "Fusion layer", pct: 92, color: "bg-zinc-400" },
        ].map((s) => (
          <div key={s.label} className="flex items-center gap-3">
            <span className="text-[10px] text-zinc-400 w-20">{s.label}</span>
            <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${s.color}`}
                initial={{ width: 0 }}
                whileInView={{ width: `${s.pct}%` }}
                viewport={{ once: true }}
                transition={{ duration: 1, delay: 0.2 }}
              />
            </div>
          </div>
        ))}
        <div className="flex items-center justify-center gap-2 pt-1">
          <div className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" />
          <span className="text-[10px] text-zinc-400">Processing…</span>
        </div>
      </div>
    ),
  },
  {
    step: "03",
    title: "Instant verdict",
    desc: "Get a clear FAKE / REAL verdict with per-frame breakdown, probability scores, and for claims: sourced research report.",
    visual: (
      <div className="bg-white border border-zinc-200 rounded-xl p-4 space-y-2">
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-100 rounded-lg">
          <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" className="w-4 h-4 shrink-0">
            <path d="m10.29 3.86-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3l-8-14a2 2 0 0 0-3.42 0z"/>
          </svg>
          <span className="text-sm font-bold text-red-600">FAKE</span>
          <span className="ml-auto text-xs text-zinc-400">94.3% confidence</span>
        </div>
        <div className="flex gap-0.5 h-5 items-end">
          {[0.9,0.85,0.95,0.8,0.97,0.88,0.75,0.92,0.84,0.98].map((v, i) => (
            <div key={i} className="flex-1 rounded-sm bg-red-400/70" style={{ height: `${v * 100}%` }} />
          ))}
        </div>
        <p className="text-[10px] text-zinc-400 text-center">Frame-by-frame analysis</p>
      </div>
    ),
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 bg-zinc-50 border-y border-zinc-200">
      <div className="max-w-5xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl md:text-[48px] font-medium text-zinc-900"
            style={{ letterSpacing: "-0.032em", fontWeight: 538, lineHeight: 1.1 }}>
            How it works
          </h2>
          <p className="mt-4 text-zinc-500 max-w-md mx-auto">
            From upload to verdict in under two seconds — for images. Video and claims take a bit longer.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {steps.map((s, i) => (
            <motion.div key={s.step}
              initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.6, delay: i * 0.12 }}
              className="flex flex-col gap-4"
            >
              <span className="text-4xl font-bold text-zinc-100 tabular-nums">{s.step}</span>
              {s.visual}
              <div>
                <h3 className="text-base font-semibold text-zinc-800 mb-1">{s.title}</h3>
                <p className="text-sm text-zinc-500 leading-relaxed">{s.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
