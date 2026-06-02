"use client"

import { motion } from "framer-motion"

/* ── Inline illustrations ── */

const VisualIllustration = () => (
  <div className="w-full px-5 pt-4 space-y-2">
    {/* Upload zone miniature */}
    <div className="border border-dashed border-zinc-200 rounded-xl py-4 flex flex-col items-center gap-1.5 bg-zinc-50">
      <svg viewBox="0 0 24 24" fill="none" stroke="#a1a1aa" strokeWidth="1.5" className="w-6 h-6">
        <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
        <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
      </svg>
      <p className="text-[10px] text-zinc-400">Drop image or video</p>
    </div>
    {/* Result row */}
    <div className="flex items-center gap-3 bg-white border border-zinc-200 rounded-xl px-4 py-3 shadow-sm">
      <div className="relative w-10 h-12 bg-zinc-100 rounded-lg overflow-hidden shrink-0">
        <svg viewBox="0 0 40 48" className="w-full h-full text-zinc-300" fill="currentColor">
          <ellipse cx="20" cy="18" rx="12" ry="14"/>
          <ellipse cx="20" cy="46" rx="18" ry="12"/>
        </svg>
        <div className="absolute inset-0.5 border-2 border-red-400 rounded" />
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-medium text-zinc-700">face_video.mp4</span>
          <span className="text-[10px] font-bold text-red-500 bg-red-50 px-1.5 py-0.5 rounded">FAKE</span>
        </div>
        <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
          <div className="h-full w-[93%] bg-gradient-to-r from-emerald-300 to-red-400 rounded-full" />
        </div>
        <div className="flex justify-between text-[9px] text-zinc-400 mt-0.5">
          <span>Real 7%</span><span>Fake 93%</span>
        </div>
      </div>
    </div>
    {/* Frame timeline */}
    <div className="flex gap-0.5 h-5 items-end px-1">
      {[0.9,0.85,0.95,0.8,0.97,0.88,0.75,0.92,0.84,0.98,0.71,0.89,0.93,0.86,0.96,0.82,0.91,0.87].map((v, i) => (
        <div key={i} className="flex-1 rounded-sm bg-red-400/70" style={{ height: `${v * 100}%` }} />
      ))}
    </div>
  </div>
)

const TextIllustration = () => (
  <div className="w-full px-5 pt-4 space-y-2">
    <div className="bg-white border border-zinc-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-zinc-100">
        <span className="text-[10px] font-semibold text-zinc-700">Fake News Detector</span>
      </div>
      <div className="p-3">
        <div className="bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-2.5 text-[10px] text-zinc-500 italic leading-relaxed mb-3">
          "Scientists confirm that drinking lemon water every morning cures all known diseases according to a 2024 Harvard study..."
        </div>
        <div className="flex items-center gap-2 p-2.5 bg-red-50 border border-red-100 rounded-lg mb-3">
          <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" className="w-3.5 h-3.5 shrink-0">
            <path d="m10.29 3.86-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3l-8-14a2 2 0 0 0-3.42 0z"/>
          </svg>
          <span className="text-[10px] font-bold text-red-600">FAKE NEWS</span>
          <span className="ml-auto text-[9px] text-zinc-400">97.2%</span>
        </div>
        {[
          { label: "Real", w: "11%",  color: "bg-emerald-400" },
          { label: "Fake", w: "89%",  color: "bg-red-400" },
        ].map((b) => (
          <div key={b.label} className="flex items-center gap-2 mb-1.5">
            <span className="text-[9px] text-zinc-400 w-6">{b.label}</span>
            <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${b.color}`} style={{ width: b.w }} />
            </div>
            <span className="text-[9px] text-zinc-500 w-7 text-right">{b.w}</span>
          </div>
        ))}
      </div>
    </div>
  </div>
)

const ClaimsIllustration = () => (
  <div className="w-full px-5 pt-4 space-y-2">
    <div className="bg-white border border-zinc-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-zinc-100">
        <span className="text-[10px] font-semibold text-zinc-700">Claim Verification</span>
      </div>
      <div className="p-3">
        <div className="bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-2.5 text-[10px] text-zinc-500 italic mb-3">
          "The Eiffel Tower is taller than the Empire State Building."
        </div>
        <div className="flex items-center gap-2 p-2.5 bg-red-50 border border-red-100 rounded-lg mb-3">
          <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" className="w-3.5 h-3.5 shrink-0">
            <path d="m10.29 3.86-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3l-8-14a2 2 0 0 0-3.42 0z"/>
          </svg>
          <span className="text-[10px] font-bold text-red-600">REFUTED</span>
          <span className="ml-auto text-[9px] text-zinc-400">88.4%</span>
        </div>
        {[
          { label: "Supported", w: "5%",  color: "bg-emerald-400" },
          { label: "Refuted",   w: "88%", color: "bg-red-400" },
          { label: "Unclear",   w: "7%",  color: "bg-zinc-300" },
        ].map((b) => (
          <div key={b.label} className="flex items-center gap-2 mb-1.5">
            <span className="text-[9px] text-zinc-400 w-12">{b.label}</span>
            <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${b.color}`} style={{ width: b.w }} />
            </div>
            <span className="text-[9px] text-zinc-500 w-7 text-right">{b.w}</span>
          </div>
        ))}
      </div>
    </div>
  </div>
)

const cards = [
  { title: "Deepfake detection — images & video",    illustration: <VisualIllustration /> },
  { title: "Fake news classified in milliseconds",   illustration: <TextIllustration /> },
  { title: "Claims fact-checked with live research", illustration: <ClaimsIllustration /> },
]

export function FeatureCards() {
  return (
    <div id="features" className="relative pt-20 pb-40 bg-zinc-50 border-y border-zinc-200">
      <div className="max-w-5xl mx-auto px-6">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-8 mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-3xl sm:text-4xl md:text-5xl lg:text-[54px] text-zinc-900 max-w-md"
            style={{ letterSpacing: "-0.032em", fontWeight: 538, lineHeight: 1.1 }}
          >
            Three detectors. One platform.
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="max-w-sm text-zinc-500 leading-relaxed mt-2"
          >
            DeepGuard handles visual media, written content, and factual claims — each backed by
            a dedicated AI model trained on real-world data.
          </motion.p>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {cards.map((card, i) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.15 + i * 0.1 }}
              className="bg-white border border-zinc-200 hover:border-brand/30 hover:shadow-sm transition-all overflow-hidden flex flex-col md:h-[440px]"
              style={{ borderRadius: "28px", isolation: "isolate" }}
            >
              <div className="px-6 pt-6 pb-2">
                <h3 className="text-zinc-900 font-medium text-base leading-snug">{card.title}</h3>
              </div>
              <div className="flex-1 relative overflow-hidden">
                <div className="md:absolute md:inset-0">{card.illustration}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}
