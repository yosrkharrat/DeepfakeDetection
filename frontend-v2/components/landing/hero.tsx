"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { DashboardMockup } from "./dashboard-mockup"
import { Navbar } from "./navbar"

export function Hero() {
  const [yOffset, setYOffset] = useState(0)
  const [dashScale, setDashScale] = useState(0.42)

  useEffect(() => {
    const update = () => setDashScale(Math.min(1, (window.innerWidth - 40) / 1200))
    update()
    window.addEventListener("resize", update)
    return () => window.removeEventListener("resize", update)
  }, [])

  useEffect(() => {
    const onScroll = () => setYOffset(Math.min(window.scrollY / 300, 1) * -20)
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  const transform = { translateX: 2, scale: 1.2, rotateX: 47, rotateY: 31, rotateZ: 324 }

  return (
    <>
      <section className="relative min-h-screen overflow-hidden bg-white">
        <Navbar />

        {/* Cyan glow */}
        <div className="absolute pointer-events-none" style={{
          top: "50%", left: "50%",
          transform: "translate(-50%, -30%)",
          width: "1200px", height: "800px",
          background: "radial-gradient(ellipse at center, rgba(14,165,233,0.07) 0%, transparent 70%)",
        }} />

        <div className="relative pt-28 md:pt-28 flex flex-col">
          {/* Hero text */}
          <div className="relative z-20 w-full flex justify-center px-6 md:mt-16">
            <div className="w-full max-w-4xl">
              {/* Eyebrow */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="flex items-center gap-2 mb-6"
              >
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-brand/10 text-brand border border-brand/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" />
                  EfficientNet-B4 + FFT · 95.87% accuracy
                </span>
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.05 }}
                className="text-4xl md:text-5xl lg:text-[58px] font-medium text-zinc-900 leading-[1.08] text-balance text-center md:text-left"
                style={{ letterSpacing: "-0.03em", fontWeight: 540 }}
              >
                See through the fake.
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.12 }}
                className="mt-5 text-lg text-zinc-500 max-w-xl text-center md:text-left"
              >
                Reality Check combines a dual-stream CNN, NLP, and live fact-checking to detect deepfakes,
                fake news, and unverified claims — all in one place.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="mt-8 flex items-center gap-4 justify-center md:justify-start"
              >
                <Link href="/dashboard"
                  className="px-5 py-2.5 bg-brand text-white font-medium rounded-lg hover:bg-brand/90 transition-colors text-sm">
                  Try the demo
                </Link>
                <a href="#how-it-works"
                  className="text-sm text-zinc-500 hover:text-zinc-800 transition-colors font-medium">
                  How it works →
                </a>
              </motion.div>

              {/* KPI chips */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.7, delay: 0.35 }}
                className="mt-10 flex flex-wrap gap-3 justify-center md:justify-start"
              >
                {[
                  { label: "Accuracy",  val: "95.87%" },
                  { label: "AUC-ROC",   val: "0.9951" },
                  { label: "Precision", val: "99.04%" },
                  { label: "F1",        val: "96.90%" },
                ].map((k) => (
                  <div key={k.label} className="flex items-center gap-2 px-3 py-1.5 bg-zinc-50 border border-zinc-200 rounded-lg">
                    <span className="text-xs text-zinc-400">{k.label}</span>
                    <span className="text-xs font-bold text-zinc-800 tabular-nums">{k.val}</span>
                  </div>
                ))}
              </motion.div>
            </div>
          </div>

          {/* Mobile mockup — flat */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="md:hidden mx-4 mt-10 mb-16 rounded-2xl overflow-hidden"
            style={{
              width: "calc(100vw - 32px)",
              height: `${Math.round(600 * dashScale)}px`,
              border: "1px solid rgba(14,165,233,0.25)",
              boxShadow: "0 0 0 4px rgba(14,165,233,0.06), 0 20px 60px -12px rgba(14,165,233,0.2)",
            }}
          >
            <div style={{ width: "1200px", height: "600px", transform: `scale(${dashScale})`, transformOrigin: "top left" }}>
              <DashboardMockup />
            </div>
          </motion.div>

          {/* Desktop 3D stage */}
          <div className="hidden md:block relative mt-16" style={{
            width: "100vw", marginLeft: "-50vw", marginRight: "-50vw",
            position: "relative", left: "50%", right: "50%",
            height: "700px", marginTop: "-60px",
          }}>
            <div className="absolute bottom-0 left-0 right-0 h-72 z-10 pointer-events-none"
              style={{ background: "linear-gradient(to top, #ffffff 20%, transparent 100%)" }} />

            <div style={{
              transform: `translateY(${yOffset}px)`,
              transition: "transform 0.1s ease-out",
              perspective: "4000px",
              perspectiveOrigin: "100% 0",
              width: "100%", height: "100%",
              transformStyle: "preserve-3d",
              position: "relative",
            }}>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 1, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  backgroundColor: "#ffffff",
                  transformOrigin: "0 0",
                  backfaceVisibility: "hidden",
                  border: "1px solid #e4e4e7",
                  borderRadius: "10px",
                  width: "1600px",
                  height: "900px",
                  margin: "280px auto auto",
                  position: "absolute",
                  top: 0, bottom: 0, left: 0, right: 0,
                  transform: `translate(${transform.translateX}%) scale(${transform.scale}) rotateX(${transform.rotateX}deg) rotateY(${transform.rotateY}deg) rotate(${transform.rotateZ}deg)`,
                  transformStyle: "preserve-3d",
                  overflow: "hidden",
                  boxShadow: "0 25px 50px -12px rgba(0,0,0,0.08)",
                }}
              >
                <DashboardMockup />
              </motion.div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
