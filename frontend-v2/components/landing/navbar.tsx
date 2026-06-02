"use client"

import { useState, type MouseEvent } from "react"
import Link from "next/link"
import { Shield, Menu, X } from "lucide-react"

const navLinks = [
  { label: "Features",     href: "#features" },
  { label: "How it works", href: "#how-it-works" },
  { label: "Model",        href: "#model" },
]

export function Navbar() {
  const [open, setOpen] = useState(false)

  const handleAnchor = (e: MouseEvent<HTMLAnchorElement>, href: string) => {
    if (!href.startsWith("#")) return
    e.preventDefault()
    const target = document.querySelector(href)
    if (!target) return
    const nav = document.querySelector("nav")
    const offset = nav instanceof HTMLElement ? nav.offsetHeight : 64
    window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - offset - 12, behavior: "smooth" })
    setOpen(false)
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-200 bg-white/95 backdrop-blur-sm">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-brand flex items-center justify-center shrink-0">
            <Shield className="w-4 h-4 text-white" />
          </div>
          <span className="text-zinc-900 font-semibold text-sm">Reality Check</span>
        </div>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((l) => (
            <a key={l.label} href={l.href} onClick={(e) => handleAnchor(e, l.href)}
              className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors">
              {l.label}
            </a>
          ))}
        </div>

        {/* Desktop CTA */}
        <div className="hidden md:flex items-center gap-3">
          <Link href="/dashboard" className="text-sm text-zinc-600 hover:text-zinc-900 font-medium transition-colors">
            Open app
          </Link>
          <Link href="/dashboard"
            className="text-sm text-white bg-brand hover:bg-brand/90 px-3.5 py-1.5 rounded-lg transition-colors font-medium">
            Try Reality Check
          </Link>
        </div>

        {/* Mobile toggle */}
        <button className="md:hidden p-2 rounded-md text-zinc-500 hover:bg-zinc-100 transition-colors"
          onClick={() => setOpen((v) => !v)}>
          {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {open && (
        <div className="md:hidden border-t border-zinc-200 bg-white px-6 py-4 flex flex-col gap-4">
          {navLinks.map((l) => (
            <a key={l.label} href={l.href} onClick={(e) => handleAnchor(e, l.href)}
              className="text-sm text-zinc-600 hover:text-zinc-900 transition-colors py-1">
              {l.label}
            </a>
          ))}
          <div className="pt-2 border-t border-zinc-100">
            <Link href="/dashboard" onClick={() => setOpen(false)}
              className="block w-fit text-sm text-white bg-brand hover:bg-brand/90 px-4 py-2 rounded-lg font-medium transition-colors">
              Try Reality Check
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}
