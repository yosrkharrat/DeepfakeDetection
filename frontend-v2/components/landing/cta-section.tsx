import Link from "next/link"
import { Shield } from "lucide-react"

export function CTASection() {
  return (
    <section className="py-24 px-6 bg-white border-t border-zinc-200">
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-10">
        <div>
          <h2 className="text-3xl md:text-4xl lg:text-[42px] font-medium text-zinc-900"
            style={{ letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Ready to detect the truth?
          </h2>
          <p className="mt-3 text-zinc-500 max-w-sm">
            Open the dashboard and start analysing images, videos, articles, and claims — no setup needed.
          </p>
        </div>
        <div className="flex flex-col sm:flex-row items-center gap-3 shrink-0">
          <Link href="/dashboard"
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-2.5 bg-brand text-white font-medium rounded-lg hover:bg-brand/90 transition-colors text-sm">
            <Shield className="w-4 h-4" />
            Open Reality Check
          </Link>
        </div>
      </div>
    </section>
  )
}
