import { Shield } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-zinc-200 py-14 px-6 bg-zinc-50">
      <div className="max-w-5xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-2">
            <div className="flex items-center gap-2.5 mb-3">
              <div className="w-6 h-6 rounded-md bg-brand flex items-center justify-center shrink-0">
                <Shield className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="text-zinc-900 font-semibold text-sm">Reality Check</span>
            </div>
            <p className="text-zinc-400 text-xs leading-relaxed max-w-xs">
              AI deepfake & misinformation detection powered by
              dual-stream CNN fusion trained on FaceForensics++ C23.
            </p>
          </div>

          {/* Detection */}
          <div>
            <h3 className="text-zinc-900 font-medium text-sm mb-4">Detection</h3>
            <ul className="space-y-3">
              {[
                { label: "Visual (image & video)", href: "/dashboard" },
                { label: "Fake news (text)",       href: "/dashboard/text" },
                { label: "Claim verification",     href: "/dashboard/claims" },
                { label: "API health",             href: "/dashboard/health" },
              ].map((l) => (
                <li key={l.label}>
                  <a href={l.href} className="text-zinc-500 hover:text-zinc-700 text-sm transition-colors">{l.label}</a>
                </li>
              ))}
            </ul>
          </div>

          {/* Model */}
          <div>
            <h3 className="text-zinc-900 font-medium text-sm mb-4">Model</h3>
            <ul className="space-y-3">
              {[
                "EfficientNet-B4 RGB",
                "FFT CNN stream",
                "Fusion layer",
                "FaceForensics++ C23",
              ].map((l) => (
                <li key={l}>
                  <span className="text-zinc-500 text-sm">{l}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-zinc-200 flex flex-col md:flex-row items-center justify-between gap-2">
          <p className="text-zinc-400 text-xs">© 2026 Reality Check.</p>
          <p className="text-zinc-400 text-xs">
            Accuracy 95.87% · AUC-ROC 0.9951 · Trained on 32,898 face crops
          </p>
        </div>
      </div>
    </footer>
  )
}
