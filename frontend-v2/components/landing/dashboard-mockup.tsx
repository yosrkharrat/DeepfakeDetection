// Static miniature dashboard mockup — used in the hero 3D stage

export function DashboardMockup() {
  return (
    <div className="w-full h-full flex bg-white" style={{ fontFamily: "-apple-system, sans-serif" }}>

      {/* Sidebar */}
      <div className="w-[180px] h-full bg-zinc-50 border-r border-zinc-200 flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-3 border-b border-zinc-200">
          <div className="flex items-center gap-2 px-2 py-1.5">
            <div className="w-5 h-5 rounded-md bg-[#0ea5e9] flex items-center justify-center">
              <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" className="w-3 h-3">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <span className="text-zinc-900 font-semibold text-xs">DeepGuard</span>
          </div>
        </div>

        {/* Search */}
        <div className="p-2.5">
          <div className="flex items-center gap-1.5 px-2 py-1.5 bg-white border border-zinc-200 rounded text-zinc-400 text-[10px]">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            Search...
          </div>
        </div>

        {/* Nav */}
        <div className="px-2.5 space-y-0.5">
          <div className="text-[9px] text-zinc-400 font-medium uppercase tracking-wider px-2 py-1">Detection</div>
          <div className="flex items-center gap-1.5 px-2 py-1.5 rounded text-[10px] bg-[#0ea5e9]/10 text-[#0ea5e9] border-l-2 border-[#0ea5e9] -ml-0.5 pl-[7px]">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3 shrink-0">
              <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            Visual
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1.5 rounded text-[10px] text-zinc-500">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3 shrink-0">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
            </svg>
            Fake News
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1.5 rounded text-[10px] text-zinc-500">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3 shrink-0">
              <polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
            </svg>
            Claim Verify
          </div>
        </div>

        <div className="flex-1" />

        {/* Status pill */}
        <div className="p-2.5">
          <div className="px-2 py-2 rounded bg-white border border-zinc-200">
            <p className="text-[9px] text-zinc-400 mb-0.5">Model</p>
            <p className="text-[10px] font-medium text-zinc-700 font-mono">fusion_best.pt</p>
            <div className="flex items-center gap-1 mt-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span className="text-[9px] text-zinc-500">AUC 0.9951</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Topbar */}
        <div className="h-12 border-b border-zinc-200 bg-white flex items-center px-6 shrink-0">
          <div>
            <p className="text-xs font-semibold text-zinc-900">Visual Detection</p>
            <p className="text-[9px] text-zinc-400">Analyze images & videos for deepfakes</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="flex items-center gap-1 text-[9px] text-zinc-500">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Models online
            </div>
            <div className="w-6 h-6 rounded-full bg-[#0ea5e9]/10 flex items-center justify-center text-[#0ea5e9] font-bold text-[9px]">
              DG
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 p-6 overflow-hidden">
          {/* Stats row */}
          <div className="grid grid-cols-4 gap-0 mb-6">
            {[
              { label: "Accuracy",  val: "95.87%" },
              { label: "Precision", val: "99.04%" },
              { label: "F1 Score",  val: "96.90%" },
              { label: "AUC-ROC",   val: "0.9951" },
            ].map((s, i) => (
              <div key={s.label} className={`flex flex-col ${i < 3 ? "pr-4 border-r border-zinc-200" : ""} ${i > 0 ? "pl-4" : ""}`}>
                <span className="text-[9px] text-zinc-400">{s.label}</span>
                <span className="text-base font-bold text-zinc-900 tabular-nums">{s.val}</span>
              </div>
            ))}
          </div>

          {/* Detection area */}
          <div className="grid grid-cols-2 gap-4">
            {/* Upload card */}
            <div className="border border-zinc-200 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-100">
                <span className="text-[10px] font-semibold text-zinc-800">Input</span>
                <span className="text-[9px] text-zinc-400">face_video.mp4</span>
              </div>
              <div className="p-3 bg-zinc-50 flex items-center justify-center h-36">
                {/* Fake face placeholder */}
                <div className="relative">
                  <div className="w-24 h-28 rounded-lg bg-zinc-200 flex items-center justify-center overflow-hidden">
                    {/* stylised face silhouette */}
                    <svg viewBox="0 0 60 72" className="w-16 h-20 text-zinc-400" fill="currentColor">
                      <ellipse cx="30" cy="28" rx="18" ry="22" />
                      <ellipse cx="30" cy="68" rx="26" ry="18" />
                    </svg>
                  </div>
                  {/* Bounding box */}
                  <div className="absolute inset-1 border-2 border-red-400 rounded" />
                  {/* FAKE badge */}
                  <div className="absolute -top-2 -right-2 bg-red-500 text-white text-[8px] font-bold px-1.5 py-0.5 rounded">
                    FAKE
                  </div>
                </div>
              </div>
            </div>

            {/* Result card */}
            <div className="border border-zinc-200 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-100">
                <span className="text-[10px] font-semibold text-zinc-800">Result</span>
                <span className="text-[9px] text-zinc-400">1.24s</span>
              </div>
              <div className="p-3 space-y-3">
                {/* Verdict */}
                <div className="flex items-center gap-2 p-2.5 bg-red-50 border border-red-100 rounded-lg">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" className="w-4 h-4 shrink-0">
                    <path d="m10.29 3.86-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3l-8-14a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                  </svg>
                  <div>
                    <p className="text-sm font-bold text-red-600">FAKE</p>
                    <p className="text-[9px] text-zinc-500">94.3% confidence</p>
                  </div>
                </div>

                {/* Prob bar */}
                <div>
                  <div className="flex justify-between text-[9px] text-zinc-400 mb-1">
                    <span>Real</span><span>Fake</span>
                  </div>
                  <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-emerald-400 to-red-400 rounded-full" style={{ width: "94%" }} />
                  </div>
                  <div className="flex justify-between text-[9px] text-zinc-400 mt-0.5">
                    <span>5.7%</span><span>94.3%</span>
                  </div>
                </div>

                {/* Frame timeline */}
                <div>
                  <p className="text-[9px] text-zinc-400 mb-1">Frame timeline</p>
                  <div className="flex gap-0.5 h-6 items-end">
                    {[0.91,0.88,0.95,0.82,0.97,0.89,0.76,0.93,0.85,0.98,0.72,0.90,0.94,0.87,0.96].map((v, i) => (
                      <div key={i} className="flex-1 rounded-sm bg-red-400" style={{ height: `${v * 100}%` }} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
