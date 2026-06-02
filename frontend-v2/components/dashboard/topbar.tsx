"use client";

import { usePathname } from "next/navigation";

const PAGE_TITLES: Record<string, { title: string; sub: string }> = {
  "/dashboard":             { title: "Visual Detection",   sub: "Analyze images & videos for deepfakes" },
  "/dashboard/text":        { title: "Fake News",          sub: "Classify text as real or fabricated" },
  "/dashboard/claims":      { title: "Claim Verification", sub: "Fact-check claims with AI research" },
  "/dashboard/performance": { title: "Performance",        sub: "Model metrics & evaluation results" },
  "/dashboard/health":      { title: "API Health",         sub: "Service status & endpoint availability" },
};

export function Topbar() {
  const pathname = usePathname();
  const page = PAGE_TITLES[pathname] ?? { title: "Reality Check", sub: "" };

  return (
    <div className="h-14 border-b border-zinc-200 bg-white flex items-center px-8 shrink-0">
      <div>
        <h1 className="text-sm font-semibold text-zinc-900">{page.title}</h1>
        {page.sub && <p className="text-[11px] text-zinc-400">{page.sub}</p>}
      </div>
      <div className="ml-auto flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          Models online
        </div>
        <div className="w-7 h-7 rounded-full bg-brand/10 flex items-center justify-center text-brand font-bold text-[10px]">
          DG
        </div>
      </div>
    </div>
  );
}
