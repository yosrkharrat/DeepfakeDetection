"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Shield, Video, FileText, CheckSquare, BarChart2, Activity, ChevronRight, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-[220px] h-full bg-zinc-50 border-r border-zinc-200 flex flex-col shrink-0">

      {/* Logo */}
      <div className="p-3 border-b border-zinc-200">
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <div className="w-6 h-6 rounded-md bg-brand flex items-center justify-center shrink-0">
            <Shield className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-zinc-900 font-semibold text-sm">DeepGuard</span>
        </div>
      </div>

      {/* Search */}
      <div className="p-3">
        <div className="flex items-center gap-2 px-2.5 py-1.5 bg-white border border-zinc-200 rounded-md text-zinc-400 text-xs">
          <Search className="w-3.5 h-3.5" />
          <span>Search...</span>
          <span className="ml-auto text-[10px] bg-zinc-100 px-1.5 py-0.5 rounded">⌘K</span>
        </div>
      </div>

      {/* Detection */}
      <div className="px-3 space-y-0.5">
        <div className="px-2 py-1 text-[10px] text-zinc-400 font-medium uppercase tracking-wider">Detection</div>
        <NavItem icon={Video}       label="Visual"       href="/dashboard"        active={pathname === "/dashboard"} />
        <NavItem icon={FileText}    label="Fake News"    href="/dashboard/text"   active={pathname === "/dashboard/text"} />
        <NavItem icon={CheckSquare} label="Claim Verify" href="/dashboard/claims" active={pathname === "/dashboard/claims"} />
      </div>

      {/* Model */}
      <div className="mt-5 px-3">
        <div className="px-2 py-1 text-[10px] text-zinc-400 font-medium uppercase tracking-wider">Model</div>
        <div className="space-y-0.5 mt-1">
          <NavItem icon={BarChart2} label="Performance" href="/dashboard/performance" active={pathname === "/dashboard/performance"} hasSubmenu />
          <NavItem icon={Activity}  label="API Health"  href="/dashboard/health"      active={pathname === "/dashboard/health"} hasSubmenu />
        </div>
      </div>

      <div className="flex-1" />

      {/* Footer card */}
      <div className="p-3 border-t border-zinc-200">
        <div className="px-2 py-2 rounded-md bg-white border border-zinc-200">
          <p className="text-[10px] text-zinc-400 mb-1">Model checkpoint</p>
          <p className="text-xs font-medium text-zinc-700 font-mono truncate">fusion_best.pt</p>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span className="text-[10px] text-zinc-500">Ready · AUC 0.9951</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

function NavItem({ icon: Icon, label, href, active, hasSubmenu }: {
  icon: React.ElementType; label: string; href: string; active: boolean; hasSubmenu?: boolean;
}) {
  return (
    <Link href={href} className={cn(
      "flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors",
      active
        ? "bg-brand/10 text-brand border-l-2 border-brand -ml-0.5 pl-[7px]"
        : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
    )}>
      <Icon className="w-4 h-4 shrink-0" />
      <span className="flex-1">{label}</span>
      {hasSubmenu && <ChevronRight className="w-3 h-3 text-zinc-400" />}
    </Link>
  );
}
