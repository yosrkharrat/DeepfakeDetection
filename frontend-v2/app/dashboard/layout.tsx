import { Sidebar } from "@/components/dashboard/sidebar";
import { Topbar } from "@/components/dashboard/topbar";
import { Toaster } from "sonner";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full overflow-hidden border-l border-zinc-200">
        <Topbar />
        <main className="flex-1 overflow-y-auto p-8">{children}</main>
      </div>
      <Toaster position="bottom-right" richColors closeButton />
    </div>
  );
}
