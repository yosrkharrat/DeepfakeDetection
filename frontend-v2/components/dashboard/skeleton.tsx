import { cn } from "@/lib/utils";

function Bone({ className }: { className?: string }) {
  return <div className={cn("bg-zinc-100 rounded animate-pulse", className)} />;
}

export function ResultSkeleton() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="border border-zinc-200 rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
          <Bone className="h-3 w-12" />
          <Bone className="h-3 w-24" />
        </div>
        <div className="p-4 bg-zinc-50 flex items-center justify-center min-h-[200px]">
          <Bone className="w-40 h-40 rounded-lg" />
        </div>
      </div>
      <div className="border border-zinc-200 rounded-2xl overflow-hidden p-6 space-y-5">
        <div className="flex justify-center">
          <Bone className="w-32 h-32 rounded-full" />
        </div>
        <Bone className="h-12 w-full rounded-xl" />
        <div className="space-y-2">
          <Bone className="h-2 w-full rounded-full" />
          <Bone className="h-2 w-full rounded-full" />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Bone className="h-12 rounded-lg" />
          <Bone className="h-12 rounded-lg" />
        </div>
      </div>
    </div>
  );
}

export function TextResultSkeleton() {
  return (
    <div className="border border-zinc-200 rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-100">
        <Bone className="h-3 w-16" />
        <Bone className="h-3 w-12" />
      </div>
      <div className="p-5 space-y-4">
        <Bone className="h-16 w-full rounded-xl" />
        <div className="space-y-2">
          <Bone className="h-2 w-full rounded-full" />
          <Bone className="h-2 w-full rounded-full" />
        </div>
      </div>
    </div>
  );
}
