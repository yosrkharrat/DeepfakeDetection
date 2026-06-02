import { VisualPanel } from "@/components/dashboard/visual-panel";
import { SessionStats } from "@/components/dashboard/session-stats";
import { SessionHistory } from "@/components/dashboard/session-history";

export default function VisualPage() {
  return (
    <>
      <VisualPanel />
      <SessionStats />
      <SessionHistory />
    </>
  );
}
