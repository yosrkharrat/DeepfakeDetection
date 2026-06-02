const metrics = [
  { label: "Accuracy",          value: "95.87%", description: "Overall correct predictions on test set" },
  { label: "Precision",         value: "99.04%", description: "Of predicted fakes, how many were actually fake" },
  { label: "Recall",            value: "94.85%", description: "Of actual fakes, how many were caught" },
  { label: "F1 Score",          value: "96.90%", description: "Harmonic mean of precision and recall" },
  { label: "AUC-ROC",           value: "0.9951", description: "Area under receiver operating characteristic curve" },
  { label: "False Positive Rate", value: "1.97%", description: "Real media incorrectly flagged as fake" },
];

const confusionData = [
  { label: "True Positives",  value: 3188, color: "bg-emerald-400", pct: 64.5 },
  { label: "True Negatives",  value: 1545, color: "bg-emerald-200", pct: 31.3 },
  { label: "False Positives", value: 31,   color: "bg-red-300",     pct: 0.6 },
  { label: "False Negatives", value: 173,  color: "bg-red-200",     pct: 3.5 },
];

export default function PerformancePage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-zinc-900">Model Performance</h2>
        <p className="text-sm text-zinc-400 mt-1">
          Evaluated on FaceForensics++ C23 test split · 4,937 samples · checkpoint <span className="font-mono">fusion_best.pt</span>
        </p>
      </div>

      {/* Metric grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="bg-white border border-zinc-200 rounded-2xl p-5">
            <p className="text-xs text-zinc-400 font-medium mb-2">{m.label}</p>
            <p className="text-3xl font-bold text-zinc-900 tabular-nums">{m.value}</p>
            <p className="text-[11px] text-zinc-400 mt-1 leading-relaxed">{m.description}</p>
          </div>
        ))}
      </div>

      {/* Confusion matrix summary */}
      <div className="bg-white border border-zinc-200 rounded-2xl p-6">
        <h3 className="text-sm font-semibold text-zinc-800 mb-5">Confusion Matrix (test set)</h3>
        <div className="space-y-3">
          {confusionData.map((row) => (
            <div key={row.label} className="flex items-center gap-4">
              <span className="text-xs text-zinc-500 w-36 shrink-0">{row.label}</span>
              <div className="flex-1 h-5 bg-zinc-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${row.color}`}
                  style={{ width: `${row.pct}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-zinc-700 tabular-nums w-14 text-right">
                {row.value.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-zinc-400 mt-4">
          Threshold: 0.5 · Total: 4,937 · Train: EfficientNet-B4 + FFT CNN dual-stream fusion
        </p>
      </div>

      {/* Dataset info */}
      <div className="bg-white border border-zinc-200 rounded-2xl p-6">
        <h3 className="text-sm font-semibold text-zinc-800 mb-4">Dataset</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-0">
          {[
            { label: "Total crops",  value: "32,898" },
            { label: "Real",         value: "10,499" },
            { label: "Fake",         value: "22,399" },
            { label: "Source videos", value: "7,000" },
          ].map((s, i) => (
            <div key={s.label} className={`flex flex-col ${i < 3 ? "pr-6 border-r border-zinc-200" : ""} ${i > 0 ? "pl-6" : ""}`}>
              <span className="text-xs text-zinc-400">{s.label}</span>
              <span className="text-xl font-bold text-zinc-900 tabular-nums mt-1">{s.value}</span>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-zinc-400 mt-4">
          FaceForensics++ C23 · Deepfakes · Face2Face · FaceShifter · FaceSwap · NeuralTextures
        </p>
      </div>
    </div>
  );
}
