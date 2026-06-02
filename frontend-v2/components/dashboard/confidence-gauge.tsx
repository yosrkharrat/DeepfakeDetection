"use client";

import { useEffect, useState } from "react";

interface Props {
  value: number; // 0–1
  isFake: boolean;
}

export function ConfidenceGauge({ value, isFake }: Props) {
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimated(value));
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const R = 52;
  const cx = 64;
  const cy = 64;
  const circumference = Math.PI * R;
  const offset = circumference * (1 - animated);
  const color = isFake ? "#ef4444" : "#10b981";
  const pct = Math.round(animated * 100);

  return (
    <div className="flex flex-col items-center">
      <svg width="128" height="72" viewBox="0 0 128 72">
        <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none" stroke="#e4e4e7" strokeWidth="10" strokeLinecap="round" />
        <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize="22" fontWeight="700"
          fill={color} style={{ fontFamily: "monospace" }}>{pct}%</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fontSize="10" fill="#a1a1aa">confidence</text>
      </svg>
      <span className={`text-lg font-bold mt-1 ${isFake ? "text-red-500" : "text-emerald-500"}`}>
        {isFake ? "FAKE" : "REAL"}
      </span>
    </div>
  );
}
