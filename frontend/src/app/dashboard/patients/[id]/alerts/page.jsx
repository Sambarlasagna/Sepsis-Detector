"use client";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle } from "lucide-react";

const vitalsData = [
  { time: "12:00", hr: 80, bp: 120 },
  { time: "12:10", hr: 85, bp: 118 },
  { time: "12:20", hr: 90, bp: 115 },
  { time: "12:30", hr: 95, bp: 110 },
  { time: "12:40", hr: 102, bp: 108 },
];

export default function AlertsPage() {
  const [riskScore, setRiskScore] = useState(78);

  return (
    <>
      <header className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-semibold text-black">Alerts</h2>
      </header>

      {/* Cards */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow flex items-center gap-3">
          <AlertTriangle className="text-red-500" />
          <p className="font-semibold text-red-500">2 Critical Alerts</p>
        </div>
      </div>
    </>
  );
}