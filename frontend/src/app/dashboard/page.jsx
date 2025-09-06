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

export default function Dashboard() {
  const [riskScore, setRiskScore] = useState(78); // Sample risk score

  return (
    <main className="flex min-h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-md p-6 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-green-700">SepsisSense</h1>
        <nav className="flex flex-col gap-4">
          <a href="#" className="text-gray-700 hover:text-green-700">Vitals</a>
          <a href="#" className="text-gray-700 hover:text-green-700">Patients</a>
          <a href="#" className="text-gray-700 hover:text-green-700">Alerts</a>
          <a href="#" className="text-gray-700 hover:text-green-700">Settings</a>
        </nav>
      </aside>

      {/* Main Content */}
      <section className="flex-1 p-8">
        {/* Header */}
        <header className="flex justify-between items-center mb-8">
          <h2 className="text-3xl font-semibold text-black">ICU Vitals Dashboard</h2>
          <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
            Refresh Data
          </button>
        </header>

        {/* Cards */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-gray-500 text-sm">Sepsis Risk Score</h3>
            <p className="text-3xl font-bold text-green-700">{riskScore}%</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-gray-500 text-sm">Vital Signs Monitored</h3>
            <p className="text-3xl font-bold text-black">HR, BP, SpO₂</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow flex items-center gap-3">
            <AlertTriangle className="text-red-500" />
            <p className="font-semibold text-red-500">2 Critical Alerts</p>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4 text-black">Heart Rate Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={vitalsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="hr" stroke="#16a34a" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </main>
  );
}
