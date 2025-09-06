"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle } from "lucide-react";
import { useParams } from "next/navigation";

const vitalsData = [
  { time: "12:00", hr: 80, bp: 120, spo2: 98 },
  { time: "12:10", hr: 85, bp: 118, spo2: 97 },
  { time: "12:20", hr: 90, bp: 115, spo2: 96 },
  { time: "12:30", hr: 95, bp: 110, spo2: 97 },
  { time: "12:40", hr: 102, bp: 108, spo2: 98 },
];

export default function VitalsPage() {
  const [riskScore, setRiskScore] = useState(78);
  const [alertsCount, setAlertsCount] = useState(0);

  // Get current path to extract patient ID
  const { id } = useParams();

  useEffect(() => {
  if (!id) return; // Wait until we have the patient ID

  const websocket = new WebSocket(`ws://127.0.0.1:8000/ws/alerts/${id}`);

  websocket.onopen = () => {
    console.log(`Vitals page connected to WebSocket for ${id}`);
  };

  websocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    setAlertsCount(data.alerts.length);
  };

  websocket.onclose = () => {
    console.log(`WebSocket for ${id} disconnected`);
  };

  return () => websocket.close();
}, [id]);  // <--- IMPORTANT: Depend on id

  return (
    <>
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

        {/* Clickable Alerts Card */}
        <Link href={`/dashboard/patients/${id}/alerts`}>
  <div className="cursor-pointer bg-white p-6 rounded-lg shadow flex items-center gap-3 hover:bg-gray-100 transition">
    <AlertTriangle className="text-red-500" />
    <p className="font-semibold text-red-500">{alertsCount} Critical Alerts</p>
  </div>
</Link>
      </div>

      {/* Heart Rate Chart */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4 text-black">Heart Rate</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={vitalsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="hr" stroke="#16a34a" strokeWidth={3} name="Heart Rate" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Blood Pressure Chart */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4 text-black">Blood Pressure</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={vitalsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="bp" stroke="#ef4444" strokeWidth={3} name="Blood Pressure" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* SpO2 Chart */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4 text-black">SpO₂</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={vitalsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="spo2" stroke="#3b82f6" strokeWidth={3} name="SpO₂" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
