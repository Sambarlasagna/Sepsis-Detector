"use client";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [ws, setWs] = useState(null);
  const [riskScore, setRiskScore] = useState(78);

  // Extract patientId from the current URL
  const pathname = usePathname();
  const pathParts = pathname.split("/");
  const patientIdIndex = pathParts.indexOf("patients") + 1;
  const patientId = pathParts[patientIdIndex];

  // Format alert time based on predicted hours
  const formatAlertTime = (hours) => {
    const futureDate = new Date();
    futureDate.setHours(futureDate.getHours() + hours);
    return futureDate.toLocaleTimeString([], {
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    });
  };

  useEffect(() => {
    if (!patientId) return;

    // Connect to patient-specific WebSocket
    const websocket = new WebSocket(`ws://127.0.0.1:8000/ws/alerts/${patientId}`);

    websocket.onopen = () => {
      console.log(`Connected to WebSocket for ${patientId}`);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setAlerts(data.alerts || []); // Update state with patient-specific alerts
    };

    websocket.onclose = () => {
      console.log(`WebSocket disconnected for ${patientId}`);
    };

    setWs(websocket);

    return () => websocket.close(); // Cleanup on unmount
  }, [patientId]);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4 text-black">
        Alerts for {patientId ? patientId.replace("patient", "Patient ") : ""}
      </h1>

      {/* Risk Score Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-gray-500 text-sm">Sepsis Risk Score</h3>
          <p className="text-3xl font-bold text-green-700">{riskScore}%</p>
        </div>
      </div>

      {/* Alerts List */}
      {alerts.length > 0 ? (
        <ul className="space-y-2">
          {alerts.map((hours, idx) => (
            <li
              key={idx}
              className="p-4 bg-white rounded shadow flex items-center gap-2"
            >
              <span className="text-red-500 font-semibold">⚠️</span>
              <span className="text-red-500 font-semibold">
                May detect sepsis at {formatAlertTime(hours)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-gray-500 text-sm">No active alerts for this patient.</p>
      )}
    </div>
  );
}