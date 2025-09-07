"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";
import useAlerts from "../../../../../../hooks/useAlerts";

export default function AlertsPage() {
  // Extract patientId from URL
  const pathname = usePathname();
  const pathParts = pathname.split("/");
  const patientIdIndex = pathParts.indexOf("patients") + 1;
  const patientId = pathParts[patientIdIndex];

  const { alert, loading: alertsLoading } = useAlerts(patientId);
  const [riskScore, setRiskScore] = useState(78); // placeholder

  const alerts = alert?.hours_until_sepsis || [];

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

  // Sort alerts descending (latest first)
  const sortedAlerts = [...alerts].sort((a, b) => b - a);

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
      {alertsLoading ? (
        <p>Checking sepsis risk...</p>
      ) : sortedAlerts.length > 0 ? (
        <ul className="space-y-2">
          {sortedAlerts.map((hours, idx) => (
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
        <p className="text-gray-500 text-sm">
          No active alerts for this patient.
        </p>
      )}
    </div>
  );
}