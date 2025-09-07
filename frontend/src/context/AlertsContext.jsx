"use client";
import { createContext, useContext, useEffect, useState } from "react";

const AlertsContext = createContext(null);

export const AlertsProvider = ({ children }) => {
  const [alertsMap, setAlertsMap] = useState({});   // { patientId: [alerts] }
  const [loadingMap, setLoadingMap] = useState({}); // { patientId: true/false }
  const wsConnections = new Map();

  // Connect WebSocket per patient
  const connectWebSocket = (patientId) => {
    if (wsConnections.has(patientId)) return; // Already connected

    setLoadingMap((prev) => ({ ...prev, [patientId]: true }));

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/alerts/${patientId}`);

    ws.onopen = () => {
      console.log(`🔗 WebSocket connected for ${patientId}`);
      setLoadingMap((prev) => ({ ...prev, [patientId]: false }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.hours_until_sepsis) {
        setAlertsMap((prev) => ({
          ...prev,
          [patientId]: [...(prev[patientId] || []), ...data.hours_until_sepsis],
        }));
      }
    };

    ws.onerror = (err) => console.error(`❌ WebSocket error for ${patientId}`, err);
    ws.onclose = () => {
      console.log(`🔌 WebSocket closed for ${patientId}`);
      wsConnections.delete(patientId);
    };

    wsConnections.set(patientId, ws);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsConnections.forEach((ws) => ws.close());
    };
  }, []);

  return (
    <AlertsContext.Provider value={{ alertsMap, loadingMap, connectWebSocket }}>
      {children}
    </AlertsContext.Provider>
  );
};

export const useAlertsContext = () => useContext(AlertsContext);