import { useEffect, useState } from "react";

export default function useAlerts(patientId: string) {
  const [alerts, setAlerts] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!patientId) return;
    setAlerts([]); // reset alerts when switching patients

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProtocol}://${window.location.hostname}:8000/ws/alerts/${patientId}`;
    console.log("🔗 Connecting to WebSocket:", wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`✅ WebSocket connected for ${patientId}`);
      setLoading(false);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setAlerts((prevAlerts) => [...prevAlerts, ...(data.hours_until_sepsis || [])]);
    };

    ws.onerror = (event: Event) => console.error("❌ WebSocket error:", event);
    ws.onclose = () => console.log(`🔌 WebSocket closed for ${patientId}`);

    return () => ws.close();
  }, [patientId]);

  return { alert: { hours_until_sepsis: alerts }, loading };
}