"use client";

import { useEffect, useState } from "react";
import Papa from "papaparse";
import { Line } from "react-chartjs-2";
import { useParams } from "next/navigation";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function VitalsPage() {
  const { id } = useParams(); // Get patient ID from URL params
  const [vitals, setVitals] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Load CSV
  useEffect(() => {
    console.log("Fetching CSV for patient", id);
    Papa.parse("/simulated_sepsis_data.csv", {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (result) => {
        console.log("CSV loaded:", result.data.length, "rows");
        const numericId = parseInt(id.replace("patient", ""));
        const patientVitals = result.data.filter(row => row.Patient_ID === numericId);
        console.log("Patient vitals rows:", patientVitals.length);
        setVitals(patientVitals);
      },
    });
  }, [id]);

  // Simulate real-time updates (1 hour = 3 seconds)
  useEffect(() => {
    if (vitals.length === 0) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev < vitals.length - 1) return prev + 1;
        clearInterval(interval); // stop when done
        return prev;
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [vitals]);

  const liveData = vitals.slice(0, currentIndex + 1);

  const chartData = {
    labels: liveData.map((d) => `Hour ${d.Hour}`),
    datasets: [
      {
        label: "Heart Rate (HR)",
        data: liveData.map((d) => d.HR),
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.5)",
        yAxisID: "y",
      },
      {
        label: "SpO₂",
        data: liveData.map((d) => d.O2Sat),
        borderColor: "rgb(54, 162, 235)",
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        yAxisID: "y1",
      },
      {
        label: "Temperature (°C)",
        data: liveData.map((d) => d.Temp),
        borderColor: "rgb(255, 206, 86)",
        backgroundColor: "rgba(255, 206, 86, 0.5)",
      },
      {
        label: "Respiratory Rate",
        data: liveData.map((d) => d.Resp),
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.5)",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    stacked: false,
    plugins: { title: { display: true, text: `Patient ${id} — Real-time Vitals` } },
    scales: {
      y: { type: "linear", display: true, position: "left", title: { display: true, text: "Heart Rate" } },
      y1: { type: "linear", display: true, position: "right", title: { display: true, text: "SpO₂" }, grid: { drawOnChartArea: false } },
    },
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Real-Time Vitals</h2>
      {vitals.length > 0 ? <Line data={chartData} options={chartOptions} /> : <p>Loading vitals...</p>}
    </div>
  );
}