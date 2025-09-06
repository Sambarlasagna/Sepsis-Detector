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
  const { id } = useParams(); // patient ID from URL
  const numericId = parseInt(id.replace("patient", ""));

  const [allVitals, setAllVitals] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);

  // Load CSV once
  useEffect(() => {
    Papa.parse("/combined_sepsis_dataset.csv", {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (result) => {
        const grouped = {};
        result.data.forEach((row) => {
          if (!grouped[row.Patient_ID]) grouped[row.Patient_ID] = [];
          grouped[row.Patient_ID].push(row);
        });
        setAllVitals(grouped);
        console.log("CSV loaded for all patients:", Object.keys(grouped));
      },
    });
  }, []);

  const vitals = allVitals[numericId] || [];

  // Simulate real-time progression
  useEffect(() => {
    if (!vitals.length) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev < vitals.length - 1) return prev + 1;
        return prev; // keep last value displayed
      });
    }, 3000); // 1 hour = 3 seconds

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
      <h2 className="text-xl font-bold mb-4">Real-Time Vitals — Patient {id}</h2>
      {vitals.length > 0 ? (
        <Line data={chartData} options={chartOptions} />
      ) : (
        <p>Loading vitals...</p>
      )}
    </div>
  );
}