"use client";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function DashboardHome() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch current user session
  useEffect(() => {
    async function fetchUser() {
      try {
        const res = await fetch("/api/me");
        if (res.ok) {
          const data = await res.json();
          setUser(data.user);
        }
      } catch (err) {
        console.error("Failed to fetch user:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchUser();
  }, []);

  const handleLogout = async () => {
    try {
      await fetch("/api/logout", { method: "POST" });
      setUser(null);
      window.location.href = "/"; // redirect after logout
    } catch (err) {
      console.error("Logout failed:", err);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Navbar */}
      <nav className="w-full bg-white shadow-md py-4 px-8 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-green-700">SepsisSense</h1>

        {user ? (
          <button
            onClick={handleLogout}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Logout
          </button>
        ) : (
          <Link href="/login">
            <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
              Login
            </button>
          </Link>
        )}
      </nav>

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center flex-1 p-8">
        <h2 className="text-4xl font-extrabold mb-4 text-gray-800">
          Welcome to SepsisSense
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl text-center mb-8">
          SepsisSense is an intelligent ICU monitoring dashboard designed to track
          patient vitals, detect early signs of sepsis, and alert healthcare
          professionals in real-time. Our mission is to enhance patient safety
          and provide actionable insights for better clinical decisions.
        </p>

        <div className="flex gap-6">
          <Link href="/about">
            <button className="px-6 py-3 bg-blue-500 text-white rounded-xl shadow hover:bg-blue-600 transition">
              About Us
            </button>
          </Link>

          <Link href="/dashboard/patients">
            <button className="px-6 py-3 bg-green-600 text-white rounded-xl shadow hover:bg-green-700 transition">
              View Patients
            </button>
          </Link>
        </div>
      </main>
    </div>
  );
}
