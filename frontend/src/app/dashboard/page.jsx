"use client";
import Link from "next/link";

export default function DashboardHome() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Top Navbar */}
      <nav className="w-full bg-white shadow-md py-4 px-8 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-green-700">SepsisSense</h1>
        <Link href="/login">
          <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
            Login
          </button>
        </Link>
      </nav>

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center flex-1 p-8">
        <h2 className="text-4xl font-extrabold mb-4 text-gray-800">Welcome to SepsisSense</h2>
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
