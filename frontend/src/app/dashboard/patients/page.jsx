"use client";
import Link from "next/link";
const patients = [
  { id: "patient1", name: "Patient 1" },
  { id: "patient2", name: "Patient 2" },
  { id: "patient3", name: "Patient 3" },
];

export default function PatientsPage() {
  

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
        <h2 className="text-4xl font-extrabold mb-4 text-gray-800">Patients</h2>
        <p className="text-lg text-gray-600 max-w-2xl text-center mb-8">
          Browse through the list of patients currently admitted and monitor their vital statistics in real-time.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {patients.map((patient) => (
            <Link key={patient.id} href={`/dashboard/patients/${patient.id}`}>
              <button className="w-48 px-6 py-4 bg-white text-green-700 font-semibold rounded-xl shadow hover:bg-green-100 transition">
                {patient.name}
              </button>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
