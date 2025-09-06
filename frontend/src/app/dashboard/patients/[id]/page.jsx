"use client";

export default function PatientDetailsPage() {
  const patient = {
    id: "patient1",
    name: "John Doe",
    age: 45,
    gender: "Male",
    admissionDate: "2025-09-01",
    condition: "Sepsis Risk - High",
  };

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-extrabold text-slate-800">
        {patient.name}’s Details
      </h2>

      {/* Patient Info */}
      <div className="grid grid-cols-2 gap-6 bg-white p-6 rounded-2xl shadow-lg border border-slate-200">
        <Info label="Age" value={patient.age} />
        <Info label="Gender" value={patient.gender} />
        <Info label="Admission Date" value={patient.admissionDate} />
        <Info label="Condition" value={patient.condition} highlight />
      </div>
    </div>
  );
}

function Info({ label, value, highlight }) {
  return (
    <div>
      <p className="text-sm text-slate-500">{label}</p>
      <p
        className={`text-lg font-medium ${
          highlight ? "text-red-600 font-semibold" : "text-black"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
