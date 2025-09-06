import Link from "next/link";

export default function DashboardLayout({ children }) {
  return (
    <main className="flex min-h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-md p-6 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-green-700">SepsisSense</h1>
        <nav className="flex flex-col gap-4">
          <Link href="/dashboard/vitals" className="text-gray-700 hover:text-green-700">Vitals</Link>
          <Link href="/dashboard/patients" className="text-gray-700 hover:text-green-700">Patients</Link>
          <Link href="/dashboard/alerts" className="text-gray-700 hover:text-green-700">Alerts</Link>
          <Link href="/dashboard/settings" className="text-gray-700 hover:text-green-700">Settings</Link>
        </nav>
      </aside>

      {/* Page Content */}
      <section className="flex-1 p-8">{children}</section>
    </main>
  );
}