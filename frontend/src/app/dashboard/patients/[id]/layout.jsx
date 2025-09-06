"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Bell, Activity, PersonStanding } from "lucide-react";

export default function DashboardLayout({ children }) {
  const pathname = usePathname();

  // Extract patientId from the URL if available
  const pathParts = pathname.split("/");
  const patientIdIndex = pathParts.indexOf("patients") + 1;
  const patientId =
    patientIdIndex > 0 && pathParts[patientIdIndex]
      ? pathParts[patientIdIndex]
      : null;

  // Build nav items dynamically if patientId exists
  const navItems = patientId
    ? [
        { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
        {
          name: "Alerts",
          href: `/dashboard/patients/${patientId}/alerts`,
          icon: Bell,
        },
        {
          name: "Vitals",
          href: `/dashboard/patients/${patientId}/vitals`,
          icon: Activity,
        },
        {
          name: "Patient Details",
          href: `/dashboard/patients/${patientId}`,
          icon: PersonStanding,
        }
      ]
    : [{ name: "Dashboard", href: "/dashboard", icon: LayoutDashboard }];

  return (
    <main className="flex min-h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-lg flex flex-col p-6">
        <h1 className="text-2xl font-bold mb-8 text-green-700">SepsisSense</h1>
        <nav className="flex flex-col gap-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition ${
                  isActive
                    ? "bg-green-100 text-green-700"
                    : "text-gray-700 hover:bg-gray-100"
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <section className="flex-1 p-8 overflow-y-auto">{children}</section>
    </main>
  );
}