"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  Users,
  AlertTriangle,
  TrendingUp,
  Heart,
  Brain,
  Stethoscope,
  Shield,
} from "lucide-react";

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
    <div className="min-h-screen bg-background">
      {/* Navigation Bar */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Heart className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">SepsisSense</span>
            </div>

            {user ? (
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                className="bg-red-500 text-white hover:bg-red-600"
              >
                Logout
              </Button>
            ) : (
              <Link href="/login">
                <Button variant="outline" size="sm">
                  Login
                </Button>
              </Link>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-background to-accent/5">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center space-y-8">
            <div className="space-y-4">
              <Badge
                variant="secondary"
                className="px-4 py-2 text-sm font-medium"
              >
                <Brain className="w-4 h-4 mr-2" />
                AI-Powered Medical Prediction
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold text-balance">
                
                <span className="text-primary">Sepsis Prediction</span>{" "}
                
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-pretty leading-relaxed">
                Harness the power of advanced machine learning to predict sepsis
                onset with unprecedented accuracy. Our platform analyzes patient
                vitals in real-time, providing critical insights that save
                lives.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link href="/dashboard/patients">
                <Button size="lg" className="px-8 py-6 text-lg font-semibold">
                  <Users className="w-5 h-5 mr-2" />
                  Patients
                </Button>
              </Link>
              <Link href="/about">
                <Button
                  variant="outline"
                  size="lg"
                  className="px-8 py-6 text-lg font-semibold bg-transparent"
                >
                  <Shield className="w-5 h-5 mr-2" />
                  About Us
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      

      {/* Features Section */}
      <div className="py-20 bg-background">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Advanced Medical Intelligence
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Our platform combines cutting-edge AI with clinical expertise to
              deliver unparalleled sepsis prediction capabilities.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="group hover:shadow-xl transition-all duration-300 hover:-translate-y-1 bg-gradient-to-br from-card to-card/50">
              <CardHeader>
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  Real-Time Analysis
                </CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  Continuous monitoring of patient vitals with instant
                  AI-powered risk assessment and predictive modeling.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="group hover:shadow-xl transition-all duration-300 hover:-translate-y-1 bg-gradient-to-br from-card to-card/50">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-accent/20 transition-colors">
                  <Stethoscope className="w-6 h-6 text-accent" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  Clinical Integration
                </CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  Seamlessly integrates with hospital systems and EHRs for
                  comprehensive care.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="group hover:shadow-xl transition-all duration-300 hover:-translate-y-1 bg-gradient-to-br from-card to-card/50">
              <CardHeader>
                <div className="w-12 h-12 bg-chart-3/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-chart-3/20 transition-colors">
                  <AlertTriangle className="w-6 h-6 text-chart-3" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  Early Warning System
                </CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  Advanced alert mechanisms notify staff of potential sepsis
                  cases hours before traditional methods.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-card border-t border-border py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Heart className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">
                SepsisSense
              </span>
            </div>
            <p className="text-muted-foreground text-center md:text-right">
              © 2025 SepsisSense. Revolutionizing healthcare through artificial
              intelligence.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
