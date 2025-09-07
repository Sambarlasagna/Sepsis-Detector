"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Heart,
  ArrowLeft,
  Activity,
  Thermometer,
  Droplets,
  Zap,
  Calendar,
  MapPin,
  Phone,
  Mail,
} from "lucide-react"
import Link from "next/link"

// Mock patient data - replace with DB in production
const patientData = {
  patient1: {
    name: "John Anderson",
    age: 67,
    gender: "Male",
    bloodType: "O+",
    admissionDate: "2024-01-15",
    room: "ICU-204",
    phone: "+1 (555) 123-4567",
    email: "john.anderson@email.com",
    vitals: {
      heartRate: 88,
      temperature: 98.6,
      bloodPressure: "120/80",
      oxygenSaturation: 97,
      respiratoryRate: 16,
    },
    riskLevel: "Low",
    riskScore: 0.23,
  },
  patient2: {
    name: "Sarah Mitchell",
    age: 45,
    gender: "Female",
    bloodType: "A-",
    admissionDate: "2024-01-18",
    room: "Ward-312",
    phone: "+1 (555) 987-6543",
    email: "sarah.mitchell@email.com",
    vitals: {
      heartRate: 102,
      temperature: 100.2,
      bloodPressure: "140/90",
      oxygenSaturation: 94,
      respiratoryRate: 22,
    },
    riskLevel: "Medium",
    riskScore: 0.67,
  },
  patient3: {
    name: "Robert Chen",
    age: 72,
    gender: "Male",
    bloodType: "B+",
    admissionDate: "2024-01-12",
    room: "ICU-108",
    phone: "+1 (555) 456-7890",
    email: "robert.chen@email.com",
    vitals: {
      heartRate: 76,
      temperature: 97.8,
      bloodPressure: "110/70",
      oxygenSaturation: 98,
      respiratoryRate: 14,
    },
    riskLevel: "Low",
    riskScore: 0.18,
  },
}

export default function PatientDetailPage({ params }) {
  const { id } = params
  const patient = patientData[id]

  if (!patient) {
    return <div className="p-8 text-center">Patient not found</div>
  }

  const getRiskColor = (level) => {
    switch (level) {
      case "Low":
        return "text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-950/20 dark:border-green-800"
      case "Medium":
        return "text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-950/20 dark:border-yellow-800"
      case "High":
        return "text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-950/20 dark:border-red-800"
      default:
        return "text-gray-600 bg-gray-50 border-gray-200"
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Heart className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">SepsisAI</span>
            </Link>
            <Button variant="outline" size="sm">
              Login
            </Button>
          </div>
        </div>
      </nav>

      {/* Header */}
      <div className="bg-gradient-to-br from-primary/5 via-background to-accent/5 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-4 mb-6">
            <Link href="/dashboard/patients">
              <Button variant="outline" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Patients
              </Button>
            </Link>
          </div>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground mb-2">{patient.name}</h1>
              <p className="text-muted-foreground">
                Patient ID: {id} • Room: {patient.room}
              </p>
            </div>
            <Badge className={`px-4 py-2 text-sm font-semibold ${getRiskColor(patient.riskLevel)}`}>
              {patient.riskLevel} Risk ({(patient.riskScore * 100).toFixed(0)}%)
            </Badge>
          </div>
        </div>
      </div>

      {/* Patient Info + Vitals */}
      <div className="py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Patient Information */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Calendar className="w-5 h-5 mr-2" />
                    Patient Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Age</p>
                      <p className="font-semibold">{patient.age} years</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Gender</p>
                      <p className="font-semibold">{patient.gender}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Blood Type</p>
                      <p className="font-semibold">{patient.bloodType}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Room</p>
                      <p className="font-semibold flex items-center">
                        <MapPin className="w-4 h-4 mr-1" />
                        {patient.room}
                      </p>
                    </div>
                  </div>
                  <div className="pt-4 border-t">
                    <p className="text-sm text-muted-foreground">Admission Date</p>
                    <p className="font-semibold">
                      {new Date(patient.admissionDate).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground flex items-center">
                      <Phone className="w-4 h-4 mr-2" />
                      {patient.phone}
                    </p>
                    <p className="text-sm text-muted-foreground flex items-center">
                      <Mail className="w-4 h-4 mr-2" />
                      {patient.email}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Vital Signs */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="w-5 h-5 mr-2" />
                    Current Vital Signs
                  </CardTitle>
                  <CardDescription>Real-time monitoring data</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div className="flex items-center space-x-3 p-4 bg-red-50 dark:bg-red-950/20 rounded-lg">
                      <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
                        <Heart className="w-5 h-5 text-red-600 dark:text-red-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Heart Rate</p>
                        <p className="text-xl font-bold text-red-600 dark:text-red-400">
                          {patient.vitals.heartRate} BPM
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3 p-4 bg-orange-50 dark:bg-orange-950/20 rounded-lg">
                      <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
                        <Thermometer className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Temperature</p>
                        <p className="text-xl font-bold text-orange-600 dark:text-orange-400">
                          {patient.vitals.temperature}°F
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                      <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                        <Droplets className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Blood Pressure</p>
                        <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                          {patient.vitals.bloodPressure}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3 p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
                      <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                        <Zap className="w-5 h-5 text-green-600 dark:text-green-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Oxygen Saturation</p>
                        <p className="text-xl font-bold text-green-600 dark:text-green-400">
                          {patient.vitals.oxygenSaturation}%
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3 p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg">
                      <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center">
                        <Activity className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Respiratory Rate</p>
                        <p className="text-xl font-bold text-purple-600 dark:text-purple-400">
                          {patient.vitals.respiratoryRate} /min
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
