"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import { Activity, Brain, Mic, Shield, Edit, LogOut } from "lucide-react"
import { supabase } from "@/lib/supabase"

// Sample chart data
const healthData = [
  { name: "Jan", value: 85 },
  { name: "Feb", value: 88 },
  { name: "Mar", value: 82 },
  { name: "Apr", value: 90 },
  { name: "May", value: 87 },
  { name: "Jun", value: 92 },
]

const emotionData = [
  { name: "Jan", value: 75 },
  { name: "Feb", value: 78 },
  { name: "Mar", value: 85 },
  { name: "Apr", value: 82 },
  { name: "May", value: 88 },
  { name: "Jun", value: 90 },
]

const speechData = [
  { name: "Jan", value: 70 },
  { name: "Feb", value: 75 },
  { name: "Mar", value: 78 },
  { name: "Apr", value: 85 },
  { name: "May", value: 82 },
  { name: "Jun", value: 88 },
]

const intrusionData = [
  { name: "Jan", value: 95 },
  { name: "Feb", value: 92 },
  { name: "Mar", value: 98 },
  { name: "Apr", value: 94 },
  { name: "May", value: 97 },
  { name: "Jun", value: 99 },
]

export default function Dashboard() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editForm, setEditForm] = useState({
    firstName: "",
    lastName: "",
    phoneNumber: "",
    email: "",
    username: "",
  })
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // Check if user is logged in
    const userData = sessionStorage.getItem("user")
    if (!userData) {
      router.push("/signin")
      return
    }

    const parsedUser = JSON.parse(userData)
    setUser(parsedUser)
    setEditForm({
      firstName: parsedUser.first_name || "",
      lastName: parsedUser.last_name || "",
      phoneNumber: parsedUser.phone_number || "",
      email: parsedUser.email || "",
      username: parsedUser.username || "",
    })
  }, [router])

  const handleLogout = () => {
    sessionStorage.removeItem("user")
    router.push("/")
  }

  const handleEditProfile = async () => {
    if (!user) return
    setIsLoading(true)

    try {
      const { error } = await supabase
        .from("signuppage")
        .update({
          first_name: editForm.firstName,
          last_name: editForm.lastName,
          phone_number: editForm.phoneNumber,
          email: editForm.email,
          username: editForm.username,
        })
        .eq("id", user.id)

      if (error) throw error

      // Update local user data
      const updatedUser = {
        ...user,
        first_name: editForm.firstName,
        last_name: editForm.lastName,
        phone_number: editForm.phoneNumber,
        email: editForm.email,
        username: editForm.username,
      }

      setUser(updatedUser)
      sessionStorage.setItem("user", JSON.stringify(updatedUser))
      setIsEditOpen(false)

      // Show success message
      alert("Profile edited successfully!")
    } catch (error: any) {
      alert("Error updating profile: " + error.message)
    } finally {
      setIsLoading(false)
    }
  }

  const navigationCards = [
    {
      title: "Health Condition",
      icon: Activity,
      description: "Monitor vital signs and health metrics",
      href: "/health-condition",
      gradient: "from-green-500/20 to-emerald-500/20",
      iconColor: "text-green-500",
    },
    {
      title: "Face Emotion",
      icon: Brain,
      description: "Analyze facial expressions and emotions",
      href: "/face-emotion",
      gradient: "from-blue-500/20 to-cyan-500/20",
      iconColor: "text-blue-500",
    },
    {
      title: "Speech Recognition",
      icon: Mic,
      description: "Process and analyze speech patterns",
      href: "/speech-recognition",
      gradient: "from-purple-500/20 to-pink-500/20",
      iconColor: "text-purple-500",
    },
    {
      title: "Intrusion Detection",
      icon: Shield,
      description: "Security monitoring and alerts",
      href: "/intrusion-detection",
      gradient: "from-red-500/20 to-orange-500/20",
      iconColor: "text-red-500",
    },
  ]

  if (!user) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-black/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold">AI Geriatric Care Dashboard</h1>
            </div>

            <div className="flex items-center space-x-4">
              {/* Profile Section */}
              <div className="flex items-center space-x-3">
                <Avatar className="h-10 w-10">
                  <AvatarImage
                    src={`https://api.dicebear.com/7.x/initials/svg?seed=${user.first_name} ${user.last_name}`}
                  />
                  <AvatarFallback className="bg-gray-700">
                    {user.first_name?.[0]}
                    {user.last_name?.[0]}
                  </AvatarFallback>
                </Avatar>
                <div className="hidden md:block">
                  <p className="text-sm font-medium">
                    {user.first_name} {user.last_name}
                  </p>
                  <p className="text-xs text-gray-400">{user.email}</p>
                </div>
              </div>

              {/* Edit Profile Button */}
              <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" className="border-gray-700 hover:bg-gray-800 bg-transparent">
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Profile
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-gray-900 border-gray-700">
                  <DialogHeader>
                    <DialogTitle>Edit Profile</DialogTitle>
                    <DialogDescription>Update your profile information here.</DialogDescription>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="firstName">First Name</Label>
                        <Input
                          id="firstName"
                          value={editForm.firstName}
                          onChange={(e) => setEditForm({ ...editForm, firstName: e.target.value })}
                          className="bg-gray-800 border-gray-700"
                        />
                      </div>
                      <div>
                        <Label htmlFor="lastName">Last Name</Label>
                        <Input
                          id="lastName"
                          value={editForm.lastName}
                          onChange={(e) => setEditForm({ ...editForm, lastName: e.target.value })}
                          className="bg-gray-800 border-gray-700"
                        />
                      </div>
                    </div>
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={editForm.email}
                        onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                        className="bg-gray-800 border-gray-700"
                      />
                    </div>
                    <div>
                      <Label htmlFor="username">Username</Label>
                      <Input
                        id="username"
                        value={editForm.username}
                        onChange={(e) => setEditForm({ ...editForm, username: e.target.value })}
                        className="bg-gray-800 border-gray-700"
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        value={editForm.phoneNumber}
                        onChange={(e) => setEditForm({ ...editForm, phoneNumber: e.target.value })}
                        className="bg-gray-800 border-gray-700"
                      />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button onClick={handleEditProfile} disabled={isLoading} className="bg-blue-600 hover:bg-blue-700">
                      {isLoading ? "Saving..." : "Save Changes"}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              {/* Logout Button */}
              <Button
                onClick={handleLogout}
                variant="outline"
                size="sm"
                className="border-red-500 text-red-500 hover:bg-red-500/10 hover:border-red-400 hover:text-red-400 transition-all duration-300 hover:shadow-lg hover:shadow-red-500/25 bg-transparent"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Welcome back, {user.first_name}!</h2>
          <p className="text-gray-400">Monitor and manage AI-enhanced geriatric care systems</p>
        </div>

        {/* Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {navigationCards.map((card, index) => (
            <Card
              key={index}
              className={`bg-gradient-to-br ${card.gradient} border-gray-700 hover:border-gray-600 transition-all duration-300 cursor-pointer group hover:scale-105 hover:shadow-2xl backdrop-blur-sm`}
              onClick={() => router.push(card.href)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <card.icon
                    className={`h-8 w-8 ${card.iconColor} group-hover:scale-110 transition-transform duration-300`}
                  />
                  <Badge variant="secondary" className="bg-white/10 text-white border-0">
                    Active
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <CardTitle className="text-lg mb-2 text-white group-hover:text-white/90">{card.title}</CardTitle>
                <CardDescription className="text-gray-300 group-hover:text-gray-200">
                  {card.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Analytics Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-green-500" />
                Health Condition Trends
              </CardTitle>
              <CardDescription>Monthly health metrics overview</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={healthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Area type="monotone" dataKey="value" stroke="#10B981" fill="#10B981" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-500" />
                Face Emotion Analysis
              </CardTitle>
              <CardDescription>Emotional state tracking</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={emotionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    dot={{ fill: "#3B82F6", strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5 text-purple-500" />
                Speech Recognition
              </CardTitle>
              <CardDescription>Voice analysis patterns</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={speechData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Area type="monotone" dataKey="value" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-red-500" />
                Intrusion Detection
              </CardTitle>
              <CardDescription>Security monitoring status</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={intrusionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#EF4444"
                    strokeWidth={3}
                    dot={{ fill: "#EF4444", strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
