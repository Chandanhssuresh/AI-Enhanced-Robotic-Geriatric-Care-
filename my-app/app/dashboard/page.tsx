import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Heart, Users, Activity, AlertTriangle, Plus, TrendingUp, Clock, Shield, Bell } from "lucide-react"
import Link from "next/link"
import { getUserProfile } from "@/app/actions/auth"

export default async function DashboardPage() {
  const profile = await getUserProfile()

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-green-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2">Welcome to your Care Dashboard, {profile?.first_name}!</h1>
        <p className="text-blue-100">
          Monitor and manage care for your loved ones with AI-powered insights and real-time updates.
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Care Recipients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">No recipients added yet</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">All systems normal</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Health Checks</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">This week</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Shield className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">Online</div>
            <p className="text-xs text-muted-foreground">All systems operational</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Getting Started */}
        <Card>
          <CardHeader>
            <CardTitle>Getting Started</CardTitle>
            <CardDescription>Set up your care network and start monitoring your loved ones</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                  <Users className="w-4 h-4 text-blue-600" />
                </div>
                <div>
                  <p className="font-medium">Add Care Recipients</p>
                  <p className="text-sm text-gray-500">Add family members to monitor</p>
                </div>
              </div>
              <Button size="sm" asChild>
                <Link href="/dashboard/care-recipients">
                  <Plus className="w-4 h-4 mr-1" />
                  Add
                </Link>
              </Button>
            </div>

            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  <Activity className="w-4 h-4 text-green-600" />
                </div>
                <div>
                  <p className="font-medium">Configure Health Monitoring</p>
                  <p className="text-sm text-gray-500">Set up vital sign tracking</p>
                </div>
              </div>
              <Button size="sm" variant="outline">
                Configure
              </Button>
            </div>

            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                  <Bell className="w-4 h-4 text-purple-600" />
                </div>
                <div>
                  <p className="font-medium">Set Up Alerts</p>
                  <p className="text-sm text-gray-500">Configure emergency notifications</p>
                </div>
              </div>
              <Button size="sm" variant="outline">
                Setup
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest updates from your care network</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              <Clock className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium mb-2">No activity yet</p>
              <p className="text-sm">Activity will appear here once you add care recipients and start monitoring.</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Care Plans */}
      <Card>
        <CardHeader>
          <CardTitle>Your Care Plan</CardTitle>
          <CardDescription>Manage your subscription and upgrade options</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 border rounded-lg bg-blue-50">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                <Heart className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="font-medium">Free Trial</p>
                <p className="text-sm text-gray-600">Explore all features for 14 days</p>
              </div>
            </div>
            <div className="text-right">
              <Badge variant="secondary" className="mb-2">
                Trial Active
              </Badge>
              <p className="text-sm text-gray-600">13 days remaining</p>
            </div>
          </div>

          <div className="mt-4 flex space-x-3">
            <Button asChild>
              <Link href="/dashboard/billing">
                <TrendingUp className="w-4 h-4 mr-2" />
                Upgrade Plan
              </Link>
            </Button>
            <Button variant="outline">View Features</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
