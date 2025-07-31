import { getUserProfile } from "@/app/actions/auth"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Bell, Menu } from "lucide-react"

export async function DashboardHeader() {
  const profile = await getUserProfile()

  const initials = profile ? `${profile.first_name?.[0] || ""}${profile.last_name?.[0] || ""}`.toUpperCase() : "U"

  return (
    <header className="bg-white shadow-sm border-b h-16 flex items-center justify-between px-6">
      <div className="flex items-center">
        <Button variant="ghost" size="sm" className="lg:hidden mr-2">
          <Menu className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold text-gray-900">Welcome back, {profile?.first_name || "User"}!</h1>
      </div>

      <div className="flex items-center space-x-4">
        <Button variant="ghost" size="sm">
          <Bell className="h-5 w-5" />
        </Button>
        <Avatar>
          <AvatarFallback className="bg-blue-100 text-blue-700">{initials}</AvatarFallback>
        </Avatar>
      </div>
    </header>
  )
}
