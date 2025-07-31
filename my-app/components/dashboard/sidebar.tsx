"use client"

import { Button } from "@/components/ui/button"
import { Heart, Home, Users, Activity, Bell, Settings, LogOut } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { signOut } from "@/app/actions/auth"
import { cn } from "@/lib/utils"

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: Home },
  { name: "Care Recipients", href: "/dashboard/care-recipients", icon: Users },
  { name: "Health Metrics", href: "/dashboard/health", icon: Activity },
  { name: "Alerts", href: "/dashboard/alerts", icon: Bell },
  { name: "Settings", href: "/dashboard/settings", icon: Settings },
]

export function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg lg:block hidden">
      <div className="flex h-16 items-center px-6 border-b">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
            <Heart className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold text-gray-900">CareBot AI</span>
        </div>
      </div>

      <nav className="mt-6 px-3">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors",
                    isActive
                      ? "bg-blue-50 text-blue-700 border-r-2 border-blue-700"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>

      <div className="absolute bottom-6 left-3 right-3">
        <form action={async () => {
          await signOut()
        }}>
          <Button variant="ghost" className="w-full justify-start text-gray-600 hover:text-gray-900" type="submit">
            <LogOut className="mr-3 h-5 w-5" />
            Sign Out
          </Button>
        </form>
      </div>
    </div>
  )
}
