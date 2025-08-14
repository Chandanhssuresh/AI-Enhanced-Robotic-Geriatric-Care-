"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Eye, EyeOff, ArrowLeft, Bot, Brain, Activity, Zap } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase"

export default function SignInPage() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    emailOrUsername: "",
    password: "",
  })
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    if (error) setError("") // Clear error when user starts typing
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.emailOrUsername || !formData.password) {
      setError("Please fill in all fields")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      // Check if input is email or username
      const isEmail = formData.emailOrUsername.includes("@")

      // Query Supabase to find matching user
      const { data, error: supabaseError } = await supabase
        .from("signuppage")
        .select("*")
        .or(isEmail ? `email.eq.${formData.emailOrUsername}` : `username.eq.${formData.emailOrUsername}`)
        .single()

      if (supabaseError || !data) {
        setError(
          "Invalid credentials. Please check your email/username and password, or sign up if you don't have an account.",
        )
        return
      }

      // Check password (Note: In production, passwords should be hashed)
      if (data.password !== formData.password) {
        setError(
          "Invalid credentials. Please check your email/username and password, or sign up if you don't have an account.",
        )
        return
      }

      // Success - redirect to dashboard
      console.log("Login successful for user:", data.username)

      // Store user info for this session in the shape the dashboard expects
      sessionStorage.setItem(
        "user",
        JSON.stringify({
          id: data.id,
          username: data.username,
          email: data.email,
          first_name: data.first_name,
          last_name: data.last_name,
          phone_number: data.phone_number,
        }),
      )

      // Redirect to dashboard
      router.push("/dashboard")
    } catch (error) {
      console.error("Login error:", error)
      setError("An error occurred during sign in. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 animate-float">
          <Bot className="w-8 h-8 text-primary/20" />
        </div>
        <div className="absolute top-40 right-20 animate-float-delayed">
          <Brain className="w-6 h-6 text-primary/30" />
        </div>
        <div className="absolute bottom-40 left-20 animate-pulse">
          <Activity className="w-10 h-10 text-primary/20" />
        </div>
        <div className="absolute top-60 left-1/2 animate-float">
          <Zap className="w-7 h-7 text-primary/25" />
        </div>
      </div>

      <div className="relative z-10 w-full max-w-md">
        {/* Back Button */}
        <Link
          href="/"
          className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Link>

        <Card className="border-border/50 backdrop-blur-sm bg-card/95">
          <CardHeader className="space-y-1 text-center">
            <CardTitle className="text-2xl font-bold">Welcome Back</CardTitle>
            <CardDescription>Sign in to your AI-Enhanced Care account</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
                  {error}
                  {error.includes("Invalid credentials") && (
                    <div className="mt-2">
                      <Link href="/signup" className="text-primary hover:underline font-medium">
                        Don't have an account? Sign up here
                      </Link>
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="emailOrUsername">Email or Username</Label>
                <Input
                  id="emailOrUsername"
                  name="emailOrUsername"
                  type="text"
                  placeholder="Enter your email or username"
                  value={formData.emailOrUsername}
                  onChange={handleInputChange}
                  className="bg-background/50"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className="bg-background/50 pr-10"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <Button type="submit" className="w-full bg-primary hover:bg-primary/90" disabled={isLoading}>
                {isLoading ? "Signing In..." : "Sign In"}
              </Button>
            </form>

            <div className="mt-6 text-center text-sm text-muted-foreground">
              Don't have an account?{" "}
              <Link href="/signup" className="text-primary hover:underline font-medium">
                Sign up here
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
