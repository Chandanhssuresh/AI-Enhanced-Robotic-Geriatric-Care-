"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Heart, Shield, Users, Zap, ChevronDown, Bot, Brain, Activity } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  const [isVisible, setIsVisible] = useState(false)
  const [scrollY, setScrollY] = useState(0)

  useEffect(() => {
    setIsVisible(true)
    const handleScroll = () => setScrollY(window.scrollY)
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const scrollToSection = () => {
    document.getElementById("about")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <div className="min-h-screen bg-background">
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

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          <div
            className={`transition-all duration-1000 ${isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
          >
            <h1 className="text-4xl font-extrabold tracking-tight lg:text-6xl mb-6 animate-fade-in-up">
              The Foundation for your{" "}
              <span className="text-foreground bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent animate-pulse">
                AI-Enhanced Care System
              </span>
            </h1>

            <p className="mx-auto max-w-[700px] text-lg text-muted-foreground mb-8 animate-fade-in-up animation-delay-300">
              A set of beautifully designed robotic solutions that you can customize, extend, and build on. Start here
              then make it your own. Compassionate. Intelligent. Secure.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16 animate-fade-in-up animation-delay-600">
              <Link href="/signup">
                <button className="group relative px-8 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg text-white font-medium transition-all duration-300 hover:bg-white/20 hover:border-white/40 hover:shadow-lg hover:shadow-white/25 hover:scale-105">
                  <span className="relative z-10">Sign Up</span>
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-white/10 to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </button>
              </Link>

              <Link href="/signin">
                <button className="group relative px-8 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg text-white font-medium transition-all duration-300 hover:bg-white/20 hover:border-white/40 hover:shadow-lg hover:shadow-white/25 hover:scale-105">
                  <span className="relative z-10">Sign In</span>
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-white/10 to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </button>
              </Link>
            </div>

            <button
              onClick={scrollToSection}
              className="animate-bounce text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Scroll to learn more"
            >
              <ChevronDown className="w-8 h-8" />
            </button>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl mb-6">Revolutionizing Geriatric Care</h2>
            <p className="mx-auto max-w-[700px] text-lg text-muted-foreground">
              Our AI-enhanced robotic systems provide comprehensive support for elderly care, combining cutting-edge
              technology with human compassion.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            <Card className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                  <Heart className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Compassionate Care</h3>
                <p className="text-muted-foreground">
                  Our robots are designed with empathy algorithms to provide emotional support and companionship to
                  elderly patients.
                </p>
              </CardContent>
            </Card>

            <Card className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                  <Shield className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">24/7 Monitoring</h3>
                <p className="text-muted-foreground">
                  Continuous health monitoring with AI-powered early detection of potential health issues and
                  emergencies.
                </p>
              </CardContent>
            </Card>

            <Card className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                  <Zap className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Smart Assistance</h3>
                <p className="text-muted-foreground">
                  Intelligent assistance with daily activities, medication reminders, and personalized care routines.
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="bg-muted/50 rounded-2xl p-12 text-center">
            <Users className="w-16 h-16 text-primary mx-auto mb-6" />
            <h3 className="text-2xl font-bold tracking-tight mb-4">Trusted by Healthcare Professionals</h3>
            <p className="mx-auto max-w-[600px] text-muted-foreground mb-8">
              Our AI-enhanced robotic care systems are developed in collaboration with geriatricians, nurses, and
              caregivers to ensure the highest standards of care and safety.
            </p>
            <div className="flex flex-wrap justify-center gap-8 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span>FDA Compliant</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span>HIPAA Secure</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span>24/7 Support</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-12 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-muted-foreground">
            © 2024 AI-Enhanced Robotic Geriatric Care. Empowering compassionate care through technology.
          </p>
        </div>
      </footer>
    </div>
  )
}
