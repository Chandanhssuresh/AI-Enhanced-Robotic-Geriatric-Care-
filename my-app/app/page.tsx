import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Heart, Shield, Clock, Users, Brain, Phone, ArrowRight, CheckCircle } from "lucide-react"
import Link from "next/link"
import { SupabaseCheck } from "@/components/supabase-check"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">CareBot AI</span>
          </div>
          <nav className="hidden md:flex items-center space-x-6">
            <Link href="#features" className="text-gray-600 hover:text-gray-900 transition-colors">
              Features
            </Link>
            <Link href="#services" className="text-gray-600 hover:text-gray-900 transition-colors">
              Services
            </Link>
            <Link href="#about" className="text-gray-600 hover:text-gray-900 transition-colors">
              About
            </Link>
            <Link href="#contact" className="text-gray-600 hover:text-gray-900 transition-colors">
              Contact
            </Link>
          </nav>
          <div className="flex items-center space-x-3">
            <Button variant="ghost" asChild>
              <Link href="/login">Sign In</Link>
            </Button>
            <Button asChild>
              <Link href="/signup">Get Started</Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Supabase Configuration Check */}
      <section className="py-6 px-4">
        <div className="container mx-auto max-w-4xl">
          <SupabaseCheck />
        </div>
      </section>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center max-w-4xl">
          <Badge className="mb-4 bg-blue-100 text-blue-800 hover:bg-blue-100">AI-Powered Geriatric Care</Badge>
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
            Compassionate Care Through
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-green-600">
              {" "}
              AI Robotics
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Revolutionary AI-powered robotic companions providing 24/7 personalized care, monitoring, and companionship
            for elderly loved ones in the comfort of their homes.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="text-lg px-8 py-6" asChild>
              <Link href="/demo">
                Schedule Demo <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="text-lg px-8 py-6 bg-transparent" asChild>
              <Link href="#features">Learn More</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 bg-white">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Advanced Care Features</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our AI robotic care system combines cutting-edge technology with human compassion
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                  <Clock className="w-6 h-6 text-blue-600" />
                </div>
                <CardTitle>24/7 Monitoring</CardTitle>
                <CardDescription>
                  Continuous health monitoring and vital sign tracking with instant alerts
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                  <Brain className="w-6 h-6 text-green-600" />
                </div>
                <CardTitle>AI Companionship</CardTitle>
                <CardDescription>
                  Intelligent conversation and emotional support through advanced AI interaction
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                  <Shield className="w-6 h-6 text-purple-600" />
                </div>
                <CardTitle>Emergency Response</CardTitle>
                <CardDescription>
                  Immediate emergency detection and response with family and medical alerts
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mb-4">
                  <Heart className="w-6 h-6 text-orange-600" />
                </div>
                <CardTitle>Medication Management</CardTitle>
                <CardDescription>
                  Smart medication reminders and dosage tracking with health integration
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center mb-4">
                  <Users className="w-6 h-6 text-teal-600" />
                </div>
                <CardTitle>Family Connection</CardTitle>
                <CardDescription>
                  Real-time updates and video calls to keep families connected and informed
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4">
                  <Phone className="w-6 h-6 text-red-600" />
                </div>
                <CardTitle>Telehealth Integration</CardTitle>
                <CardDescription>
                  Seamless connection with healthcare providers and virtual consultations
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section id="services" className="py-20 px-4 bg-gray-50">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Care Plans</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Choose the perfect care plan tailored to your loved one&apos;s needs
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <Card className="border-2 hover:border-blue-200 transition-colors">
              <CardHeader>
                <CardTitle className="text-2xl">Essential Care</CardTitle>
                <CardDescription className="text-lg">Basic monitoring and companionship</CardDescription>
                <div className="text-3xl font-bold text-blue-600 mt-4">
                  $299<span className="text-lg text-gray-500">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Daily health check-ins
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Medication reminders
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Basic companionship
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Family updates
                  </li>
                </ul>
                <Button className="w-full mt-6 bg-transparent" variant="outline">
                  Choose Plan
                </Button>
              </CardContent>
            </Card>

            <Card className="border-2 border-blue-500 relative">
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <Badge className="bg-blue-500 text-white">Most Popular</Badge>
              </div>
              <CardHeader>
                <CardTitle className="text-2xl">Complete Care</CardTitle>
                <CardDescription className="text-lg">Comprehensive monitoring and support</CardDescription>
                <div className="text-3xl font-bold text-blue-600 mt-4">
                  $599<span className="text-lg text-gray-500">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    24/7 health monitoring
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Emergency response
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    AI companionship
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Telehealth integration
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Family dashboard
                  </li>
                </ul>
                <Button className="w-full mt-6">Choose Plan</Button>
              </CardContent>
            </Card>

            <Card className="border-2 hover:border-blue-200 transition-colors">
              <CardHeader>
                <CardTitle className="text-2xl">Premium Care</CardTitle>
                <CardDescription className="text-lg">Advanced care with specialized support</CardDescription>
                <div className="text-3xl font-bold text-blue-600 mt-4">
                  $899<span className="text-lg text-gray-500">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    All Complete Care features
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Specialized health programs
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Priority support
                  </li>
                  <li className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                    Custom care plans
                  </li>
                </ul>
                <Button className="w-full mt-6 bg-transparent" variant="outline">
                  Choose Plan
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-blue-600 to-green-600">
        <div className="container mx-auto text-center">
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Transform Elder Care?</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
            Join thousands of families who trust CareBot AI for their loved ones&apos; wellbeing
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100 text-lg px-8 py-6">
              Start Free Trial
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-white text-white hover:bg-white hover:text-blue-600 text-lg px-8 py-6 bg-transparent"
            >
              Contact Sales
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4">
        <div className="container mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
                  <Heart className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold">CareBot AI</span>
              </div>
              <p className="text-gray-400">
                Revolutionizing geriatric care through AI-powered robotics and compassionate technology.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Product</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Features
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Pricing
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Demo
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Support</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Help Center
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Contact
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Privacy
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    About
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    Careers
                  </Link>
                </li>
                <li>
                  <Link href="#" className="hover:text-white transition-colors">
                    News
                  </Link>
                </li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 CareBot AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
