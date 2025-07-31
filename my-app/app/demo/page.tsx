import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Heart, Calendar, Clock } from "lucide-react"
import Link from "next/link"

export default function DemoPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">CareBot AI</span>
          </Link>
          <Button variant="outline" asChild>
            <Link href="/">Back to Home</Link>
          </Button>
        </div>
      </header>

      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">Schedule Your Demo</h1>
            <p className="text-xl text-gray-600">See how CareBot AI can transform care for your loved one</p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Calendar className="w-5 h-5 mr-2" />
                Book a Personalized Demo
              </CardTitle>
              <CardDescription>
                Our care specialists will show you how our AI robotic system works and answer all your questions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name *</Label>
                  <Input id="firstName" placeholder="John" required />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name *</Label>
                  <Input id="lastName" placeholder="Doe" required />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address *</Label>
                <Input id="email" type="email" placeholder="john@example.com" required />
              </div>

              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number *</Label>
                <Input id="phone" type="tel" placeholder="(555) 123-4567" required />
              </div>

              <div className="space-y-2">
                <Label htmlFor="relationship">Relationship to Care Recipient</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select relationship" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="child">Adult Child</SelectItem>
                    <SelectItem value="spouse">Spouse</SelectItem>
                    <SelectItem value="sibling">Sibling</SelectItem>
                    <SelectItem value="caregiver">Professional Caregiver</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="careNeeds">Primary Care Needs</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select primary need" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="monitoring">Health Monitoring</SelectItem>
                    <SelectItem value="companionship">Companionship</SelectItem>
                    <SelectItem value="medication">Medication Management</SelectItem>
                    <SelectItem value="emergency">Emergency Response</SelectItem>
                    <SelectItem value="comprehensive">Comprehensive Care</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="preferredTime">Preferred Demo Time</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select preferred time" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="morning">Morning (9 AM - 12 PM)</SelectItem>
                    <SelectItem value="afternoon">Afternoon (12 PM - 5 PM)</SelectItem>
                    <SelectItem value="evening">Evening (5 PM - 8 PM)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="questions">Questions or Special Requirements</Label>
                <Textarea
                  id="questions"
                  placeholder="Tell us about any specific questions or requirements you have..."
                  rows={4}
                />
              </div>

              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Clock className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-blue-900">What to Expect</h3>
                    <ul className="text-sm text-blue-800 mt-2 space-y-1">
                      <li>• 30-minute personalized demonstration</li>
                      <li>• Live interaction with our AI care system</li>
                      <li>• Q&A session with care specialists</li>
                      <li>• Custom care plan recommendations</li>
                    </ul>
                  </div>
                </div>
              </div>

              <Button className="w-full" size="lg">
                Schedule My Demo
              </Button>

              <p className="text-sm text-gray-500 text-center">
                By scheduling a demo, you agree to our privacy policy. We&apos;ll never share your information.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
