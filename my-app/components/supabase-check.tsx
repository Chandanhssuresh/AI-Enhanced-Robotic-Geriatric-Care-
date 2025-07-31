"use client"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, ExternalLink, CheckCircle } from "lucide-react"

export function SupabaseCheck() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  const isConfigured =
    supabaseUrl &&
    supabaseAnonKey &&
    supabaseUrl !== "your_supabase_project_url_here" &&
    supabaseAnonKey !== "your_supabase_anon_key_here"

  if (isConfigured) {
    return (
      <Alert className="border-green-200 bg-green-50">
        <CheckCircle className="h-4 w-4 text-green-600" />
        <AlertDescription className="text-green-800">
          Supabase is properly configured and ready to use!
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Card className="border-orange-200 bg-orange-50">
      <CardHeader>
        <CardTitle className="flex items-center text-orange-800">
          <AlertTriangle className="h-5 w-5 mr-2" />
          Supabase Configuration Required
        </CardTitle>
        <CardDescription className="text-orange-700">
          To use authentication and database features, you need to set up Supabase environment variables.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <h4 className="font-medium text-orange-800">Setup Steps:</h4>
          <ol className="list-decimal list-inside space-y-1 text-sm text-orange-700">
            <li>
              Create a new project at{" "}
              <a href="https://supabase.com" className="underline" target="_blank" rel="noopener noreferrer">
                supabase.com
              </a>
            </li>
            <li>Go to Project Settings → API</li>
            <li>Copy your Project URL and anon/public key</li>
            <li>
              Create a <code className="bg-orange-100 px-1 rounded">.env.local</code> file in your project root
            </li>
            <li>Add the environment variables as shown below</li>
          </ol>
        </div>

        <div className="bg-orange-100 p-3 rounded-md">
          <p className="text-sm font-medium text-orange-800 mb-2">Add to .env.local:</p>
          <pre className="text-xs text-orange-700 whitespace-pre-wrap">
            {`NEXT_PUBLIC_SUPABASE_URL=your_project_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here`}
          </pre>
        </div>

        <div className="flex space-x-2">
          <Button size="sm" asChild>
            <a href="https://supabase.com/dashboard" target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-1" />
              Open Supabase Dashboard
            </a>
          </Button>
          <Button size="sm" variant="outline" asChild>
            <a
              href="https://supabase.com/docs/guides/getting-started/quickstarts/nextjs"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Setup Guide
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
