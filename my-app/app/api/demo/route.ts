import { createClient } from "@/utils/supabase/server"
import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const supabase = await createClient()
    const body = await request.json()

    const { data, error } = await supabase
      .from("demo_requests")
      .insert([
        {
          first_name: body.firstName,
          last_name: body.lastName,
          email: body.email,
          phone: body.phone,
          relationship: body.relationship,
          care_needs: body.careNeeds,
          preferred_time: body.preferredTime,
          questions: body.questions,
        },
      ])
      .select()

    if (error) {
      console.error("Error inserting demo request:", error)
      return NextResponse.json({ error: "Failed to submit demo request" }, { status: 500 })
    }

    return NextResponse.json({ message: "Demo request submitted successfully", data }, { status: 201 })
  } catch (error) {
    console.error("Error processing demo request:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

export async function GET() {
  try {
    const supabase = await createClient()

    const { data, error } = await supabase.from("demo_requests").select("*").order("created_at", { ascending: false })

    if (error) {
      console.error("Error fetching demo requests:", error)
      return NextResponse.json({ error: "Failed to fetch demo requests" }, { status: 500 })
    }

    return NextResponse.json({ data }, { status: 200 })
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
