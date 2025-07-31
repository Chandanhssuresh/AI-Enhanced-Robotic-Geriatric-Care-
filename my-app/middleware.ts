import { createServerClient } from "@supabase/ssr"
import { NextResponse, type NextRequest } from "next/server"

export async function middleware(request: NextRequest) {
  // Check if Supabase environment variables are available
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    console.warn("Supabase environment variables not found. Skipping authentication middleware.")
    return NextResponse.next()
  }

  let supabaseResponse = NextResponse.next({
    request,
  })

  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll()
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) => request.cookies.set(name, value))
        supabaseResponse = NextResponse.next({
          request,
        })
        cookiesToSet.forEach(({ name, value, options }) => supabaseResponse.cookies.set(name, value, options))
      },
    },
  })

  try {
    // Refresh session if expired - required for Server Components
    const {
      data: { user },
    } = await supabase.auth.getUser()

    // Protect dashboard routes
    if (request.nextUrl.pathname.startsWith("/dashboard")) {
      if (!user) {
        return NextResponse.redirect(new URL("/login", request.url))
      }
    }

    // Redirect authenticated users away from auth pages
    if ((request.nextUrl.pathname === "/login" || request.nextUrl.pathname === "/signup") && user) {
      return NextResponse.redirect(new URL("/dashboard", request.url))
    }
  } catch (error) {
    console.error("Middleware authentication error:", error)
    // Continue without authentication if there's an error
  }

  return supabaseResponse
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)"],
}
