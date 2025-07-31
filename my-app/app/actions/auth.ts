"use server"

import { createClient } from "@/utils/supabase/server"
import { redirect } from "next/navigation"
import { revalidatePath } from "next/cache"

export async function signUp(formData: FormData) {
  try {
    const supabase = await createClient()

    const data = {
      email: formData.get("email") as string,
      password: formData.get("password") as string,
      options: {
        data: {
          first_name: formData.get("firstName") as string,
          last_name: formData.get("lastName") as string,
        },
      },
    }

    const { data: authData, error } = await supabase.auth.signUp(data)

    if (error) {
      console.error("Sign up error:", error)
      return { error: error.message }
    }

    // Create profile in public.profiles table
    if (authData.user) {
      const { error: profileError } = await supabase.from("profiles").insert([
        {
          id: authData.user.id,
          first_name: formData.get("firstName") as string,
          last_name: formData.get("lastName") as string,
          email: authData.user.email,
          role: "family_member",
        },
      ])

      if (profileError) {
        console.error("Profile creation error:", profileError)
        return { error: "Failed to create user profile" }
      }
    }

    revalidatePath("/", "layout")
    redirect("/dashboard")
  } catch (error) {
    console.error("Signup error:", error)
    return { error: "Supabase is not configured. Please check your environment variables." }
  }
}

export async function signIn(formData: FormData) {
  try {
    const supabase = await createClient()

    const data = {
      email: formData.get("email") as string,
      password: formData.get("password") as string,
    }

    const { error } = await supabase.auth.signInWithPassword(data)

    if (error) {
      console.error("Sign in error:", error)
      return { error: error.message }
    }

    revalidatePath("/", "layout")
    redirect("/dashboard")
  } catch (error) {
    console.error("Signin error:", error)
    return { error: "Supabase is not configured. Please check your environment variables." }
  }
}

export async function signOut() {
  try {
    const supabase = await createClient()

    const { error } = await supabase.auth.signOut()

    if (error) {
      console.error("Sign out error:", error)
      return { error: error.message }
    }

    revalidatePath("/", "layout")
    redirect("/")
  } catch (error) {
    console.error("Signout error:", error)
    return { error: "Failed to sign out" }
  }
}

export async function getUser() {
  try {
    const supabase = await createClient()

    const {
      data: { user },
      error,
    } = await supabase.auth.getUser()

    if (error) {
      console.error("Get user error:", error)
      return null
    }

    return user
  } catch (error) {
    console.error("Get user error:", error)
    return null
  }
}

export async function getUserProfile() {
  try {
    const supabase = await createClient()

    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) return null

    const { data: profile, error } = await supabase.from("profiles").select("*").eq("id", user.id).single()

    if (error) {
      console.error("Get profile error:", error)
      return null
    }

    return profile
  } catch (error) {
    console.error("Get profile error:", error)
    return null
  }
}
