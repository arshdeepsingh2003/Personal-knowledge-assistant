// FILE: frontend/app/page.js
// PURPOSE: The root "/" route. Does nothing except redirect.
//          Logged-in users → /dashboard
//          Everyone else   → /login
'use client'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'

export default function RootPage() {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading) {
      router.replace(user ? '/dashboard' : '/login')
    }
  }, [user, loading, router])

  // Render nothing — this page only redirects
  return null
}