'use client'
// FILE: frontend/app/sso-callback/page.js
// PURPOSE: The /sso-callback route.
//          After Google OAuth, Clerk redirects here.
//          We get the Clerk session token and exchange it for our JWT.
export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { useAuth } from '@/hooks/useAuth'

export default function SSOCallbackPage() {
  const router = useRouter()
  const { user, isLoaded, isSignedIn } = useUser()
  const { setUser } = useAuth()
  const [status, setStatus] = useState('loading')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    async function handleOAuth() {
      console.log('SSO Callback:', { isLoaded, isSignedIn, user })
      
      if (!isLoaded) {
        setStatus('clerk_loading')
        return
      }

      if (!isSignedIn || !user) {
        console.log('Not signed in with Clerk, redirecting to login')
        router.push('/login')
        return
      }

      setStatus('authenticated')

      try {
        const clerkUserId = user.id
        const email = user.primaryEmailAddress?.emailAddress
        const name = user.fullName || user.firstName || email?.split('@')[0]
        
        console.log('Clerk user data:', { clerkUserId, email, name })

        const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000'
        console.log('Calling backend:', `${baseUrl}/auth/clerk`)
        
        const res = await fetch(`${baseUrl}/auth/clerk`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({ 
            clerk_user_id: clerkUserId,
            email: email,
            name: name,
          }),
        })

        console.log('Backend response:', res.status, res.statusText)

        if (!res.ok) {
          let body = {}
          try {
            body = await res.json()
          } catch {}
          console.error('Auth failed:', res.status, res.statusText, body)
          setStatus('auth_failed')
          setErrorMsg(body?.detail || `Server error (${res.status})`)
          return
        }

        const data = await res.json()
        console.log('Auth success, user:', data.user)
        
        const { setToken } = await import('@/lib/api')
        setToken(data.access_token)
        sessionStorage.setItem('kc_token', data.access_token)
        sessionStorage.setItem('kc_authed', '1')
        document.cookie = 'kc_session=1; path=/; SameSite=Lax'
        
        // Update auth context with user data
        setUser(data.user)
        
        setStatus('redirecting')
        router.replace('/dashboard')
      } catch (err) {
        console.error('Error:', err)
        setStatus('error')
      }
    }

    if (isLoaded) {
      handleOAuth()
    }
  }, [isLoaded, isSignedIn, user, router, setUser])

  if (status === 'redirecting' || status === 'authenticated') {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-base)', fontFamily: 'var(--font-sans)',
      }}>
        <p>Redirecting to dashboard...</p>
      </div>
    )
  }

  if (status === 'error' || status === 'auth_failed' || status === 'no_token') {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-base)', fontFamily: 'var(--font-sans)',
      }}>
        <div style={{
          padding: '32px 36px', borderRadius: 16,
          background: 'var(--bg-surface)', border: '1px solid var(--danger)',
          textAlign: 'center',
        }}>
          <p style={{ color: 'var(--danger)', marginBottom: 16 }}>
            Sign-in failed: {errorMsg}
          </p>
          <button
            onClick={() => router.push('/login')}
            style={{
              padding: '9px 24px', borderRadius: 10,
              background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer',
            }}
          >
            Back to login
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg-base)', fontFamily: 'var(--font-sans)', gap: 16,
    }}>
      <div style={{
        width: 48, height: 48, borderRadius: 14,
        background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <svg style={{ width: 22, height: 22, color: 'var(--accent)',
          animation: 'spin-slow 1.1s linear infinite' }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M21 12a9 9 0 11-6.219-8.56"/>
        </svg>
      </div>
      <p style={{ fontSize: 13, color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
        Completing sign-in... ({status})
      </p>
    </div>
  )
}