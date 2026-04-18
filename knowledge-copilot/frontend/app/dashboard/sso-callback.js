'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'

// This page handles the redirect back from Google via Clerk.
// URL: /sso-callback
// Clerk redirects here after OAuth, we pick up the session token,
// exchange it for our own JWT, then redirect to dashboard.

export default function SSOCallback() {
  const router = useRouter()
  const { loginWithClerk } = useAuth()
  const [error, setError] = useState(null)

  useEffect(() => {
    async function handle() {
      try {
        // Wait for Clerk to finish processing the OAuth callback
        if (typeof window === 'undefined' || !window.Clerk) {
          setTimeout(handle, 300)
          return
        }

        await window.Clerk.load()

        // Get the active session token from Clerk
        const session = window.Clerk.session
        if (!session) throw new Error('No active Clerk session')

        const token = await session.getToken()
        await loginWithClerk(token)

        document.cookie = 'kc_session=1; path=/; SameSite=Lax'
        router.replace('/dashboard')
      } catch (err) {
        setError(err.message)
      }
    }
    handle()
  }, [loginWithClerk, router])

  if (error) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-base)',
        fontFamily: 'var(--font-sans)',
      }}>
        <div style={{
          padding: '32px',
          borderRadius: 16,
          background: 'var(--bg-surface)',
          border: '1px solid var(--danger)',
          textAlign: 'center',
          maxWidth: 360,
        }}>
          <p style={{ color: 'var(--danger)', fontSize: 14, marginBottom: 16 }}>
            Sign-in failed: {error}
          </p>
          <button
            onClick={() => router.push('/login')}
            style={{
              padding: '8px 20px',
              borderRadius: 8,
              background: 'var(--accent)',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              fontSize: 13,
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
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg-base)',
      fontFamily: 'var(--font-sans)',
      gap: 16,
    }}>
      <div style={{
        width: 42,
        height: 42,
        borderRadius: 13,
        background: 'var(--accent-soft)',
        border: '1.5px solid var(--accent)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: 'var(--shadow-accent)',
      }}>
        <svg style={{ width: 20, height: 20, color: 'var(--accent)',
          animation: 'spin-slow 1.1s linear infinite' }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M21 12a9 9 0 11-6.219-8.56"/>
        </svg>
      </div>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        Completing sign-in…
      </p>
    </div>
  )
}