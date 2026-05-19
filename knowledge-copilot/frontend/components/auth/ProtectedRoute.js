// FILE: frontend/components/auth/ProtectedRoute.js
// PURPOSE: Wraps a page and redirects to /login if the user is not authenticated.
//          Usage: wrap your page export in <ProtectedRoute>...</ProtectedRoute>
'use client'
import { useEffect, useRef } from 'react'
import { useAuth } from '@/hooks/useAuth'

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  const redirected = useRef(false)

  useEffect(() => {
    if (!loading && !user && !redirected.current) {
      redirected.current = true
      document.cookie = 'kc_session=; path=/; SameSite=Lax; expires=Thu, 01 Jan 1970 00:00:00 GMT'
      window.location.href = '/login'
    }
  }, [user, loading])

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-base)', gap: 14,
        fontFamily: 'var(--font-sans)',
      }}>
        <div style={{
          width: 42, height: 42, borderRadius: 13,
          background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: 'var(--shadow-accent)',
        }}>
          <svg style={{ width: 20, height: 20, color: 'var(--accent)',
            animation: 'spin-slow 1.1s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
          loading…
        </p>
      </div>
    )
  }

  if (!user) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-base)', gap: 14,
        fontFamily: 'var(--font-sans)',
      }}>
        <div style={{
          width: 42, height: 42, borderRadius: 13,
          background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: 'var(--shadow-accent)',
        }}>
          <svg style={{ width: 20, height: 20, color: 'var(--accent)',
            animation: 'spin-slow 1.1s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
          redirecting…
        </p>
      </div>
    )
  }

  return <>{children}</>
}