'use client'
// FILE: frontend/app/login/page.js
// PURPOSE: The /login route — email+password form + Google sign-in button.
export const dynamic = 'force-dynamic'
import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/hooks/useAuth'

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
    </svg>
  )
}

export default function LoginPage() {
  const router       = useRouter()
  const searchParams = useSearchParams()
  const { login, user, loading: authLoading } = useAuth()

  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [showPass, setShowPass] = useState(false)
  const [mounted,  setMounted]  = useState(false)

  useEffect(() => { setMounted(true) }, [])

  useEffect(() => {
    if (!authLoading && user) {
      router.replace(searchParams.get('next') ?? '/dashboard')
    }
  }, [user, authLoading, router, searchParams])

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email.trim(), password)
      document.cookie = 'kc_session=1; path=/; SameSite=Lax'
      router.replace(searchParams.get('next') ?? '/dashboard')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleGoogleLogin() {
    setError('')
    // Use direct redirect to Clerk sign-in page
    window.location.href = '/sign-in?redirect_url=' + encodeURIComponent('/sso-callback')
  }

  function focusBorder(e) {
    e.target.style.borderColor = 'var(--accent)'
    e.target.style.boxShadow   = '0 0 0 3px var(--accent-glow)'
  }
  function blurBorder(e) {
    e.target.style.borderColor = 'var(--border-med)'
    e.target.style.boxShadow   = 'none'
  }

  if (!mounted || authLoading) return null

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      background: 'var(--bg-base)',
      fontFamily: 'var(--font-sans)',
    }}>
      {/* ── Left panel — branding ── */}
      <div style={{
        width: '42%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'flex-start',
        padding: '60px 56px',
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border-med)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Ambient glow blob */}
        <div style={{
          position: 'absolute', top: '-80px', left: '-80px',
          width: 400, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, var(--accent-glow) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 56, position: 'relative' }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: 'var(--shadow-accent)',
          }}>
            <svg style={{ width: 18, height: 18, color: '#fff' }}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            Knowledge Copilot
          </span>
        </div>

        <h1 style={{
          fontSize: 34, fontWeight: 600, color: 'var(--text-primary)',
          letterSpacing: '-0.04em', lineHeight: 1.15, marginBottom: 16, position: 'relative',
        }}>
          Your documents,<br />
          <span style={{ color: 'var(--accent)' }}>answered instantly.</span>
        </h1>

        <p style={{
          fontSize: 15, color: 'var(--text-secondary)', lineHeight: 1.7,
          fontFamily: 'var(--font-serif)', fontStyle: 'italic',
          maxWidth: 340, position: 'relative',
        }}>
          Upload PDFs, notes, and web pages. Ask questions and get answers grounded in your own knowledge base.
        </p>

        {/* Feature bullets */}
        <div style={{ marginTop: 48, display: 'flex', flexDirection: 'column', gap: 14, position: 'relative' }}>
          {[
            'Retrieval-augmented generation (RAG)',
            'Multi-document search with citations',
            'Streaming responses with source scores',
          ].map((f, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 20, height: 20, borderRadius: 6,
                background: 'var(--accent-soft)', border: '1px solid var(--accent)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
              }}>
                <svg style={{ width: 10, height: 10, color: 'var(--accent)' }}
                  viewBox="0 0 12 12" fill="currentColor">
                  <path d="M10 3L5 8.5 2 5.5l-1 1L5 10.5l6-7-1-0.5z"/>
                </svg>
              </div>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{f}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Right panel — form ── */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center',
        justifyContent: 'center', padding: '40px 48px',
      }}>
        <div style={{ width: '100%', maxWidth: 380 }}>
          <h2 style={{
            fontSize: 24, fontWeight: 600, color: 'var(--text-primary)',
            letterSpacing: '-0.03em', marginBottom: 6,
          }}>
            Welcome back
          </h2>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 32 }}>
            Don't have an account?{' '}
            <Link href="/signup" style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 500 }}>
              Sign up free
            </Link>
          </p>

          {/* Google button */}
          <button
            onClick={handleGoogleLogin}
            style={{
              width: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
              padding: '11px 20px', borderRadius: 12,
              background: 'var(--bg-raised)', border: '1px solid var(--border-med)',
              color: 'var(--text-primary)', fontSize: 14, fontWeight: 500,
              cursor: 'pointer', marginBottom: 20, transition: 'all 0.15s',
              fontFamily: 'var(--font-sans)', boxShadow: 'var(--shadow-sm)',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-strong)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-med)'; e.currentTarget.style.boxShadow = 'var(--shadow-sm)' }}
          >
            <GoogleIcon />
            Continue with Google
          </button>

          {/* Divider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
            <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
            <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>or</span>
            <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
          </div>

          {/* Error message */}
          {error && (
            <div style={{
              marginBottom: 16, padding: '10px 14px', borderRadius: 10,
              background: 'var(--danger-soft)', border: '1px solid var(--danger)',
              color: 'var(--danger)', fontSize: 13, lineHeight: 1.5,
            }}>
              {error}
            </div>
          )}

          {/* Email + password form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{
                fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)',
                display: 'block', marginBottom: 6,
                fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
              }}>
                EMAIL
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                placeholder="you@example.com"
                style={{
                  width: '100%', padding: '10px 14px', borderRadius: 10,
                  background: 'var(--bg-raised)', border: '1.5px solid var(--border-med)',
                  color: 'var(--text-primary)', fontSize: 14,
                  fontFamily: 'var(--font-sans)', outline: 'none',
                  transition: 'border-color 0.15s, box-shadow 0.15s',
                  caretColor: 'var(--accent)',
                }}
                onFocus={focusBorder}
                onBlur={blurBorder}
              />
            </div>

            <div>
              <label style={{
                fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)',
                display: 'block', marginBottom: 6,
                fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
              }}>
                PASSWORD
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  style={{
                    width: '100%', padding: '10px 40px 10px 14px', borderRadius: 10,
                    background: 'var(--bg-raised)', border: '1.5px solid var(--border-med)',
                    color: 'var(--text-primary)', fontSize: 14,
                    fontFamily: 'var(--font-sans)', outline: 'none',
                    transition: 'border-color 0.15s, box-shadow 0.15s',
                    caretColor: 'var(--accent)',
                  }}
                  onFocus={focusBorder}
                  onBlur={blurBorder}
                />
                {/* Show/hide password toggle */}
                <button
                  type="button"
                  onClick={() => setShowPass(p => !p)}
                  style={{
                    position: 'absolute', right: 12, top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--text-muted)', padding: 2, display: 'flex',
                  }}
                >
                  {showPass ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/>
                      <line x1="1" y1="1" x2="23" y2="23"/>
                    </svg>
                  ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                  )}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="accent-btn"
              style={{ marginTop: 6, padding: '12px', borderRadius: 12, fontSize: 14, width: '100%' }}
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p style={{
            marginTop: 20, fontSize: 12, color: 'var(--text-muted)',
            textAlign: 'center', lineHeight: 1.5,
          }}>
            By signing in you agree to our{' '}
            <span style={{ color: 'var(--accent)', cursor: 'pointer' }}>Terms</span>
            {' '}and{' '}
            <span style={{ color: 'var(--accent)', cursor: 'pointer' }}>Privacy Policy</span>.
          </p>
        </div>
      </div>
    </div>
  )
}