// FILE: frontend/app/signup/page.js
// PURPOSE: The /signup route — name + email + password registration form.
'use client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/hooks/useAuth'

function PasswordStrengthMeter({ password }) {
  const checks = [
    { label: '8-16 chars', ok: password.length >= 8 && password.length <= 16 },
    { label: 'Uppercase',   ok: /[A-Z]/.test(password) },
    { label: 'Lowercase',   ok: /[a-z]/.test(password) },
    { label: 'Special',     ok: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password) },
  ]
  if (!password) return null
  return (
    <div style={{ display: 'flex', gap: 8, marginTop: 7, flexWrap: 'wrap' }}>
      {checks.map(c => (
        <span key={c.label} style={{
          fontSize: 11, fontFamily: 'var(--font-mono)',
          color: c.ok ? 'var(--success)' : 'var(--text-muted)',
          display: 'flex', alignItems: 'center', gap: 3,
        }}>
          {c.ok ? '✓' : '○'} {c.label}
        </span>
      ))}
    </div>
  )
}

export default function SignupPage() {
  const router = useRouter()
  const { signup, user, loading: authLoading } = useAuth()

  const [name,     setName]     = useState('')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [mounted,  setMounted]  = useState(false)

  useEffect(() => { setMounted(true) }, [])
  useEffect(() => {
    if (!authLoading && user) router.replace('/dashboard')
  }, [user, authLoading, router])

  function focusBorder(e) {
    e.target.style.borderColor = 'var(--accent)'
    e.target.style.boxShadow   = '0 0 0 3px var(--accent-glow)'
  }
  function blurBorder(e) {
    e.target.style.borderColor = 'var(--border-med)'
    e.target.style.boxShadow   = 'none'
  }

  const fieldStyle = {
    width: '100%', padding: '10px 14px', borderRadius: 10,
    background: 'var(--bg-raised)', border: '1.5px solid var(--border-med)',
    color: 'var(--text-primary)', fontSize: 14,
    fontFamily: 'var(--font-sans)', outline: 'none',
    transition: 'border-color 0.15s, box-shadow 0.15s',
    caretColor: 'var(--accent)',
  }

  const labelStyle = {
    fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)',
    display: 'block', marginBottom: 6,
    fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    if (password.length < 8 || password.length > 16) {
      setError('Password must be 8-16 characters')
      return
    }
    if (!/[A-Z]/.test(password)) {
      setError('Password must contain at least one uppercase letter')
      return
    }
    if (!/[a-z]/.test(password)) {
      setError('Password must contain at least one lowercase letter')
      return
    }
    if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
      setError('Password must contain at least one special character')
      return
    }
    setLoading(true)
    try {
      const user = await signup(email.trim(), password, name.trim())
      document.cookie = 'kc_session=1; path=/; SameSite=Lax'
      if (user && user.id) {
        router.replace('/dashboard')
      } else {
        router.replace('/login')
      }
    } catch (err) {
      setError(err.message || 'Signup failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  if (!mounted || authLoading) return null

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg-base)', padding: '40px 24px',
      fontFamily: 'var(--font-sans)',
    }}>
      {/* Background glow blobs */}
      <div style={{
        position: 'fixed', top: '10%', right: '15%', width: 300, height: 300,
        borderRadius: '50%',
        background: 'radial-gradient(circle, var(--accent-glow) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />
      <div style={{
        position: 'fixed', bottom: '10%', left: '10%', width: 200, height: 200,
        borderRadius: '50%',
        background: 'radial-gradient(circle, var(--accent-glow) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Card */}
      <div style={{
        width: '100%', maxWidth: 420,
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-med)',
        borderRadius: 20, padding: '40px 36px',
        boxShadow: 'var(--shadow-lg)', position: 'relative',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 32 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 9, background: 'var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: 'var(--shadow-accent)',
          }}>
            <svg style={{ width: 15, height: 15, color: '#fff' }}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            Knowledge Copilot
          </span>
        </div>

        <h2 style={{ fontSize: 22, fontWeight: 600, color: 'var(--text-primary)',
          letterSpacing: '-0.03em', marginBottom: 6 }}>
          Create an account
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 28 }}>
          Already have one?{' '}
          <Link href="/login" style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 500 }}>
            Sign in
          </Link>
        </p>

        {error && (
          <div style={{
            marginBottom: 16, padding: '10px 14px', borderRadius: 10,
            background: 'var(--danger-soft)', border: '1px solid var(--danger)',
            color: 'var(--danger)', fontSize: 13,
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div>
            <label style={labelStyle}>FULL NAME</label>
            <input
              type="text" value={name}
              onChange={e => setName(e.target.value)}
              required placeholder="Jane Smith"
              style={fieldStyle} onFocus={focusBorder} onBlur={blurBorder}
            />
          </div>

          <div>
            <label style={labelStyle}>EMAIL</label>
            <input
              type="email" value={email}
              onChange={e => setEmail(e.target.value)}
              required placeholder="you@example.com"
              style={fieldStyle} onFocus={focusBorder} onBlur={blurBorder}
            />
          </div>

          <div>
            <label style={labelStyle}>PASSWORD</label>
            <input
              type="password" value={password}
              onChange={e => setPassword(e.target.value)}
              required placeholder="Min. 8, Max. 16 chars (upper, lower, special)"
              style={fieldStyle} onFocus={focusBorder} onBlur={blurBorder}
            />
            <PasswordStrengthMeter password={password} />
          </div>

          <button
            type="submit" disabled={loading}
            className="accent-btn"
            style={{ marginTop: 6, padding: '12px', borderRadius: 12, fontSize: 14, width: '100%' }}
          >
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  )
}