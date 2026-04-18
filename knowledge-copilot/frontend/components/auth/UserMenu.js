// FILE: frontend/components/auth/UserMenu.js
// PURPOSE: Avatar button in the top-right of the dashboard header.
//          Opens a dropdown with the user's name, email, auth provider badge,
//          and a Sign out button.
'use client'
import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'

export default function UserMenu() {
  const { user, logout } = useAuth()
  const router           = useRouter()
  const [open, setOpen]  = useState(false)
  const menuRef          = useRef(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  async function handleLogout() {
    setOpen(false)
    await logout()
    // Clear the session presence cookie
    document.cookie = 'kc_session=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT'
    router.replace('/login')
  }

  if (!user) return null

  // Build initials from name, fallback to first letter of email
  const initials = user.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user.email[0].toUpperCase()

  return (
    <div ref={menuRef} style={{ position: 'relative' }}>
      {/* Avatar button */}
      <button
        onClick={() => setOpen(o => !o)}
        title={user.name || user.email}
        style={{
          width: 32, height: 32, borderRadius: 10, padding: 0,
          background: user.avatar_url ? 'transparent' : 'var(--accent)',
          border: `1px solid ${open ? 'var(--accent)' : 'var(--border-med)'}`,
          cursor: 'pointer', overflow: 'hidden',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: 'box-shadow 0.15s, border-color 0.15s',
          boxShadow: open ? '0 0 0 3px var(--accent-glow)' : 'none',
        }}
      >
        {user.avatar_url ? (
          <img src={user.avatar_url} alt={user.name}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <span style={{ fontSize: 12, fontWeight: 600, color: '#fff',
            fontFamily: 'var(--font-sans)' }}>
            {initials}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className="animate-scale-in"
          style={{
            position: 'absolute', top: 'calc(100% + 8px)', right: 0,
            width: 228,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-med)',
            borderRadius: 14, overflow: 'hidden',
            boxShadow: 'var(--shadow-lg)', zIndex: 200,
          }}
        >
          {/* User info block */}
          <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)' }}>
            <p style={{
              fontSize: 13, fontWeight: 500, color: 'var(--text-primary)',
              marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {user.name}
            </p>
            <p style={{
              fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {user.email}
            </p>
            {/* Auth provider badge */}
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4, marginTop: 7,
              fontSize: 10, padding: '2px 8px', borderRadius: 20,
              background: 'var(--accent-soft)', border: '1px solid var(--accent)',
              color: 'var(--accent)', fontFamily: 'var(--font-mono)',
            }}>
              {user.auth_provider === 'clerk' ? '🔑 Google' : '✉ Email'}
            </span>
          </div>

          {/* Menu items */}
          <div style={{ padding: '6px' }}>
            <button
              onClick={() => { setOpen(false); router.push('/dashboard') }}
              style={{
                width: '100%', padding: '9px 12px', borderRadius: 8,
                background: 'none', border: 'none',
                color: 'var(--text-secondary)', fontSize: 13, textAlign: 'left',
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 9,
                fontFamily: 'var(--font-sans)', transition: 'background 0.12s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-raised)'}
              onMouseLeave={e => e.currentTarget.style.background = 'none'}
            >
              <svg style={{ width: 14, height: 14, color: 'var(--text-muted)' }}
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
              Dashboard
            </button>

            <div style={{ height: 1, background: 'var(--border)', margin: '4px 6px' }} />

            <button
              onClick={handleLogout}
              style={{
                width: '100%', padding: '9px 12px', borderRadius: 8,
                background: 'none', border: 'none',
                color: 'var(--danger)', fontSize: 13, textAlign: 'left',
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 9,
                fontFamily: 'var(--font-sans)', transition: 'background 0.12s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--danger-soft)'}
              onMouseLeave={e => e.currentTarget.style.background = 'none'}
            >
              <svg style={{ width: 14, height: 14 }} viewBox="0 0 24 24"
                fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
              Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}