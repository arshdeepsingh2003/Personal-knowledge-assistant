// FILE: frontend/hooks/useAuth.js
// PURPOSE: Central auth state. Provides user, login, signup, logout, loginWithClerk.
//          Wrap your app in <AuthProvider> (done in app/layout.js).
//          Use the useAuth() hook in any component to access auth state.
'use client'
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { apiLogin, apiSignup, apiLogout, apiGetMe, apiClerkAuth, setToken, clearToken } from '@/lib/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user,    setUser]    = useState(null)
  const [loading, setLoading] = useState(true)

  // On mount: try to restore a previous session from sessionStorage
  useEffect(() => {
    async function restore() {
      try {
        const flag  = sessionStorage.getItem('kc_authed')
        const token = sessionStorage.getItem('kc_token')
        if (!flag || !token) { setLoading(false); return }

        // Restore token into the api.js in-memory variable
        setToken(token)

        // Validate the token is still good by fetching /auth/me
        const me = await apiGetMe()
        setUser(me)
      } catch {
        // Token expired or invalid — clear everything
        sessionStorage.removeItem('kc_authed')
        sessionStorage.removeItem('kc_token')
        clearToken()
      } finally {
        setLoading(false)
      }
    }
    restore()
  }, [])

  // Store token in sessionStorage so it survives page refreshes
  // (sessionStorage is cleared when the browser tab closes — more secure than localStorage)
  function persistToken(token) {
    setToken(token)
    sessionStorage.setItem('kc_token',  token)
    sessionStorage.setItem('kc_authed', '1')
  }

  const login = useCallback(async (email, password) => {
    const data = await apiLogin({ email, password })
    persistToken(data.access_token)
    setUser(data.user)
    return data.user
  }, [])

  const signup = useCallback(async (email, password, name) => {
    try {
      const data = await apiSignup({ email, password, name })
      persistToken(data.access_token)
      setUser(data.user)
      return data.user
    } catch (err) {
      throw err
    }
  }, [])

  // Called from sso-callback/page.js after Google OAuth via Clerk
  const loginWithClerk = useCallback(async (clerkToken) => {
    const data = await apiClerkAuth(clerkToken)
    persistToken(data.access_token)
    setUser(data.user)
    return data.user
  }, [])

  const logout = useCallback(async () => {
    await apiLogout()
    clearToken()
    sessionStorage.removeItem('kc_token')
    sessionStorage.removeItem('kc_authed')
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, loginWithClerk, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// Use this hook inside any component to get auth state
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth() must be used inside <AuthProvider>')
  return ctx
}