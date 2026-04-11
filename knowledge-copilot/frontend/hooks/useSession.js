'use client'
import { useState, useEffect } from 'react'
import { createSession, getSession } from '@/lib/api'

const SESSION_KEY = 'kc_session_id'

export function useSession() {
  const [sessionId, setSessionId] = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [mounted,   setMounted]   = useState(false)   // ← new

  useEffect(() => {
    setMounted(true)   // signals: we are now on the client

    async function init() {
      const stored = localStorage.getItem(SESSION_KEY)
      if (stored) {
        const session = await getSession(stored)
        if (session) {
          setSessionId(stored)
          setLoading(false)
          return
        }
      }
      const { session_id } = await createSession()
      localStorage.setItem(SESSION_KEY, session_id)
      setSessionId(session_id)
      setLoading(false)
    }
    init()
  }, [])

  async function resetSession() {
    const { session_id } = await createSession()
    localStorage.setItem(SESSION_KEY, session_id)
    setSessionId(session_id)
    return session_id
  }

  return { sessionId, loading: loading || !mounted, resetSession }
}