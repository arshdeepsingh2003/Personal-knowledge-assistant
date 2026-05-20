'use client'
import { useState, useEffect, useCallback } from 'react'
import { createSession, getSession, listSessions, renameSession as apiRename, deleteSession as apiDelete } from '@/lib/api'

const SESSION_KEY = 'kc_session_id'
const SESSIONS_LIST_KEY = 'kc_session_ids'

export function useSession() {
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [mounted, setMounted] = useState(false)

  function createLocalSession() {
    const id = crypto.randomUUID?.() ?? Date.now().toString(36) + Math.random().toString(36).slice(2, 10)
    return { session_id: id }
  }

  useEffect(() => {
    setMounted(true)
    async function init() {
      try {
        const stored = localStorage.getItem(SESSION_KEY)
        let ids = JSON.parse(localStorage.getItem(SESSIONS_LIST_KEY) || '[]')

        const apiResult = await listSessions()
        const apiSessions = apiResult?.sessions ?? []

        const apiIds = apiSessions.map(s => s.id)
        ids = [...new Set([...ids, ...apiIds])]
        localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(ids))

        setSessions(apiSessions)

        if (stored && ids.includes(stored)) {
          const session = await getSession(stored)
          if (session) {
            setSessionId(stored)
            setLoading(false)
            return
          }
        }

        if (apiSessions.length > 0) {
          const latest = apiSessions[0]
          localStorage.setItem(SESSION_KEY, latest.id)
          setSessionId(latest.id)
          setLoading(false)
          return
        }

        const { session_id } = await createSession().catch(() => createLocalSession())
        localStorage.setItem(SESSION_KEY, session_id)
        ids = [session_id, ...ids.filter(id => id !== session_id)]
        localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(ids))
        setSessionId(session_id)
        setSessions([{ id: session_id, title: 'New conversation', created_at: new Date().toISOString(), message_count: 0 }])
      } catch (e) {
        console.error('Session init failed', e)
        const fallback = createLocalSession()
        localStorage.setItem(SESSION_KEY, fallback.session_id)
        setSessionId(fallback.session_id)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  async function resetSession() {
    const { session_id } = await createSession().catch(() => createLocalSession())
    localStorage.setItem(SESSION_KEY, session_id)
    const ids = JSON.parse(localStorage.getItem(SESSIONS_LIST_KEY) || '[]')
    const updated = [session_id, ...ids.filter(id => id !== session_id)]
    localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(updated))
    setSessionId(session_id)
    setSessions(prev => [{ id: session_id, title: 'New conversation', created_at: new Date().toISOString(), message_count: 0 }, ...prev])
    return session_id
  }

  async function switchSession(id) {
    localStorage.setItem(SESSION_KEY, id)
    setSessionId(id)
  }

  function removeSessionFromList(id) {
    const ids = JSON.parse(localStorage.getItem(SESSIONS_LIST_KEY) || '[]')
    localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(ids.filter(sid => sid !== id)))
    setSessions(prev => prev.filter(s => s.id !== id))
    localStorage.removeItem(`kc_msgs_${id}`)
  }

  async function renameSession(id, title) {
    try {
      const result = await apiRename(id, title)
      setSessions(prev => prev.map(s => s.id === id ? { ...s, title: result.title } : s))
    } catch (e) {
      console.error('Rename failed', e)
    }
  }

  async function deleteConversation(id) {
    try {
      await apiDelete(id)
      removeSessionFromList(id)
      if (sessionId === id) {
        const remaining = sessions.filter(s => s.id !== id)
        if (remaining.length > 0) {
          switchSession(remaining[0].id)
        } else {
          resetSession()
        }
      }
    } catch (e) {
      console.error('Delete failed', e)
    }
  }

  async function refreshSessions() {
    try {
      const result = await listSessions()
      setSessions(result?.sessions ?? [])
    } catch {}
  }

  return {
    sessionId,
    sessions,
    loading: loading || !mounted,
    resetSession,
    switchSession,
    removeSessionFromList,
    renameSession,
    deleteConversation,
    refreshSessions,
  }
}
