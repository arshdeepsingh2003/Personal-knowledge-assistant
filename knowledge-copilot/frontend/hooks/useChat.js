'use client'
import { useState, useRef, useCallback, useEffect } from 'react'
import { askStreaming, getMessages } from '@/lib/api'

function cacheKey(sessionId) {
  return `kc_msgs_${sessionId}`
}

function saveCachedMessages(sessionId, messages) {
  if (!sessionId) return
  try {
    localStorage.setItem(cacheKey(sessionId), JSON.stringify(messages))
  } catch {}
}

export function useChat(sessionId, options = {}) {
  const { onMessageComplete } = options
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const sessionIdRef = useRef(sessionId)
  const loadingRef = useRef(null)

  useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return

    const session = sessionId
    loadingRef.current = session

    setMessages([])
    setError(null)
    setThinking(false)

    async function load() {
      try {
        const result = await getMessages(session)
        if (loadingRef.current !== session) return

        const msgs = (result?.messages ?? []).map(m => ({
          role: m.role,
          content: m.content,
          timestamp: m.created_at,
          sources: m.sources || [],
        }))

        setMessages(msgs)
        saveCachedMessages(session, msgs)
      } catch {
        if (loadingRef.current === session) {
          setMessages([])
        }
      }
    }

    load()
  }, [sessionId])

  const sendMessage = useCallback(async (query, options = {}) => {
    if (!sessionId || !query.trim()) return

    const userMsg = { role: 'user', content: query, timestamp: new Date().toISOString() }
    const assistantMsg = { role: 'assistant', content: '', sources: [], streaming: true, id: Date.now() }

    setMessages(prev => {
      const updated = [...prev, userMsg, assistantMsg]
      saveCachedMessages(sessionId, updated)
      return updated
    })
    setThinking(true)
    setError(null)

    abortRef.current = new AbortController()

    try {
      await askStreaming(sessionId, query, {
        signal: abortRef.current.signal,
        searchMode: options.searchMode,
        documentId: options.documentId,
        onSources(sources) {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1], sources }
            next[next.length - 1] = last
            saveCachedMessages(sessionId, next)
            return next
          })
        },
        onToken(token) {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.content += token
            next[next.length - 1] = last
            saveCachedMessages(sessionId, next)
            return next
          })
        },
        onDone() {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1], streaming: false }
            next[next.length - 1] = last
            saveCachedMessages(sessionId, next)
            return next
          })
          setThinking(false)
          onMessageComplete?.()
        },
      })
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message)
      setThinking(false)
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1], streaming: false, error: err.message }
        next[next.length - 1] = last
        saveCachedMessages(sessionId, next)
        return next
      })
      onMessageComplete?.()
    }
  }, [sessionId])

  function clearMessages() {
    abortRef.current?.abort()
    setMessages([])
    setError(null)
    setThinking(false)
    if (sessionId) {
      localStorage.removeItem(cacheKey(sessionId))
    }
  }

  return { messages, thinking, error, sendMessage, clearMessages }
}
