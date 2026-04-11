'use client'
import { useState, useRef, useCallback } from 'react'
import { askStreaming } from '@/lib/api'

export function useChat(sessionId) {
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false)
  const [error,    setError]    = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (!sessionId || !query.trim()) return

    setMessages(prev => [
      ...prev,
      { role: 'user', content: query },
      { role: 'assistant', content: '', sources: [], streaming: true, id: Date.now() },
    ])
    setThinking(true)
    setError(null)

    abortRef.current = new AbortController()

    try {
      await askStreaming(sessionId, query, {
        signal: abortRef.current.signal,
        onSources(sources) {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1], sources }
            next[next.length - 1] = last
            return next
          })
        },
        onToken(token) {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.content += token
            next[next.length - 1] = last
            return next
          })
        },
        onDone() {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1], streaming: false }
            next[next.length - 1] = last
            return next
          })
          setThinking(false)
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
        return next
      })
    }
  }, [sessionId])

  function clearMessages() {
    abortRef.current?.abort()
    setMessages([])
    setError(null)
    setThinking(false)
  }

  return { messages, thinking, error, sendMessage, clearMessages }
}