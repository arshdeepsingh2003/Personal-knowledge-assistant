'use client'
import { useState, useRef, useCallback } from 'react'
import { askStreaming } from '@/lib/api'

/*
manages a chat system with streaming responses

It handles:

sending user messages
receiving AI response token by token (streaming)
showing loading state
handling errors
stopping requests
*/

export function useChat(sessionId) {
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false) //AI is typing…
  const [error,    setError]    = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (!sessionId || !query.trim()) return

    // Add user message immediately in ui
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setThinking(true)
    setError(null)

    // Placeholder for streaming assistant message
    const assistantIndex = messages.length + 1
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [], streaming: true }])

    abortRef.current = new AbortController()

    try {
      await askStreaming(sessionId, query, {
        signal: abortRef.current.signal,

        onSources(sources) {
          setMessages(prev => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.sources = sources
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
            const last = { ...next[next.length - 1] }
            last.streaming = false
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
      // Replace the empty assistant placeholder with error state
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.content  = ''
        last.error    = err.message
        last.streaming = false
        next[next.length - 1] = last
        return next
      })
    }
  }, [sessionId, messages.length])

  function clearMessages() {
    abortRef.current?.abort()
    setMessages([])
    setError(null)
    setThinking(false)
  }

  return { messages, thinking, error, sendMessage, clearMessages }
}