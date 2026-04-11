/*
This file is acting like a middleman between your UI and backend.

👉 Instead of writing fetch() everywhere in your app, you:

centralize all API calls here
reuse functions like createSession(), askStreaming() etc.
*/


const BASE = process.env.NEXT_PUBLIC_API_URL

if (!BASE) {
  throw new Error("NEXT_PUBLIC_API_URL is not defined in .env")
}

// ERROR HANDLING:Extracts error message from backend response
async function parseError(res) {
  const err = await res.json().catch(() => ({}))
  return err?.detail ?? err?.error?.message ?? 'Request failed'
}

// Sessions APIs

export async function createSession() {
  const res = await fetch(`${BASE}/sessions`, { method: 'POST' })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

// restore chat history
export async function getSession(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}`)
  if (!res.ok) return null
  return res.json()
}

// Document Upload System

export async function uploadDocument(file, options = {}) {
  const form = new FormData()
  form.append('file', file)
  form.append('chunk_size', String(options.chunkSize ?? 1000))
  form.append('chunk_overlap', String(options.chunkOverlap ?? 200))
  form.append('strategy', options.strategy ?? 'recursive')

  const res = await fetch(`${BASE}/documents`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    throw new Error(await parseError(res))
  }

  return res.json()
}

export async function getDocumentStatus() {
  const res = await fetch(`${BASE}/documents/status`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

// Streaming Version (Real-time AI): Ask (streaming) 
export async function askStreaming(
  sessionId,
  query,
  { onToken, onSources, onDone, signal } = {}
) {
  let res

  try {
    res = await fetch(`${BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        k: 5,
        score_threshold: 0.3,
        stream: true,
      }),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') {
      console.log('Request cancelled')
      return
    }
    throw err
  }

  if (!res.ok) {
    throw new Error(await parseError(res))
  }

  if (!res.body) {
    throw new Error('Streaming not supported or empty response body')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()

  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop()

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue

      const raw = line.slice(6).trim()
      if (!raw) continue

      try {
        const event = JSON.parse(raw)

        if (event.type === 'sources') onSources?.(event.sources)
        if (event.type === 'token') onToken?.(event.content)
        if (event.type === 'done') onDone?.()
      } catch (e) {
        console.warn('Invalid SSE chunk:', raw)
      }
    }
  }
}

// Ask (blocking fallback) : waits full response

export async function askBlocking(sessionId, query) {
  const res = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      query,
      k: 5,
      score_threshold: 0.3,
      stream: false,
    }),
  })

  if (!res.ok) {
    throw new Error(await parseError(res))
  }

  return res.json()
}

// Health

export async function getHealth() {
  try {
    const res = await fetch(`${BASE}/health`)
    if (!res.ok) return { status: 'down' }
    return res.json()
  } catch {
    return { status: 'down' }
  }
}