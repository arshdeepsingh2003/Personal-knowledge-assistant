// FILE: frontend/lib/api.js
// PURPOSE: All network calls to the backend in one place.
//          Token lives in a module-level variable (_token) — not in localStorage.
//          This is the safest client-side option for JWTs in an SPA.

const ROOT = process.env.NEXT_PUBLIC_API_URL 
const BASE = `${ROOT}/api/v1`
const AUTH_BASE = ROOT

// ── In-memory token store ─────────────────────────────────────────────────────
// The token never touches localStorage (XSS readable).
// It lives here in JS module memory — gone on hard refresh, persisted in
// sessionStorage (see useAuth.js) and re-loaded on mount.
let _token = null

export function setToken(token)  { _token = token }
export function getToken()       { return _token  }
export function clearToken()     { _token = null   }

// ── Helpers ───────────────────────────────────────────────────────────────────
function jsonHeaders() {
  const h = { 'Content-Type': 'application/json' }
  if (_token) h['Authorization'] = `Bearer ${_token}`
  return h
}

function authOnlyHeaders() {
  const h = {}
  if (_token) h['Authorization'] = `Bearer ${_token}`
  return h
}

async function handleResponse(res) {
  if (res.ok) return res.json()
  const body = await res.json().catch(() => ({}))
  throw new Error(body?.detail ?? body?.error?.message ?? `HTTP ${res.status}`)
}

// ── Auth endpoints ────────────────────────────────────────────────────────────

export async function apiSignup({ email, password, name }) {
  const res = await fetch(`${AUTH_BASE}/auth/signup`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ email, password, name }),
  })
  return handleResponse(res)
}

export async function apiLogin({ email, password }) {
  const res = await fetch(`${AUTH_BASE}/auth/login`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ email, password }),
  })
  return handleResponse(res)
}

export async function apiClerkAuth(clerkSessionToken) {
  const res = await fetch(`${AUTH_BASE}/auth/clerk`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ clerk_session_token: clerkSessionToken }),
  })
  return handleResponse(res)
}

export async function apiGetMe() {
  const res = await fetch(`${AUTH_BASE}/auth/me`, { headers: jsonHeaders() })
  return handleResponse(res)
}

export async function apiLogout() {
  await fetch(`${AUTH_BASE}/auth/logout`, {
    method: 'POST', headers: jsonHeaders(),
  }).catch(() => {})
  clearToken()
}

// ── RAG / chat endpoints (require auth header) ────────────────────────────────

export async function createSession() {
  const res = await fetch(`${BASE}/sessions`, {
    method: 'POST', headers: jsonHeaders(),
  })
  return handleResponse(res)
}

export async function getSession(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}`, { headers: jsonHeaders() })
  if (!res.ok) return null
  return res.json()
}

export async function uploadDocument(file, options = {}) {
  const form = new FormData()
  form.append('file',          file)
  form.append('chunk_size',    options.chunkSize    ?? 1000)
  form.append('chunk_overlap', options.chunkOverlap ?? 200)
  form.append('strategy',      options.strategy     ?? 'recursive')

  const res = await fetch(`${BASE}/documents`, {
    method:  'POST',
    headers: authOnlyHeaders(),   // no Content-Type — browser sets multipart boundary
    body:    form,
  })
  return handleResponse(res)
}

export async function askStreaming(
  sessionId, query,
  { onToken, onSources, onDone, signal } = {}
) {
  const res = await fetch(`${BASE}/ask`, {
    method:  'POST',
    headers: jsonHeaders(),
    body:    JSON.stringify({
      session_id:      sessionId,
      query,
      k:               5,
      score_threshold: 0.3,
      stream:          true,
    }),
    signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err?.detail ?? err?.error?.message ?? 'Request failed')
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const event = JSON.parse(raw)
        if (event.type === 'sources') onSources?.(event.sources)
        if (event.type === 'token')   onToken?.(event.content)
        if (event.type === 'done')    onDone?.()
      } catch { /* malformed SSE line — skip */ }
    }
  }
}

export async function getDocumentStatus() {
  const res = await fetch(`${BASE}/documents/status`, { headers: jsonHeaders() })
  return handleResponse(res)
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}