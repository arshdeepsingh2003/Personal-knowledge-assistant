// FILE: frontend/app/dashboard/page.js
// PURPOSE: The /dashboard route — the main chat interface.
//          Wrapped in ProtectedRoute so unauthenticated users
//          are redirected to /login automatically.
'use client'
import { useEffect, useRef, useState } from 'react'
import { useSession }     from '@/hooks/useSession'
import { useChat }        from '@/hooks/useChat'
import { useUpload }      from '@/hooks/useUpload'
import { useTheme }       from '@/hooks/useTheme'
import { useAuth }        from '@/hooks/useAuth'
import ProtectedRoute     from '@/components/auth/ProtectedRoute'
import UserMenu           from '@/components/auth/UserMenu'
import UploadZone         from '@/components/UploadZone'
import FileList           from '@/components/FileList'
import Message            from '@/components/Message'
import ChatInput          from '@/components/ChatInput'
import ThemeToggle        from '@/components/ThemeToggle'

const STARTERS = [
  'Summarise this document in 3 key points',
  'What are the main conclusions?',
  'List any important dates or figures mentioned',
  'What questions does this document answer?',
]

// ── Inner component (rendered only when user is authenticated) ─────────────
function DashboardContent() {
  const { user }                                                            = useAuth()
  const { sessionId, loading, resetSession, sessions, switchSession, removeSessionFromList, renameSession, deleteConversation, refreshSessions } = useSession()
  const { messages, thinking, sendMessage, clearMessages }                  = useChat(sessionId, { onMessageComplete: refreshSessions })
  const { files, uploading, error: uploadError, upload, removeFile }        = useUpload()
  const { theme, toggle: toggleTheme, mounted: themeMounted }               = useTheme()
  const bottomRef   = useRef(null)
  const hasMessages = messages.length > 0
  const hasFiles    = files.some(f => f.status === 'indexed')

  // Sync theme class to <html>
  useEffect(() => {
    if (!themeMounted) return
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme, themeMounted])

  // Auto-scroll to newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleNewChat() {
    clearMessages()
    await resetSession()
    refreshSessions()
  }

  const [renamingId, setRenamingId] = useState(null)
  const [renameValue, setRenameValue] = useState('')
  const [menuOpenId, setMenuOpenId] = useState(null)
  const renameInputRef = useRef(null)

  useEffect(() => {
    if (renamingId) renameInputRef.current?.focus()
  }, [renamingId])

  useEffect(() => {
    function handleClick(e) {
      if (!e.target.closest('.conv-menu')) setMenuOpenId(null)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  function handleStartRename(s) {
    setRenamingId(s.id)
    setRenameValue(s.title || '')
    setMenuOpenId(null)
  }

  async function handleFinishRename(id) {
    const trimmed = renameValue.trim()
    if (trimmed && trimmed !== sessions.find(s => s.id === id)?.title) {
      await renameSession(id, trimmed)
    }
    setRenamingId(null)
    setRenameValue('')
  }

  function handleDelete(id) {
    setMenuOpenId(null)
    deleteConversation(id)
  }

  // ── Session loading spinner ──────────────────────────────────────
  if (loading) {
    return (
      <div suppressHydrationWarning style={{
        height: '100vh', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-base)', gap: 14,
      }}>
        <div style={{
          width: 42, height: 42, borderRadius: 13,
          background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: 'var(--shadow-accent)',
        }}>
          <svg style={{ width: 20, height: 20, color: 'var(--accent)',
            animation: 'spin-slow 1.1s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
          initialising…
        </p>
      </div>
    )
  }

  // ── Main layout ──────────────────────────────────────────────────
  return (
    <div style={{ height: '100vh', display: 'flex', overflow: 'hidden',
      background: 'var(--bg-base)' }}>

      {/* ══════ SIDEBAR ══════ */}
      <aside className="sidebar" style={{
        width: 264, flexShrink: 0, display: 'flex', flexDirection: 'column',
        background: 'var(--bg-surface)', borderRight: '1px solid var(--border-med)',
        overflow: 'hidden', boxShadow: 'var(--shadow-md)', position: 'relative', zIndex: 2,
      }}>
        {/* Brand row */}
        <div style={{
          padding: '18px 18px 14px', borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8, background: 'var(--accent)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0, boxShadow: 'var(--shadow-accent)',
            }}>
              <svg style={{ width: 14, height: 14, color: '#fff' }}
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <div>
              <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)',
                letterSpacing: '-0.02em', lineHeight: 1.2 }}>
                Knowledge Copilot
              </p>
              <p style={{ fontSize: 10, color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)', marginTop: 1 }}>
                RAG assistant
              </p>
            </div>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>

        {/* Sidebar content */}
        <div style={{ padding: '14px 16px 0', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 18 }}>
          
          {/* Documents section */}
          <div>
            <p style={{
              fontSize: 9.5, fontWeight: 600, letterSpacing: '0.09em',
              textTransform: 'uppercase', color: 'var(--text-muted)',
              marginBottom: 10, fontFamily: 'var(--font-mono)',
            }}>
              Documents
            </p>
            <UploadZone onUpload={upload} uploading={uploading} />
            {uploadError && (
              <div style={{
                marginTop: 8, padding: '7px 10px', borderRadius: 8,
                background: 'var(--danger-soft)', border: '1px solid var(--danger)',
                fontSize: 11, color: 'var(--danger)', lineHeight: 1.4,
              }}>
                ⚠ {uploadError}
              </div>
            )}
            <FileList files={files} onRemove={removeFile} />
          </div>

          {/* Recent Conversations section */}
          <div style={{
            paddingTop: files.length > 0 ? 14 : 0,
            borderTop: files.length > 0 ? '1px solid var(--border)' : 'none',
            flex: sessions.length > 0 ? 1 : 0,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <p style={{
              fontSize: 9.5, fontWeight: 600, letterSpacing: '0.09em',
              textTransform: 'uppercase', color: 'var(--text-muted)',
              marginBottom: 10, fontFamily: 'var(--font-mono)',
            }}>
              Recent Conversations
            </p>
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 4,
              overflowY: 'auto', flex: 1,
            }}>
              {sessions.map(s => {
                const isCurrent = s.id === sessionId
                const isRenaming = renamingId === s.id
                const isMenuOpen = menuOpenId === s.id
                let title = s.title
                if (!title) {
                  try {
                    const cached = JSON.parse(localStorage.getItem(`kc_msgs_${s.id}`) || '[]')
                    const first = cached.find(m => m.role === 'user')
                    if (first) {
                      title = first.content.length > 48
                        ? first.content.slice(0, 48) + '…'
                        : first.content
                    }
                  } catch {}
                }
                if (!title) {
                  title = s.created_at
                    ? new Date(s.created_at).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
                    : 'Conversation'
                }
                const dateLabel = s.created_at
                  ? new Date(s.created_at).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) + ' · ' + new Date(s.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  : ''
                return (
                  <div key={s.id} style={{ position: 'relative' }}>
                    {isRenaming ? (
                      <div style={{
                        display: 'flex', gap: 4, padding: '6px 8px', borderRadius: 10,
                        background: 'var(--accent-soft)', border: '1px solid var(--accent)',
                      }}>
                        <input
                          ref={renameInputRef}
                          value={renameValue}
                          onChange={e => setRenameValue(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') handleFinishRename(s.id)
                            if (e.key === 'Escape') setRenamingId(null)
                          }}
                          onBlur={() => handleFinishRename(s.id)}
                          style={{
                            flex: 1, border: 'none', outline: 'none', fontSize: 12,
                            background: 'transparent', color: 'var(--text-primary)',
                            fontFamily: 'inherit', padding: 0,
                          }}
                          maxLength={200}
                        />
                        <button onClick={() => handleFinishRename(s.id)}
                          style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'var(--accent)', padding: 0, fontSize: 14 }}>
                          ↵
                        </button>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={() => switchSession(s.id)}
                          className="conversation-item animate-fade-in"
                          style={{
                            display: 'flex', alignItems: 'flex-start', gap: 8,
                            padding: '8px 10px', borderRadius: 10, width: '100%',
                            background: isCurrent ? 'var(--accent-soft)' : 'transparent',
                            border: `1px solid ${isCurrent ? 'var(--accent)' : 'transparent'}`,
                            color: isCurrent ? 'var(--accent)' : 'var(--text-secondary)',
                            fontSize: 12, textAlign: 'left', cursor: 'pointer',
                            transition: 'all 0.18s ease',
                            fontWeight: isCurrent ? 500 : 400,
                          }}
                          onMouseEnter={e => {
                            if (!isCurrent) {
                              e.currentTarget.style.background = 'var(--bg-raised)'
                              e.currentTarget.style.borderColor = 'var(--border-med)'
                              e.currentTarget.style.color = 'var(--text-primary)'
                            }
                          }}
                          onMouseLeave={e => {
                            if (!isCurrent) {
                              e.currentTarget.style.background = 'transparent'
                              e.currentTarget.style.borderColor = 'transparent'
                              e.currentTarget.style.color = 'var(--text-secondary)'
                            }
                          }}
                        >
                          <svg style={{ width: 14, height: 14, color: isCurrent ? 'var(--accent)' : 'var(--text-muted)', flexShrink: 0, marginTop: 2 }}
                            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                          </svg>
                          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, lineHeight: 1.3 }}>
                              {title}
                            </span>
                            <span style={{ fontSize: 9.5, color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
                              {dateLabel}
                            </span>
                          </div>
                        </button>
                        <div className="conv-menu" style={{ position: 'absolute', right: 4, top: 4, zIndex: 5 }}>
                          <button
                            onClick={e => { e.stopPropagation(); setMenuOpenId(isMenuOpen ? null : s.id) }}
                            style={{
                              border: 'none', background: 'none', cursor: 'pointer',
                              padding: '2px 4px', borderRadius: 6, color: 'var(--text-muted)',
                              fontSize: 14, lineHeight: 1, opacity: 0.6,
                            }}
                            onMouseEnter={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = 'var(--bg-raised)' }}
                            onMouseLeave={e => { e.currentTarget.style.opacity = '0.6'; e.currentTarget.style.background = 'transparent' }}
                          >⋯</button>
                          {isMenuOpen && (
                            <div style={{
                              position: 'absolute', right: 0, top: '100%', zIndex: 100,
                              background: 'var(--bg-surface)', border: '1px solid var(--border-med)',
                              borderRadius: 8, boxShadow: 'var(--shadow-lg)',
                              padding: 4, minWidth: 130, marginTop: 2,
                              display: 'flex', flexDirection: 'column', gap: 1,
                            }}>
                              <button onClick={() => handleStartRename(s)}
                                style={{
                                  display: 'flex', alignItems: 'center', gap: 7,
                                  width: '100%', padding: '8px 12px', border: 'none', background: 'none',
                                  fontSize: 12, cursor: 'pointer', color: 'var(--text-secondary)',
                                  borderRadius: 6, textAlign: 'left', whiteSpace: 'nowrap',
                                }}
                                onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-raised)'; e.currentTarget.style.color = 'var(--text-primary)' }}
                                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)' }}
                              >
                                <svg style={{ width: 13, height: 13, flexShrink: 0 }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                                  <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
                                </svg>
                                Rename
                              </button>
                              <button onClick={() => handleDelete(s.id)}
                                style={{
                                  display: 'flex', alignItems: 'center', gap: 7,
                                  width: '100%', padding: '8px 12px', border: 'none', background: 'none',
                                  fontSize: 12, cursor: 'pointer', color: 'var(--danger)',
                                  borderRadius: 6, textAlign: 'left', whiteSpace: 'nowrap',
                                }}
                                onMouseEnter={e => { e.currentTarget.style.background = 'var(--danger-soft)' }}
                                onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
                              >
                                <svg style={{ width: 13, height: 13, flexShrink: 0 }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <polyline points="3 6 5 6 21 6"/>
                                  <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                                </svg>
                                Delete
                              </button>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Sidebar bottom */}
        <div style={{ padding: '10px 16px 16px' }}>
          {files.length > 0 && (
            <div className="stats-bar" style={{
              marginBottom: 10, padding: '8px 12px', borderRadius: 10,
              background: 'var(--bg-raised)', border: '1px solid var(--border)',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)' }}>
                {files.filter(f => f.status === 'indexed').length} indexed
              </span>
              <span style={{
                fontSize: 10, color: 'var(--success)', background: 'var(--success-soft)',
                padding: '2px 7px', borderRadius: 20, fontFamily: 'var(--font-mono)', fontWeight: 500,
              }}>
                ● live
              </span>
            </div>
          )}
          <button onClick={handleNewChat} className="accent-btn" style={{
            width: '100%', padding: '10px 0', borderRadius: 12, fontSize: 13,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          }}>
            <svg style={{ width: 13, height: 13 }} viewBox="0 0 24 24"
              fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            New conversation
          </button>
        </div>
      </aside>

      {/* ══════ MAIN CHAT AREA ══════ */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column',
        minWidth: 0, overflow: 'hidden' }}>

        {/* Header bar */}
        <header className="top-header" style={{
          height: 54, display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', padding: '0 24px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-surface)', flexShrink: 0, boxShadow: 'var(--shadow-sm)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)' }}>
              {(() => {
                const current = sessions.find(s => s.id === sessionId)
                if (current?.title && current.title !== 'New conversation') {
                  return current.title.length > 50 ? current.title.slice(0, 50) + '…' : current.title
                }
                return hasMessages ? 'Conversation' : 'New conversation'
              })()}
            </span>
            {thinking && (
              <span className="shimmer-text" style={{
                fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
              }}>
                generating…
              </span>
            )}
          </div>

          {/* Right side: Clear button + user avatar menu */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {hasMessages && (
              <button onClick={handleNewChat} className="ghost-btn" style={{
                fontSize: 12, padding: '5px 12px', borderRadius: 8,
                fontFamily: 'var(--font-sans)',
              }}>
                Clear
              </button>
            )}
            {/* UserMenu shows the avatar + logout dropdown */}
            <UserMenu />
          </div>
        </header>

        {/* Messages scroll area */}
        <div
          className={`chat-scroll ${theme === 'dark' ? 'chat-bg' : ''}`}
          style={{
            flex: 1, overflowY: 'auto', padding: '28px 24px',
            background: theme !== 'dark' ? 'var(--bg-base)' : undefined,
          }}
        >
          {!hasMessages ? (
            /* Empty state */
            <div style={{
              height: '100%', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              gap: 24, maxWidth: 520, margin: '0 auto', textAlign: 'center',
            }}>
              {/* Animated icon container */}
              <div style={{
                position: 'relative',
                marginBottom: 8,
              }}>
                <div style={{
                  width: 80, height: 80, borderRadius: 24,
                  background: 'linear-gradient(145deg, var(--accent-soft), var(--bg-surface))',
                  border: '1.5px solid var(--accent)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 0 0 10px var(--accent-glow), var(--shadow-accent)',
                }}>
                  <svg style={{ width: 36, height: 36, color: 'var(--accent)' }}
                    viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                  </svg>
                </div>
                {/* Subtle pulse ring */}
                <div style={{
                  position: 'absolute', inset: -4, borderRadius: 28,
                  border: '1px solid var(--accent)', opacity: 0.3,
                  animation: 'pulse-glow 2.5s ease-in-out infinite',
                }} />
              </div>

              <div style={{ marginTop: 4 }}>
                <h2 style={{ fontSize: 25, fontWeight: 600, color: 'var(--text-primary)',
                  letterSpacing: '-0.03em', marginBottom: 12, lineHeight: 1.2 }}>
                  {hasFiles
                    ? `Ready, ${user?.name?.split(' ')[0] ?? 'there'}`
                    : 'Upload a document to begin'}
                </h2>
                <p style={{ fontSize: 15, color: 'var(--text-secondary)', lineHeight: 1.65,
                  fontFamily: 'var(--font-serif)', fontStyle: 'italic', maxWidth: 360 }}>
                  {hasFiles
                    ? 'Ask anything about your indexed documents below.'
                    : 'Drop a PDF, Markdown, or text file in the sidebar.'}
                </p>
              </div>

              {/* Starter suggestion cards */}
              {hasFiles && (
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr', 
                  gap: 10, 
                  width: '100%',
                  marginTop: 8,
                }}>
                  {STARTERS.map((s, i) => (
                    <button key={i} onClick={() => sendMessage(s)}
                      className="starter-card animate-fade-up"
                      style={{
                        animationDelay: `${i * 70}ms`, 
                        padding: '14px 16px', 
                        borderRadius: 14,
                        background: 'var(--bg-surface)', 
                        border: '1px solid var(--border-med)',
                        color: 'var(--text-secondary)', 
                        fontSize: 13, 
                        textAlign: 'left',
                        cursor: 'pointer', 
                        lineHeight: 1.5, 
                        transition: 'all 0.2s ease',
                        fontFamily: 'var(--font-sans)', 
                        boxShadow: 'var(--shadow-sm)',
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = 'var(--accent)'
                        e.currentTarget.style.color = 'var(--text-primary)'
                        e.currentTarget.style.background = 'var(--accent-soft)'
                        e.currentTarget.style.boxShadow = 'var(--shadow-accent)'
                        e.currentTarget.style.transform = 'translateY(-2px)'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = 'var(--border-med)'
                        e.currentTarget.style.color = 'var(--text-secondary)'
                        e.currentTarget.style.background = 'var(--bg-surface)'
                        e.currentTarget.style.boxShadow = 'var(--shadow-sm)'
                        e.currentTarget.style.transform = 'translateY(0)'
                      }}
                    >
                      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <svg style={{ width: 14, height: 14, color: 'var(--accent)', flexShrink: 0 }}
                          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10"/>
                          <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>
                          <line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        {s}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            /* Message list */
            <div style={{ maxWidth: 720, margin: '0 auto' }}>
              {messages.map((msg, i) => (
                <Message key={msg.id ?? i} message={msg} index={i} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="input-zone" style={{
          padding: '12px 24px 16px', borderTop: '1px solid var(--border)',
          background: 'var(--bg-surface)', flexShrink: 0,
        }}>
          <div style={{ maxWidth: 720, margin: '0 auto' }}>
            <ChatInput onSend={sendMessage} disabled={thinking || !sessionId} />
            <p style={{
              fontSize: 10.5, color: 'var(--text-faint)', textAlign: 'center',
              marginTop: 8, fontFamily: 'var(--font-mono)', letterSpacing: '0.03em',
            }}>
              grounded in your documents · Enter to send · Shift+Enter for newline
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}

// ── Exported page — ProtectedRoute handles the auth gate ───────────────────
export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}