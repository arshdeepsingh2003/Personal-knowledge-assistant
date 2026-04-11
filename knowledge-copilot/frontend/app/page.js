'use client'
import { useEffect, useRef } from 'react'
import { useSession }  from '@/hooks/useSession'
import { useChat }     from '@/hooks/useChat'
import { useUpload }   from '@/hooks/useUpload'
import { useTheme }    from '@/hooks/useTheme'
import UploadZone      from '@/components/UploadZone'
import FileList        from '@/components/FileList'
import Message         from '@/components/Message'
import ChatInput       from '@/components/ChatInput'
import ThemeToggle     from '@/components/ThemeToggle'

const STARTERS = [
  'Summarise this document in 3 key points',
  'What are the main conclusions?',
  'List any important dates or figures mentioned',
  'What questions does this document answer?',
]

export default function Home() {
  const { sessionId, loading, resetSession } = useSession()
  const { messages, thinking, sendMessage, clearMessages } = useChat(sessionId)
  const { files, uploading, error: uploadError, upload } = useUpload()
  const { theme, toggle: toggleTheme, mounted: themeMounted } = useTheme()
  const bottomRef   = useRef(null)
  const hasMessages = messages.length > 0
  const hasFiles    = files.some(f => f.status === 'indexed')

  useEffect(() => {
    if (!themeMounted) return
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme, themeMounted])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleNewChat() {
    clearMessages()
    await resetSession()
  }

  /* ── Loading ──────────────────────────────────────────────────────────── */
  if (loading) {
    return (
      <div
        suppressHydrationWarning
        style={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg-base)',
          gap: 14,
        }}
      >
        <div style={{
          width: 42,
          height: 42,
          borderRadius: 13,
          background: 'var(--accent-soft)',
          border: '1.5px solid var(--accent)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: 'var(--shadow-accent)',
        }}>
          <svg style={{ width: 20, height: 20, color: 'var(--accent)',
            animation: 'spin-slow 1.1s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        </div>
        <p style={{
          fontSize: 12,
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.05em',
        }}>
          initialising…
        </p>
      </div>
    )
  }

  /* ── Layout ───────────────────────────────────────────────────────────── */
  return (
    <div style={{ height: '100vh', display: 'flex', overflow: 'hidden', background: 'var(--bg-base)' }}>

      {/* ════════════════════════════ SIDEBAR ════════════════════════════ */}
      <aside
        className="sidebar"
        style={{
          width: 264,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--bg-surface)',
          borderRight: '1px solid var(--border-med)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-md)',
          position: 'relative',
          zIndex: 2,
        }}
      >
        {/* Brand */}
        <div style={{
          padding: '18px 18px 14px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            {/* Logo */}
            <div style={{
              width: 28,
              height: 28,
              borderRadius: 8,
              background: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              boxShadow: 'var(--shadow-accent)',
            }}>
              <svg style={{ width: 14, height: 14, color: '#fff' }}
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <div>
              <p style={{
                fontSize: 13,
                fontWeight: 600,
                color: 'var(--text-primary)',
                letterSpacing: '-0.02em',
                lineHeight: 1.2,
              }}>
                Knowledge Copilot
              </p>
              <p style={{
                fontSize: 10,
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                marginTop: 1,
              }}>
                RAG assistant
              </p>
            </div>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>

        {/* Docs section */}
        <div style={{ padding: '14px 16px 0', flex: 1, overflowY: 'auto' }}>
          <p style={{
            fontSize: 9.5,
            fontWeight: 600,
            letterSpacing: '0.09em',
            textTransform: 'uppercase',
            color: 'var(--text-muted)',
            marginBottom: 10,
            fontFamily: 'var(--font-mono)',
          }}>
            Documents
          </p>

          <UploadZone onUpload={upload} uploading={uploading} />

          {uploadError && (
            <div style={{
              marginTop: 8,
              padding: '7px 10px',
              borderRadius: 8,
              background: 'var(--danger-soft)',
              border: '1px solid var(--danger)',
              fontSize: 11,
              color: 'var(--danger)',
              lineHeight: 1.4,
            }}>
              ⚠ {uploadError}
            </div>
          )}

          <FileList files={files} />
        </div>

        {/* Bottom section */}
        <div style={{ padding: '10px 16px 16px' }}>
          {/* Stats bar */}
          {files.length > 0 && (
            <div
              className="stats-bar"
              style={{
                marginBottom: 10,
                padding: '8px 12px',
                borderRadius: 10,
                background: 'var(--bg-raised)',
                border: '1px solid var(--border)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <span style={{
                fontSize: 11,
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)',
              }}>
                {files.filter(f => f.status === 'indexed').length} indexed
              </span>
              <span style={{
                fontSize: 10,
                color: 'var(--success)',
                background: 'var(--success-soft)',
                padding: '2px 7px',
                borderRadius: 20,
                fontFamily: 'var(--font-mono)',
                fontWeight: 500,
              }}>
                ● live
              </span>
            </div>
          )}

          {/* New chat */}
          <button
            onClick={handleNewChat}
            className="accent-btn"
            style={{
              width: '100%',
              padding: '10px 0',
              borderRadius: 12,
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
            }}
          >
            <svg style={{ width: 13, height: 13 }} viewBox="0 0 24 24"
              fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            New conversation
          </button>
        </div>
      </aside>

      {/* ════════════════════════════ MAIN ═══════════════════════════════ */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
        overflow: 'hidden',
      }}>

        {/* Header */}
        <header
          className="top-header"
          style={{
            height: 54,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-surface)',
            flexShrink: 0,
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{
              fontSize: 13,
              fontWeight: 500,
              color: 'var(--text-secondary)',
            }}>
              {hasMessages ? 'Conversation' : 'New conversation'}
            </span>

            {thinking && (
              <span
                className="shimmer-text"
                style={{
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.04em',
                }}
              >
                generating…
              </span>
            )}
          </div>

          {hasMessages && (
            <button
              onClick={handleNewChat}
              className="ghost-btn"
              style={{
                fontSize: 12,
                padding: '5px 12px',
                borderRadius: 8,
                fontFamily: 'var(--font-sans)',
              }}
            >
              Clear
            </button>
          )}
        </header>

        {/* Messages */}
        <div
          className={`chat-scroll ${theme === 'dark' ? 'chat-bg' : ''}`}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '28px 24px',
            background: theme !== 'dark' ? 'var(--bg-base)' : undefined,
          }}
        >
          {!hasMessages ? (
            /* Empty state */
            <div style={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 28,
              maxWidth: 500,
              margin: '0 auto',
              textAlign: 'center',
            }}>
              {/* Hero icon */}
              <div style={{
                width: 68,
                height: 68,
                borderRadius: 22,
                background: 'var(--accent-soft)',
                border: '1.5px solid var(--accent)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 0 8px var(--accent-glow), var(--shadow-accent)',
              }}>
                <svg style={{ width: 30, height: 30, color: 'var(--accent)' }}
                  viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                </svg>
              </div>

              <div>
                <h2 style={{
                  fontSize: 22,
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  letterSpacing: '-0.03em',
                  marginBottom: 10,
                }}>
                  {hasFiles ? 'Ready to answer' : 'Upload a document to begin'}
                </h2>
                <p style={{
                  fontSize: 14.5,
                  color: 'var(--text-secondary)',
                  lineHeight: 1.65,
                  fontFamily: 'var(--font-serif)',
                  fontStyle: 'italic',
                }}>
                  {hasFiles
                    ? 'Ask anything about your indexed documents below.'
                    : 'Drop a PDF, Markdown, or text file in the sidebar.'}
                </p>
              </div>

              {/* Starter chips */}
              {hasFiles && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: 8,
                  width: '100%',
                }}>
                  {STARTERS.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => sendMessage(s)}
                      className="starter-card animate-fade-up"
                      style={{
                        animationDelay: `${i * 70}ms`,
                        padding: '11px 14px',
                        borderRadius: 12,
                        background: 'var(--bg-surface)',
                        border: '1px solid var(--border-med)',
                        color: 'var(--text-secondary)',
                        fontSize: 12.5,
                        textAlign: 'left',
                        cursor: 'pointer',
                        lineHeight: 1.45,
                        transition: 'all 0.17s',
                        fontFamily: 'var(--font-sans)',
                        boxShadow: 'var(--shadow-sm)',
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = 'var(--accent)'
                        e.currentTarget.style.color = 'var(--text-primary)'
                        e.currentTarget.style.background = 'var(--accent-soft)'
                        e.currentTarget.style.boxShadow = 'var(--shadow-accent)'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = 'var(--border-med)'
                        e.currentTarget.style.color = 'var(--text-secondary)'
                        e.currentTarget.style.background = 'var(--bg-surface)'
                        e.currentTarget.style.boxShadow = 'var(--shadow-sm)'
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>

          ) : (
            <div style={{ maxWidth: 720, margin: '0 auto' }}>
              {messages.map((msg, i) => (
                <Message key={msg.id ?? i} message={msg} index={i} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div
          className="input-zone"
          style={{
            padding: '12px 24px 16px',
            borderTop: '1px solid var(--border)',
            background: 'var(--bg-surface)',
            flexShrink: 0,
            boxShadow: '0 -1px 0 var(--border)',
          }}
        >
          <div style={{ maxWidth: 720, margin: '0 auto' }}>
            <ChatInput onSend={sendMessage} disabled={thinking || !sessionId} />
            <p style={{
              fontSize: 10.5,
              color: 'var(--text-faint)',
              textAlign: 'center',
              marginTop: 8,
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.03em',
            }}>
              grounded in your documents · Enter to send · Shift+Enter for newline
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}