import ReactMarkdown from 'react-markdown'

function SourceBadge({ source, index }) {
  const pct = Math.round((source.score ?? 0) * 100)
  const scoreColor = pct >= 70
    ? 'var(--success)'
    : pct >= 40
      ? 'var(--accent)'
      : 'var(--text-muted)'

  return (
    <span
      className="source-badge animate-scale-in"
      style={{
        animationDelay: `${index * 55}ms`,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        fontSize: 11,
        padding: '3px 9px',
        borderRadius: 20,
        background: 'var(--bg-well)',
        border: '1px solid var(--border-med)',
        color: 'var(--text-secondary)',
        fontFamily: 'var(--font-mono)',
        cursor: 'default',
        transition: 'all 0.18s',
        boxShadow: 'var(--shadow-sm)',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--accent)'
        e.currentTarget.style.background  = 'var(--accent-soft)'
        e.currentTarget.style.color       = 'var(--text-primary)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border-med)'
        e.currentTarget.style.background  = 'var(--bg-well)'
        e.currentTarget.style.color       = 'var(--text-secondary)'
      }}
    >
      <span style={{ color: 'var(--accent)', fontSize: 9 }}>◆</span>
      <span style={{
        maxWidth: 130,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {source.file_name}
      </span>
      {source.page != null && (
        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>p.{source.page + 1}</span>
      )}
      <span style={{ color: scoreColor, fontWeight: 500 }}>{pct}%</span>
    </span>
  )
}

const md = {
  p:    ({children}) => (
    <p style={{ marginBottom: '0.55em', lineHeight: 1.7, lastChild: { marginBottom: 0 } }}>
      {children}
    </p>
  ),
  ul:   ({children}) => (
    <ul style={{ paddingLeft: '1.2em', marginBottom: '0.55em', listStyleType: 'disc' }}>
      {children}
    </ul>
  ),
  ol:   ({children}) => (
    <ol style={{ paddingLeft: '1.2em', marginBottom: '0.55em', listStyleType: 'decimal' }}>
      {children}
    </ol>
  ),
  li:   ({children}) => <li style={{ marginBottom: '0.2em' }}>{children}</li>,
  strong: ({children}) => (
    <strong style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{children}</strong>
  ),
  code: ({inline, children}) => inline ? (
    <code style={{
      fontFamily: 'var(--font-mono)',
      fontSize: '0.82em',
      background: 'var(--bg-well)',
      border: '1px solid var(--border-med)',
      padding: '1px 5px',
      borderRadius: 5,
      color: 'var(--accent)',
    }}>
      {children}
    </code>
  ) : (
    <pre style={{
      fontFamily: 'var(--font-mono)',
      fontSize: '0.8em',
      background: 'var(--bg-well)',
      border: '1px solid var(--border)',
      padding: '10px 14px',
      borderRadius: 10,
      overflowX: 'auto',
      margin: '0.5em 0',
      color: 'var(--text-secondary)',
    }}>
      <code>{children}</code>
    </pre>
  ),
  blockquote: ({children}) => (
    <blockquote style={{
      borderLeft: '2px solid var(--accent)',
      paddingLeft: '0.9em',
      color: 'var(--text-secondary)',
      fontStyle: 'italic',
      margin: '0.5em 0',
      fontFamily: 'var(--font-serif)',
      fontSize: '1.02em',
    }}>
      {children}
    </blockquote>
  ),
}

export default function Message({ message, index }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div
        className="animate-fade-up"
        style={{
          animationDelay: `${Math.min(index * 25, 150)}ms`,
          display: 'flex',
          justifyContent: 'flex-end',
          marginBottom: 18,
        }}
      >
        <div
          className="user-bubble"
          style={{
            maxWidth: '72%',
            padding: '11px 16px',
            borderRadius: '18px 18px 4px 18px',
            background: 'var(--user-bubble)',
            color: 'var(--user-text)',
            fontSize: 14,
            lineHeight: 1.65,
            boxShadow: 'var(--shadow-md)',
            border: '1px solid transparent',
          }}
        >
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div
      className="animate-fade-up"
      style={{
        animationDelay: `${Math.min(index * 25, 150)}ms`,
        display: 'flex',
        gap: 10,
        marginBottom: 22,
        alignItems: 'flex-start',
      }}
    >
      {/* Avatar */}
      <div style={{
        width: 28,
        height: 28,
        borderRadius: 8,
        background: 'var(--accent-soft)',
        border: '1px solid var(--accent)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        marginTop: 2,
        boxShadow: 'var(--shadow-accent)',
      }}>
        <svg style={{ width: 13, height: 13, color: 'var(--accent)' }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="8" r="4"/>
          <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
        </svg>
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Bubble */}
        <div
          className="assistant-bubble"
          style={{
            padding: '12px 16px',
            borderRadius: '4px 18px 18px 18px',
            background: 'var(--bg-raised)',
            border: '1px solid var(--border-med)',
            fontSize: 14,
            lineHeight: 1.7,
            color: 'var(--text-primary)',
            boxShadow: 'var(--shadow-md)',
          }}
        >
          {message.error ? (
            <div style={{
              padding: '8px 12px',
              borderRadius: 8,
              background: 'var(--danger-soft)',
              border: '1px solid var(--danger)',
              color: 'var(--danger)',
              fontSize: 13,
            }}>
              ⚠ {message.error}
            </div>
          ) : message.content ? (
            <>
              <ReactMarkdown components={md}>
                {message.content}
              </ReactMarkdown>
              {message.streaming && (
                <span style={{
                  display: 'inline-block',
                  width: 7,
                  height: 15,
                  background: 'var(--accent)',
                  borderRadius: 2,
                  marginLeft: 2,
                  verticalAlign: 'middle',
                  animation: 'pulse-glow 0.9s ease-in-out infinite',
                  boxShadow: '0 0 8px var(--accent-glow)',
                }} />
              )}
            </>
          ) : (
            /* Thinking dots */
            <span style={{ display: 'inline-flex', gap: 5, alignItems: 'center', padding: '2px 0' }}>
              {[0,1,2].map(i => (
                <span key={i} style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: 'var(--accent)',
                  display: 'inline-block',
                  animation: `bounce-dot 1.2s ease-in-out ${i * 0.18}s infinite`,
                  boxShadow: '0 0 6px var(--accent-glow)',
                }} />
              ))}
            </span>
          )}
        </div>

        {/* Sources */}
        {message.sources?.length > 0 && !message.streaming && (
          <div style={{
            marginTop: 8,
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
          }}>
            {message.sources.map((s, i) => (
              <SourceBadge key={i} source={s} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}