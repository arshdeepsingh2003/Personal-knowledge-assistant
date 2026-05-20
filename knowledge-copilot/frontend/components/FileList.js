export default function FileList({ files, onRemove }) {
  if (!files.length) return null

  const STATUS = {
    indexing: { symbol: '◌', color: 'var(--accent)',   label: 'indexing' },
    indexed:  { symbol: '◆', color: 'var(--success)',  label: 'indexed'  },
    error:    { symbol: '✕', color: 'var(--danger)',   label: 'error'    },
  }

  return (
    <ul style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 5 }}>
      {files.map((f, i) => {
        const s = STATUS[f.status]
        return (
          <li
            key={`${f.name}-${i}`}
            className="file-pill animate-slide-in"
            style={{
              animationDelay: `${i * 45}ms`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '7px 10px',
              borderRadius: 9,
              background: 'var(--bg-well)',
              border: '1px solid var(--border)',
              gap: 7,
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
              <span style={{
                fontSize: 10,
                color: s.color,
                flexShrink: 0,
                animation: f.status === 'indexing' ? 'spin-slow 2s linear infinite' : 'none',
              }}>
                {s.symbol}
              </span>
              <span style={{
                fontSize: 11.5,
                color: 'var(--text-secondary)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontFamily: 'var(--font-mono)',
              }}>
                {f.name}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              {f.chunks != null && (
                <span style={{
                  fontSize: 9.5,
                  color: 'var(--text-muted)',
                  flexShrink: 0,
                  padding: '2px 6px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  fontFamily: 'var(--font-mono)',
                  background: 'var(--bg-overlay)',
                }}>
                  {f.chunks}c
                </span>
              )}
              <button
                onClick={e => { e.stopPropagation(); onRemove?.(f.name) }}
                title={`Remove ${f.name}`}
                style={{
                  width: 20, height: 20, borderRadius: 6,
                  border: 'none', background: 'transparent',
                  color: 'var(--text-muted)', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0, opacity: 0.5, transition: 'all 0.15s',
                  fontSize: 13, lineHeight: 1,
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.opacity = '1'
                  e.currentTarget.style.color = 'var(--danger)'
                  e.currentTarget.style.background = 'var(--danger-soft)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.opacity = '0.5'
                  e.currentTarget.style.color = 'var(--text-muted)'
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                ✕
              </button>
            </div>
          </li>
        )
      })}
    </ul>
  )
}