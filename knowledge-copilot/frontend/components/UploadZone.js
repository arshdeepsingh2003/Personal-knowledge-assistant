'use client'
import { useRef, useState } from 'react'

export default function UploadZone({ onUpload, uploading }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)

  function handleFiles(fileList) {
    const allowed = ['.pdf', '.txt', '.md', '.markdown']
    Array.from(fileList).forEach(file => {
      const ext = '.' + file.name.split('.').pop().toLowerCase()
      if (allowed.includes(ext)) onUpload(file)
    })
  }

  return (
    <div
      onClick={() => !uploading && inputRef.current?.click()}
      onDragOver={e  => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault()
        setDrag(false)
        if (!uploading) handleFiles(e.dataTransfer.files)
      }}
      className={`upload-zone${drag ? ' dragging' : ''}`}
      style={{
        borderRadius: 12,
        padding: '12px 10px',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        cursor: uploading ? 'wait' : 'pointer',
        border: `1.5px dashed ${drag ? 'var(--accent)' : 'var(--border-med)'}`,
        background: drag ? 'var(--accent-soft)' : 'var(--bg-raised)',
        transition: 'all 0.2s ease',
        boxShadow: drag ? '0 0 0 3px var(--accent-glow)' : 'none',
        userSelect: 'none',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown"
        multiple
        style={{ display: 'none' }}
        onChange={e => handleFiles(e.target.files)}
      />

      {/* Icon container */}
      <div style={{
        width: 32,
        height: 32,
        borderRadius: 8,
        background: 'var(--accent-soft)',
        border: '1px solid var(--accent)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        transform: drag ? 'scale(1.05)' : 'scale(1)',
        transition: 'transform 0.2s',
        boxShadow: drag ? 'var(--shadow-accent)' : 'none',
      }}>
        {uploading ? (
          <svg style={{ width: 14, height: 14, color: 'var(--accent)',
            animation: 'spin-slow 1s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        ) : (
          <svg style={{ width: 14, height: 14, color: 'var(--accent)' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        )}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: 11.5, fontWeight: 500, color: 'var(--text-secondary)', lineHeight: 1.3 }}>
          {uploading ? 'Indexing...' : drag ? 'Release to upload' : 'Drop files or click'}
        </p>
        <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2,
          fontFamily: 'var(--font-mono)' }}>
          PDF · MD · TXT
        </p>
      </div>
    </div>
  )
}