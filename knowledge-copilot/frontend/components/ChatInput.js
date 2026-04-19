'use client'
import { useState, useRef, useEffect } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef(null)

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function onInput(e) {
    setValue(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
  }

  const canSend = !disabled && value.trim().length > 0

  const placeholder = disabled 
    ? 'Connecting...' 
    : isFocused 
      ? 'Ask a question...' 
      : 'Ask anything about your documents...'

  return (
    <div
      className="input-wrapper"
      style={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: 10,
        padding: '10px 12px',
        borderRadius: 18,
        background: 'var(--bg-raised)',
        border: `1.5px solid ${canSend || isFocused ? 'var(--accent)' : 'var(--border-med)'}`,
        boxShadow: canSend || isFocused
          ? '0 0 0 3px var(--accent-glow), var(--shadow-md)'
          : 'var(--shadow-sm)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
    >
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={onInput}
        onKeyDown={onKeyDown}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        disabled={disabled}
        placeholder={placeholder}
        style={{
          flex: 1,
          resize: 'none',
          background: 'transparent',
          outline: 'none',
          border: 'none',
          fontSize: 14,
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-sans)',
          lineHeight: 1.6,
          maxHeight: 140,
          caretColor: 'var(--accent)',
        }}
      />

      <button
        onClick={submit}
        disabled={!canSend || disabled}
        style={{
          width: 36,
          height: 36,
          borderRadius: 12,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: canSend ? 'var(--accent)' : 'var(--bg-well)',
          border: `1px solid ${canSend ? 'var(--accent)' : 'var(--border)'}`,
          color: canSend ? '#fff' : 'var(--text-muted)',
          cursor: canSend ? 'pointer' : 'not-allowed',
          transition: 'all 0.2s ease',
          boxShadow: canSend ? 'var(--shadow-accent)' : 'none',
          transform: 'scale(1)',
        }}
        onMouseEnter={e => {
          if (canSend) {
            e.currentTarget.style.transform = 'scale(1.08)'
            e.currentTarget.style.boxShadow = '0 4px 16px var(--accent-glow-strong)'
          }
        }}
        onMouseLeave={e => {
          e.currentTarget.style.transform = 'scale(1)'
          e.currentTarget.style.boxShadow = canSend ? 'var(--shadow-accent)' : 'none'
        }}
      >
        {disabled ? (
          <svg style={{ width: 16, height: 16,
            animation: 'spin-slow 0.9s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        ) : (
          <svg style={{ width: 16, height: 16 }}
            viewBox="0 0 20 20" fill="currentColor">
            <path d="M3.105 2.289a.75.75 0 00-.826.95l1.903 6.557H13.5a.75.75 0 010 1.5H4.182l-1.903 6.557a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z"/>
          </svg>
        )}
      </button>
    </div>
  )
}