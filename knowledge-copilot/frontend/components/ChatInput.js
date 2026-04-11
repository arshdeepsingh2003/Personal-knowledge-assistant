'use client'
import { useState, useRef } from 'react'

/*
ChatInput lets user:

type a message ✍️
send it (Enter or button) 🚀s
auto-resize textarea 📏
disable input when needed ⛔
*/

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('') //what user types
  const textareaRef = useRef(null) //access textarea DOM (for resizing)

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function onKeyDown(e) {
    // Send on Enter, new line on Shift+Enter
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function onInput(e) {
    setValue(e.target.value)
    // Auto-grow textarea up to ~6 lines
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 144) + 'px'
  }

  return (
    <div className="flex items-end gap-2 p-3
                    border border-gray-200 dark:border-gray-700
                    rounded-2xl bg-white dark:bg-gray-900
                    shadow-sm">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={onInput}
        onKeyDown={onKeyDown}
        disabled={disabled}
        placeholder="Ask a question about your documents…"
        className="flex-1 resize-none bg-transparent outline-none
                   text-sm text-gray-800 dark:text-gray-200
                   placeholder-gray-400 dark:placeholder-gray-600
                   max-h-36 leading-relaxed"
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="shrink-0 w-9 h-9 flex items-center justify-center
                   rounded-xl bg-violet-600 text-white
                   disabled:opacity-40 disabled:cursor-not-allowed
                   hover:bg-violet-700 transition-colors"
      >
        <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path d="M3.105 2.289a.75.75 0 00-.826.95l1.903 6.557H13.5a.75.75 0 010 1.5H4.182l-1.903 6.557a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z"/>
        </svg>
      </button>
    </div>
  )
}