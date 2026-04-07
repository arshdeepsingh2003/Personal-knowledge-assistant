'use client'
import { useEffect, useRef } from 'react'
import { useSession } from '@/hooks/useSession'
import { useChat }    from '@/hooks/useChat'
import { useUpload }  from '@/hooks/useUpload'
import UploadZone     from '@/components/UploadZone'
import FileList       from '@/components/FileList'
import Message        from '@/components/Message'
import ChatInput      from '@/components/ChatInput'

export default function Home() {
  const { sessionId, loading, resetSession } = useSession()
  const { messages, thinking, sendMessage, clearMessages } = useChat(sessionId)
  const { files, uploading, error: uploadError, upload } = useUpload()
  const bottomRef = useRef(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleNewChat() {
    clearMessages()
    await resetSession()
  }

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center
                      bg-gray-50 dark:bg-gray-950">
        <div className="flex gap-2">
          {[0,1,2].map(i => (
            <div key={i}
              style={{ animationDelay: `${i * 150}ms` }}
              className="w-2 h-2 rounded-full bg-violet-500 animate-bounce"
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex bg-gray-50 dark:bg-gray-950 overflow-hidden">

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="w-64 shrink-0 flex flex-col
                        bg-white dark:bg-gray-900
                        border-r border-gray-200 dark:border-gray-800">

        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-800">
          <h1 className="font-semibold text-gray-900 dark:text-gray-100">
            Knowledge Copilot
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Chat with your documents
          </p>
        </div>

        {/* Upload section */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-800">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400
                        uppercase tracking-wide mb-3">
            Documents
          </p>
          <UploadZone onUpload={upload} uploading={uploading} />
          {uploadError && (
            <p className="text-xs text-red-500 mt-2">{uploadError}</p>
          )}
          <FileList files={files} />
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* New chat button */}
        <div className="p-4">
          <button
            onClick={handleNewChat}
            className="w-full py-2.5 px-4 rounded-xl
                       bg-violet-600 hover:bg-violet-700
                       text-white text-sm font-medium
                       transition-colors"
          >
            + New chat
          </button>
        </div>
      </aside>

      {/* ── Main chat area ────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">

        {/* Top bar */}
        <header className="h-14 flex items-center px-6
                           border-b border-gray-200 dark:border-gray-800
                           bg-white dark:bg-gray-900 shrink-0">
          <h2 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {messages.length > 0 ? 'Chat' : 'Start a conversation'}
          </h2>
          {thinking && (
            <span className="ml-3 text-xs text-violet-500 animate-pulse">
              Thinking…
            </span>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto chat-scroll px-6 py-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center
                            justify-center text-center">
              <div className="text-5xl mb-4">💬</div>
              <h3 className="text-lg font-medium
                             text-gray-700 dark:text-gray-300 mb-2">
                No messages yet
              </h3>
              <p className="text-sm text-gray-400 max-w-sm">
                Upload a document on the left, then ask anything about it.
              </p>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <Message key={i} message={msg} />
              ))}
              <div ref={bottomRef} />
            </>
          )}
        </div>

        {/* Input */}
        <div className="p-4 bg-white dark:bg-gray-900
                        border-t border-gray-200 dark:border-gray-800 shrink-0">
          <ChatInput onSend={sendMessage} disabled={thinking || !sessionId} />
          <p className="text-xs text-center text-gray-400 mt-2">
            Answers are grounded in your uploaded documents only
          </p>
        </div>
      </main>
    </div>
  )
}