'use client'

import ReactMarkdown from 'react-markdown'

/*
Message Component:
- Renders user & AI messages
- Supports markdown
- Shows streaming typing effect
- Displays sources (RAG)
*/

function SourceBadge({ source }) {
  return (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-1
                     bg-gray-100 dark:bg-gray-800 rounded-full
                     text-gray-500 dark:text-gray-400">
      <span>📄</span>

      <span className="max-w-[140px] truncate">
        {source.file_name || 'Unknown'}
      </span>

      {source.page != null && (
        <span>p.{source.page + 1}</span>
      )}

      {source.score != null && (
        <span className="text-violet-500">
          {(source.score * 100).toFixed(0)}%
        </span>
      )}
    </span>
  )
}

export default function Message({ message }) {
  const isUser = message.role === 'user'

  // 🟣 USER MESSAGE
  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%] px-4 py-3 rounded-2xl rounded-tr-sm
                        bg-violet-600 text-white text-sm break-words">
          {message.content}
        </div>
      </div>
    )
  }

  // 🤖 AI MESSAGE
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[85%]">

        {/* Avatar */}
        <div className="w-7 h-7 rounded-full bg-violet-100 dark:bg-violet-900
                        flex items-center justify-center text-sm mb-1">
          🤖
        </div>

        {/* Message Bubble */}
        <div className="px-4 py-3 rounded-2xl rounded-tl-sm
                        bg-gray-100 dark:bg-gray-800
                        text-gray-800 dark:text-gray-200 text-sm break-words">

          {/* ❌ Error */}
          {message.error ? (
            <p className="text-red-500">{message.error}</p>
          ) : message.content ? (

            /* ✅ Markdown Content */
            <ReactMarkdown
              components={{
                p: ({ children }) => (
                  <p className="mb-2 last:mb-0">{children}</p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc pl-4 mb-2">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal pl-4 mb-2">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="mb-1">{children}</li>
                ),
                code: ({ inline, children }) => (
                  <code
                    className={`${
                      inline
                        ? 'bg-gray-200 dark:bg-gray-700 px-1 rounded text-xs'
                        : 'block bg-gray-900 text-white p-2 rounded mt-2 text-xs overflow-x-auto'
                    } font-mono`}
                  >
                    {children}
                  </code>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>

          ) : (
            /* ⏳ Initial streaming (no content yet) */
            <span className="inline-block w-2 h-4 bg-violet-400
                             rounded animate-pulse" />
          )}

          {/* ⚡ Streaming cursor */}
          {message.streaming && message.content && (
            <span className="inline-block w-2 h-4 ml-0.5 bg-violet-400
                             rounded animate-pulse align-middle" />
          )}
        </div>

        {/* 📚 Sources */}
        {message.sources?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {message.sources.map((s, i) => (
              <SourceBadge key={s.id || `${s.file_name}-${i}`} source={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}