'use client'

/*
FileList displays:
- list of uploaded files
- their status (loading / done / error)
- number of chunks (if processed)
*/

export default function FileList({ files }) {
  if (!files || files.length === 0) return null

  const icon = {
    indexing: '⏳',
    indexed: '✅',
    error: '❌',
  }

  return (
    <ul className="mt-3 space-y-2">
      {files.map((f, index) => (
        <li
          key={f.id || `${f.name}-${index}`} // safer key
          className="flex items-center justify-between text-xs px-3 py-2
                     rounded-lg bg-gray-100 dark:bg-gray-800"
        >
          {/* Left side: file name + status */}
          <span className="truncate text-gray-700 dark:text-gray-300 max-w-[150px]">
            {icon[f.status] || '📄'} {f.name}
          </span>

          {/* Right side: chunks or error */}
          <span className="text-gray-400 shrink-0 ml-2">
            {f.status === 'error' && 'Failed'}
            {f.chunks != null && `${f.chunks} chunks`}
          </span>
        </li>
      ))}
    </ul>
  )
}