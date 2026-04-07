'use client'
import { useRef, useState } from 'react'
/*
UploadZone lets users:

click to upload files 
drag & drop files 
only allow specific file types
send files to parent via onUpload
*/

export default function UploadZone({ onUpload, uploading }) {
  const inputRef  = useRef(null)
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
      onClick={() => inputRef.current?.click()}
      onDragOver={e  => { e.preventDefault(); setDrag(true)  }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault()
        setDrag(false)
        handleFiles(e.dataTransfer.files)
      }}
      className={`
        relative flex flex-col items-center justify-center
        border-2 border-dashed rounded-xl p-6 cursor-pointer
        transition-colors text-center
        ${drag || uploading
          ? 'border-violet-400 bg-violet-50 dark:bg-violet-950/20'
          : 'border-gray-300 dark:border-gray-700 hover:border-violet-400'}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown"
        multiple
        className="hidden"
        onChange={e => handleFiles(e.target.files)}
      />
      <div className="text-3xl mb-2">📄</div>
      <p className="text-sm text-gray-600 dark:text-gray-400">
        {uploading ? 'Indexing…' : 'Drop files or click to upload'}
      </p>
      <p className="text-xs text-gray-400 dark:text-gray-600 mt-1">
        PDF · MD · TXT
      </p>
    </div>
  )
}