'use client'
import { useState, useCallback } from 'react'
import { uploadDocument, getDocumentStatus } from '@/lib/api'

/*
useUpload() manages:

uploading a file 📄
showing its status (indexing / done / error)
tracking uploaded files list
*/

export function useUpload() {
  const [files,     setFiles]     = useState([])    // list of uploaded files
  const [uploading, setUploading] = useState(false) // upload in progress
  const [error,     setError]     = useState(null)

  const upload = useCallback(async (file) => {
    setUploading(true)
    setError(null)

    // Optimistic UI — show file immediately with pending state
    // “File uploaded… processing”
    setFiles(prev => [...prev, { name: file.name, status: 'indexing', chunks: null }])

    try {
      const result = await uploadDocument(file)
      setFiles(prev => prev.map(f =>
        f.name === file.name
          ? { name: f.name, status: 'indexed', chunks: result.chunks_added }
          : f
      ))
    } catch (err) {
      setError(err.message)
      setFiles(prev => prev.map(f =>
        f.name === file.name ? { ...f, status: 'error' } : f
      ))
    } finally {
      setUploading(false)
    }
  }, [])

  return { files, uploading, error, upload }
}