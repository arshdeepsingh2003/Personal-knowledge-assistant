'use client'
import { useState, useCallback } from 'react'
import { uploadDocument } from '@/lib/api'

export function useUpload() {
  const [files,     setFiles]     = useState([])
  const [uploading, setUploading] = useState(false)
  const [error,     setError]     = useState(null)

  const upload = useCallback(async (file) => {
    setUploading(true)
    setError(null)
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