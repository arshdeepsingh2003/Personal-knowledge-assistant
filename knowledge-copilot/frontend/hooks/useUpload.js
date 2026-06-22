'use client'
import { useState, useCallback } from 'react'
import { uploadDocument } from '@/lib/api'

export function useUpload() {
  const [files,       setFiles]       = useState([])
  const [uploading,   setUploading]   = useState(false)
  const [error,       setError]       = useState(null)
  const [documentIds, setDocumentIds] = useState([])

  const upload = useCallback(async (file, conversationId = null) => {
    setUploading(true)
    setError(null)
    setFiles(prev => [...prev, { name: file.name, status: 'indexing', chunks: null }])

    try {
      const result = await uploadDocument(file, { conversationId })
      setDocumentIds(prev => [...prev, result.document_id])
      setFiles(prev => prev.map(f =>
        f.name === file.name
          ? { name: f.name, status: 'indexed', chunks: result.chunks_added, documentId: result.document_id }
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

  function removeFile(name) {
    setFiles(prev => prev.filter(f => f.name !== name))
    setDocumentIds(prev => {
      const removed = prev.find(docId => {
        const file = files.find(f => f.name === name)
        return file && file.documentId === docId
      })
      return removed ? prev.filter(id => id !== removed) : prev
    })
  }

  return { files, uploading, error, upload, removeFile, documentIds }
}