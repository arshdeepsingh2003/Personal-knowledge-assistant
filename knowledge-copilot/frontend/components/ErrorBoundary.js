'use client'
import React from 'react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: '#0c0a08', fontFamily: 'sans-serif',
          padding: 24, gap: 12,
        }}>
          <div style={{
            width: 48, height: 48, borderRadius: 14,
            background: 'rgba(240,160,48,0.15)', border: '1.5px solid #f0a030',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg style={{ width: 22, height: 22, color: '#f0a030' }}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
          </div>
          <p style={{ color: '#f0a030', fontSize: 14, fontWeight: 500 }}>
            Something went wrong
          </p>
          <p style={{ color: '#806848', fontSize: 12, textAlign: 'center', maxWidth: 400 }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button onClick={() => window.location.reload()} style={{
            marginTop: 8, padding: '8px 20px', borderRadius: 8,
            background: '#f0a030', color: '#fff', border: 'none',
            cursor: 'pointer', fontSize: 13, fontFamily: 'sans-serif',
          }}>
            Reload page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
