'use client'
export const dynamic = 'force-dynamic'

import { SignUp } from '@clerk/nextjs'

export default function SignUpPage() {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg-base)',
    }}>
      <SignUp 
        routing="path"
        path="/sign-up"
        signInUrl="/sign-in"
        forceRedirectUrl="/sso-callback"
      />
    </div>
  )
}
