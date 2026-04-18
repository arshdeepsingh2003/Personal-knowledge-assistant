// FILE: frontend/app/layout.js
// PURPOSE: The root layout. Wraps EVERY page in the app.
//          Provides ClerkProvider (Google OAuth) and AuthProvider (our JWT auth).
import './globals.css'
import { ClerkProvider } from '@clerk/nextjs'
import { AuthProvider } from '@/hooks/useAuth'

export const metadata = {
  title:       'Knowledge Copilot',
  description: 'Chat with your documents using RAG',
}

export default function RootLayout({ children }) {
  return (
    <ClerkProvider
      publishableKey={process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
      afterSignInUrl="/dashboard"
      afterSignUpUrl="/dashboard"
    >
      <html lang="en" suppressHydrationWarning>
        <head>
          {/* Runs before paint — prevents dark mode flash */}
          <script dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var t = localStorage.getItem('kc_theme');
                  if (t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                    document.documentElement.classList.add('dark');
                  }
                } catch(e) {}
              })();
            `
          }} />
        </head>
        <body suppressHydrationWarning>
          <AuthProvider>
            {children}
          </AuthProvider>
        </body>
      </html>
    </ClerkProvider>
  )
}