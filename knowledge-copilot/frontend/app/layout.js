import './globals.css'

export const metadata = {
  title:       'Knowledge Copilot',
  description: 'Chat with your documents using RAG',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Theme script — runs before paint to avoid flash */}
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
        {children}
      </body>
    </html>
  )
}