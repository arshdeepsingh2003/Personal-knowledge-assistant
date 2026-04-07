import './globals.css'

export const metadata = {
  title:       'Knowledge Copilot',
  description: 'Chat with your documents using RAG',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}