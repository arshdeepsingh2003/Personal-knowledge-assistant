// FILE: frontend/middleware.js
// IMPORTANT: This file MUST be at the ROOT of the frontend folder.
//            i.e. frontend/middleware.js — NOT inside any subfolder.
//
// PURPOSE: Runs at the edge before every request.
//          Redirects unauthenticated users away from protected pages.
//          Redirects authenticated users away from /login and /signup.
import { NextResponse } from 'next/server'

// Pages that require the user to be logged in
const PROTECTED_PREFIXES = ['/dashboard']

// Pages only for logged-out users
const AUTH_ONLY_PATHS = ['/login', '/signup']

export function middleware(request) {
  const { pathname } = request.nextUrl

  // We set a cookie called "kc_session" on login (in the browser).
  // This cookie is not httpOnly — it's just a presence flag.
  // The actual JWT lives in sessionStorage and is validated server-side.
  const isAuthed = request.cookies.has('kc_session')

  const needsAuth  = PROTECTED_PREFIXES.some(p => pathname.startsWith(p))
  const isAuthPage = AUTH_ONLY_PATHS.some(p => pathname === p || pathname.startsWith(p + '/'))

  // Not logged in, trying to access a protected page → redirect to /login
  if (needsAuth && !isAuthed) {
    const loginUrl = new URL('/login', request.url)
    loginUrl.searchParams.set('next', pathname)  // remember where they were going
    return NextResponse.redirect(loginUrl)
  }

  // Already logged in, trying to visit /login or /signup → go to /dashboard
  if (isAuthPage && isAuthed) {
    return NextResponse.redirect(new URL('/dashboard', request.url))
  }

  return NextResponse.next()
}

// Only run middleware on these routes (improves performance)
export const config = {
  matcher: [
    '/dashboard/:path*',
    '/login',
    '/signup',
    '/sign-in',
    '/sign-up',
    '/sso-callback',
  ],
}