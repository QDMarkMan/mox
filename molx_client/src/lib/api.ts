const ABSOLUTE_URL = /^https?:\/\//i

const rawBase = (import.meta.env.VITE_API_BASE_URL ?? '').trim()
export const API_BASE_URL = rawBase ? rawBase.replace(/\/$/, '') : ''

/** API prefix exposed by the FastAPI server. */
export const API_PREFIX = '/api/v1'

/**
 * Resolve a path (e.g. `/api/v1/session`) against the configured API base URL.
 * Falls back to relative paths when no base is set so that the dev proxy works.
 */
export function apiUrl(path: string): string {
  if (!path) {
    return API_BASE_URL || ''
  }

  if (ABSOLUTE_URL.test(path)) {
    return path
  }

  const normalized = path.startsWith('/') ? path : `/${path}`
  return API_BASE_URL ? `${API_BASE_URL}${normalized}` : normalized
}
