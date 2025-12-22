import { useState, useCallback, useEffect } from 'react'
import { SidebarLayout } from '@/components/layout/sidebar-layout'
import { ChatPanel } from '@/components/chat/chat-panel'

const SESSION_API = '/api/v1/session'

export interface ChatSession {
  id: string
  title: string
  createdAt: Date
  lastActivity: Date
  messageCount: number
}

const fallbackTitle = 'New Task'

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])

  const loadSessions = useCallback(async () => {
    try {
      const response = await fetch(SESSION_API)
      if (!response.ok) {
        return
      }
      const data = await response.json()
      const normalized: ChatSession[] = (data?.sessions ?? []).map((session: any) => ({
        id: session.session_id,
        title: session.preview?.trim() || fallbackTitle,
        createdAt: new Date(session.created_at),
        lastActivity: new Date(session.last_activity),
        messageCount: session.message_count ?? 0,
      }))
      normalized.sort((a, b) => b.lastActivity.getTime() - a.lastActivity.getTime())
      setSessions(normalized)
    } catch (err) {
      console.warn('Failed to load sessions', err)
    }
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  const createSession = useCallback(async (initialMessage?: string) => {
    const response = await fetch(`${SESSION_API}/create`, { method: 'POST' })
    if (!response.ok) {
      throw new Error('Failed to create session')
    }
    const data = await response.json()
    const titleSeed = initialMessage?.trim()
    const newSession: ChatSession = {
      id: data.session_id,
      title: titleSeed?.slice(0, 60) || fallbackTitle,
      createdAt: new Date(data.created_at),
      lastActivity: new Date(data.created_at),
      messageCount: 0,
    }
    setSessions(prev => [newSession, ...prev])
    setSessionId(newSession.id)
    return newSession.id
  }, [])

  const deleteSession = useCallback(async (id: string) => {
    try {
      const response = await fetch(`${SESSION_API}/${id}`, { method: 'DELETE' })
      if (!response.ok) {
        throw new Error('Failed to delete session')
      }
      setSessions(prev => prev.filter(s => s.id !== id))
      if (sessionId === id) {
        setSessionId(null)
      }
    } catch (err) {
      console.error('Unable to delete session', err)
    }
  }, [sessionId])

  const handleSyncSessionPreview = useCallback((id: string, preview: string) => {
    setSessions(prev =>
      prev.map(session =>
        session.id === id
          ? {
              ...session,
              title: preview.slice(0, 80) || fallbackTitle,
              lastActivity: new Date(),
              messageCount: session.messageCount + 1,
            }
          : session
      )
    )
  }, [])

  return (
    <SidebarLayout
      sessionId={sessionId}
      onSelectSession={setSessionId}
      sessions={sessions}
      onDeleteSession={deleteSession}
    >
      <ChatPanel
        sessionId={sessionId}
        onCreateSession={createSession}
        onSyncSessionPreview={handleSyncSessionPreview}
      />
    </SidebarLayout>
  )
}
