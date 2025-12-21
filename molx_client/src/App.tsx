import { useState, useCallback } from 'react'
import { SidebarLayout } from '@/components/layout/sidebar-layout'
import { ChatPanel } from '@/components/chat/chat-panel'

export interface ChatSession {
  id: string
  title: string
  createdAt: Date
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])

  // Create a new session and return the new session ID
  const createSession = useCallback((initialMessage?: string) => {
    const newId = `session-${Date.now()}`
    const title = initialMessage?.slice(0, 30) || 'New Task'
    const newSession: ChatSession = {
      id: newId,
      title: title + (initialMessage && initialMessage.length > 30 ? '...' : ''),
      createdAt: new Date()
    }
    setSessions(prev => [newSession, ...prev])
    setSessionId(newId)
    return newId
  }, [])

  // Delete a session
  const deleteSession = useCallback((id: string) => {
    setSessions(prev => prev.filter(s => s.id !== id))
    if (sessionId === id) {
      setSessionId(null)
    }
  }, [sessionId])

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
      />
    </SidebarLayout>
  )
}
