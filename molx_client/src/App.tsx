import { useState, useCallback } from 'react'
import { ChatList } from '@/components/chat/chat-list'
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
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <div className="w-[280px] shrink-0 border-r border-border bg-muted/30">
        <ChatList
          activeId={sessionId}
          onSelect={setSessionId}
          sessions={sessions}
          onDeleteSession={deleteSession}
        />
      </div>

      {/* Main Content */}
      <main className="flex flex-1 flex-col overflow-hidden bg-background">
        <ChatPanel
          sessionId={sessionId}
          onCreateSession={createSession}
        />
      </main>
    </div>
  )
}
