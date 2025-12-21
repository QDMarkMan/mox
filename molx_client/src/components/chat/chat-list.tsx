import { useState } from 'react'
import { Plus, MessageSquare, Trash2 } from 'lucide-react'
import { cn } from '@/utils'

interface Session {
  id: string
  title: string
  createdAt: string
}

interface ChatListProps {
  activeSessionId: string | null
  onSessionSelect: (id: string | null) => void
}

export function ChatList({ activeSessionId, onSessionSelect }: ChatListProps) {
  const [sessions, setSessions] = useState<Session[]>([
    { id: '1', title: 'SAR Analysis Demo', createdAt: '2024-12-20' },
  ])

  const handleCreateSession = async () => {
    try {
      const response = await fetch('/api/v1/session/create', {
        method: 'POST',
      })
      const data = await response.json()
      const newSession: Session = {
        id: data.session_id,
        title: `New Chat ${sessions.length + 1}`,
        createdAt: new Date().toISOString().split('T')[0],
      }
      setSessions([newSession, ...sessions])
      onSessionSelect(newSession.id)
    } catch (error) {
      console.error('Failed to create session:', error)
      // Create local session as fallback
      const fallbackSession: Session = {
        id: `local-${Date.now()}`,
        title: `New Chat ${sessions.length + 1}`,
        createdAt: new Date().toISOString().split('T')[0],
      }
      setSessions([fallbackSession, ...sessions])
      onSessionSelect(fallbackSession.id)
    }
  }

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSessions(sessions.filter((s) => s.id !== id))
    if (activeSessionId === id) {
      onSessionSelect(null)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={handleCreateSession}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-background px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          <Plus className="h-4 w-4" />
          New Chat
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-auto px-3">
        <div className="space-y-1">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => onSessionSelect(session.id)}
              className={cn(
                'group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors',
                activeSessionId === session.id
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent'
              )}
            >
              <div className="flex items-center gap-2 overflow-hidden">
                <MessageSquare className="h-4 w-4 shrink-0" />
                <span className="truncate">{session.title}</span>
              </div>
              <button
                onClick={(e) => handleDeleteSession(session.id, e)}
                className={cn(
                  'shrink-0 rounded p-1 opacity-0 transition-opacity hover:bg-destructive/20 group-hover:opacity-100',
                  activeSessionId === session.id && 'hover:bg-primary-foreground/20'
                )}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border p-3">
        <p className="text-center text-xs text-muted-foreground">
          {sessions.length} conversation{sessions.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  )
}
