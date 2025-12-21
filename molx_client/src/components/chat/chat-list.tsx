
import {
  Plus,
  MessageSquare,
  Trash2,
  User
} from 'lucide-react'
import { cn } from '@/utils'
import type { ChatSession } from '@/App'

interface ChatListProps {
  activeId: string | null
  onSelect: (id: string | null) => void
  sessions: ChatSession[]
  onDeleteSession: (id: string) => void
}

export function ChatList({ activeId, onSelect, sessions, onDeleteSession }: ChatListProps) {
  const createNewSession = () => {
    onSelect(null) // Null means "New Chat" / Welcome Page
  }

  const deleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    onDeleteSession(id)
  }

  // Group sessions by date
  const today = new Date()
  const todaySessions = sessions.filter(s => {
    const d = new Date(s.createdAt)
    return d.toDateString() === today.toDateString()
  })
  const olderSessions = sessions.filter(s => {
    const d = new Date(s.createdAt)
    return d.toDateString() !== today.toDateString()
  })

  return (
    <div className="flex h-full flex-col bg-background/50 backdrop-blur-xl">
      {/* App Branding / Header in Sidebar */}
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
          >
            <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" />
            <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
            <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
            <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
          </svg>
        </div>
        <span className="text-base font-semibold tracking-tight">MolX Agent</span>
      </div>

      {/* Main Navigation */}
      <div className="px-3 py-2">
        <button
          onClick={createNewSession}
          className={cn(
            "group flex w-full items-center justify-between rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 text-sm font-medium text-primary shadow-sm transition-all hover:bg-primary/10 hover:shadow-md hover:border-primary/30",
            !activeId && "ring-1 ring-primary/20 bg-primary/10"
          )}
        >
          <span className="flex items-center gap-2">
            <Plus className="h-4 w-4 text-primary group-hover:scale-110 transition-transform" />
            <span>New Task</span>
          </span>
          <span className="rounded bg-background/50 px-1.5 py-0.5 text-[10px] text-primary/70">⌘N</span>
        </button>

        <div className="mt-3 space-y-1">
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground">
            <span className="text-base">🧩</span>
            <span>Knowledge Garden</span>
          </button>
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground">
            <span className="text-base">🧭</span>
            <span>Explore More</span>
          </button>
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-2 py-2 scrollbar-thin">
        {/* Today's Sessions */}
        {todaySessions.length > 0 && (
          <div className="mb-4">
            <div className="mb-1 flex items-center justify-between px-2">
              <span className="text-[11px] font-medium text-muted-foreground/70 uppercase tracking-wider">Today</span>
            </div>
            <div className="space-y-0.5">
              {todaySessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelect(session.id)}
                  className={cn(
                    "group relative flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] transition-colors hover:bg-muted/50",
                    activeId === session.id ? "bg-muted font-medium text-foreground" : "text-muted-foreground"
                  )}
                >
                  <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-70" />
                  <span className="truncate text-left">{session.title}</span>

                  <div
                    onClick={(e) => deleteSession(e, session.id)}
                    className="absolute right-2 hidden rounded-md p-0.5 hover:bg-background group-hover:block"
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Older Sessions */}
        {olderSessions.length > 0 && (
          <div className="mb-4">
            <div className="mb-1 flex items-center justify-between px-2">
              <span className="text-[11px] font-medium text-muted-foreground/70 uppercase tracking-wider">History</span>
              <span className="text-[10px]">🕒</span>
            </div>
            <div className="space-y-0.5">
              {olderSessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelect(session.id)}
                  className={cn(
                    "group relative flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] transition-colors hover:bg-muted/50",
                    activeId === session.id ? "bg-muted font-medium text-foreground" : "text-muted-foreground"
                  )}
                >
                  <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-70" />
                  <span className="truncate text-left">{session.title}</span>

                  <div
                    onClick={(e) => deleteSession(e, session.id)}
                    className="absolute right-2 hidden rounded-md p-0.5 hover:bg-background group-hover:block"
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {sessions.length === 0 && (
          <div className="px-2 py-4 text-center">
            <p className="text-[11px] text-muted-foreground/50 italic">
              No conversations yet
            </p>
            <p className="mt-1 text-[10px] text-muted-foreground/40">
              Click "New Task" to start
            </p>
          </div>
        )}
      </div>

      {/* User Profile / Footer */}
      <div className="border-t border-border p-2">
        <div className="flex items-center gap-2.5 rounded-md p-1.5 transition-colors hover:bg-muted/50">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-white">
            <User className="h-3.5 w-3.5" />
          </div>
          <div className="flex flex-1 flex-col overflow-hidden">
            <span className="truncate text-[13px] font-medium">User</span>
            <span className="truncate text-[10px] text-muted-foreground">Pro Plan</span>
          </div>
          <span className="text-muted-foreground text-xs">⚙️</span>
        </div>
      </div>
    </div>
  )
}
