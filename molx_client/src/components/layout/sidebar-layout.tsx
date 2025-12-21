/**
 * SidebarLayout - Layout wrapper using shadcn/ui Sidebar components.
 * Provides: SidebarProvider, AppSidebar, SidebarInset, and SidebarTrigger.
 */
import { ReactNode } from 'react'
import {
  SidebarProvider,
  SidebarInset,
} from '@/components/ui/sidebar'
import { AppSidebar } from './app-sidebar'
import type { ChatSession } from '@/App'

interface SidebarLayoutProps {
  children: ReactNode
  sessionId: string | null
  onSelectSession: (id: string | null) => void
  sessions: ChatSession[]
  onDeleteSession: (id: string) => void
}

export function SidebarLayout({
  children,
  sessionId,
  onSelectSession,
  sessions,
  onDeleteSession
}: SidebarLayoutProps) {
  return (
    <SidebarProvider>
      <AppSidebar
        sessionId={sessionId}
        onSelectSession={onSelectSession}
        sessions={sessions}
        onDeleteSession={onDeleteSession}
      />
      <SidebarInset>
        <main className="flex flex-1 flex-col overflow-hidden">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
