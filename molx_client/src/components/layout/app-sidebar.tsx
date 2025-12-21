/**
 * AppSidebar - Main application sidebar using shadcn/ui sidebar components.
 * Contains: Branding, New Task button, Navigation, Session History, User Profile.
 */
import {
  Plus,
  MessageSquare,
  Trash2
} from 'lucide-react'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import type { ChatSession } from '@/App'

interface AppSidebarProps {
  sessionId: string | null
  onSelectSession: (id: string | null) => void
  sessions: ChatSession[]
  onDeleteSession: (id: string) => void
}

export function AppSidebar({
  sessionId,
  onSelectSession,
  sessions,
  onDeleteSession
}: AppSidebarProps) {
  const createNewSession = () => {
    onSelectSession(null) // Null means "New Chat" / Welcome Page
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
    <Sidebar collapsible="icon">
      {/* Header with Branding */}
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground text-lg">
                  🚀
                </div>
                <div className="flex flex-col gap-0.5 leading-none">
                  <span className="font-semibold text-md">MolX Agent</span>
                  {/* <span className="text-xs text-muted-foreground">v1.0</span> */}
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {/* New Task Button */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={createNewSession}
                  isActive={!sessionId}
                  tooltip="New Task"
                  className="bg-primary/10 hover:bg-primary/20 text-primary border font-medium"
                >
                  <Plus className="size-4" />
                  <span>New Task</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Knowledge Garden">
                  <span className="text-sm">📚</span>
                  <span>Knowledge Garden</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Explore More">
                  <span className="text-sm">🧭</span>
                  <span>Explore More</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Today's Sessions */}
        {todaySessions.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>Today</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {todaySessions.map((session) => (
                  <SidebarMenuItem key={session.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectSession(session.id)}
                      isActive={sessionId === session.id}
                      tooltip={session.title}
                    >
                      <MessageSquare className="size-4" />
                      <span>{session.title}</span>
                    </SidebarMenuButton>
                    <SidebarMenuAction
                      onClick={(e) => deleteSession(e, session.id)}
                      showOnHover
                    >
                      <Trash2 className="size-4" />
                      <span className="sr-only">Delete</span>
                    </SidebarMenuAction>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {/* Older Sessions */}
        {olderSessions.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>History</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {olderSessions.map((session) => (
                  <SidebarMenuItem key={session.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectSession(session.id)}
                      isActive={sessionId === session.id}
                      tooltip={session.title}
                    >
                      <MessageSquare className="size-4" />
                      <span>{session.title}</span>
                    </SidebarMenuButton>
                    <SidebarMenuAction
                      onClick={(e) => deleteSession(e, session.id)}
                      showOnHover
                    >
                      <Trash2 className="size-4" />
                      <span className="sr-only">Delete</span>
                    </SidebarMenuAction>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {/* Empty State */}
        {sessions.length === 0 && (
          <SidebarGroup>
            <SidebarGroupContent>
              <div className="px-2 py-4 text-center">
                <p className="text-xs text-muted-foreground">
                  No conversations yet
                </p>
                <p className="mt-1 text-[10px] text-muted-foreground/60">
                  Click "New Task" to start
                </p>
              </div>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>

      {/* Footer with User Profile */}
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" tooltip="User Profile">
              <div className="flex aspect-square size-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-white text-sm">
                👤
              </div>
              <div className="flex flex-col gap-0.5 leading-none">
                <span className="font-medium">User</span>
                <span className="text-xs text-muted-foreground">Pro Plan</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        {/* Collapse Toggle */}
        <div className="flex items-center justify-between px-2 pt-2 border-t border-sidebar-border">
          <span className="text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">V1.0</span>
          <SidebarTrigger />
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  )
}
