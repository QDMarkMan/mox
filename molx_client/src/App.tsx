import { useState } from 'react'
import { ChatList } from '@/components/chat/chat-list'
import { ChatPanel } from '@/components/chat/chat-panel'
import { Header } from '@/components/layout/header'

function App() {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  return (
    <div className="flex h-screen flex-col bg-background">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Chat List */}
        <aside className="w-64 border-r border-border bg-card">
          <ChatList
            activeSessionId={activeSessionId}
            onSessionSelect={setActiveSessionId}
          />
        </aside>

        {/* Main - Chat Panel */}
        <main className="flex-1 overflow-hidden">
          <ChatPanel sessionId={activeSessionId} />
        </main>
      </div>
    </div>
  )
}

export default App
