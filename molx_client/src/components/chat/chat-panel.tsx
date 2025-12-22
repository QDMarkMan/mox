import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { ChatInput, type ChatInputFile } from './chat-input'
import { WelcomePage } from './welcome-page'
import { ChatMessage, ChatMessageLoading } from './chat-message'
import { useStreamingChat } from '@/hooks/use-streaming-chat'

interface ChatPanelProps {
  sessionId: string | null
  onCreateSession?: (initialMessage?: string) => Promise<string>
  onSyncSessionPreview?: (sessionId: string, preview: string) => void
}

export function ChatPanel({ sessionId, onCreateSession, onSyncSessionPreview }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [pendingMessage, setPendingMessage] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const {
    messages,
    isLoading,
    sendMessage
  } = useStreamingChat({
    sessionId,
    onFinish: () => {
      // Optional: Handle completion
    }
  })

  // Track the last message content to detect streaming updates
  const lastMessageContent = useMemo(() => {
    const lastMessage = messages[messages.length - 1]
    return lastMessage?.content || ''
  }, [messages])

  // Auto-scroll to bottom when messages change or content is streaming
  useEffect(() => {
    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
      }
    }

    // Immediate scroll for better responsiveness
    scrollToBottom()

    // Delayed scroll to handle layout shifts/images
    const timeoutId = setTimeout(scrollToBottom, 100)
    
    return () => clearTimeout(timeoutId)
  }, [messages, lastMessageContent, isLoading])

  // Send pending message when session is created
  useEffect(() => {
    if (sessionId && pendingMessage) {
      sendMessage(pendingMessage)
      onSyncSessionPreview?.(sessionId, pendingMessage)
      setPendingMessage(null)
    }
  }, [sessionId, pendingMessage, sendMessage, onSyncSessionPreview])

  const handleSubmit = useCallback(async (value: string, files?: ChatInputFile[]) => {
    if (!value.trim() && (!files || files.length === 0)) return

    // Note: File uploads are not yet supported by the streaming hook
    if (files && files.length > 0) {
      console.warn('File uploads are not yet supported in streaming mode')
    }

    // If no session exists, create one first
    if (!sessionId && onCreateSession) {
      // Store the message to send after session is created
      setPendingMessage(value)
      await onCreateSession(value)
      setInput('')
      return
    }

    await sendMessage(value)
    if (sessionId && onSyncSessionPreview) {
      onSyncSessionPreview(sessionId, value)
    }
    setInput('')
  }, [sessionId, onCreateSession, onSyncSessionPreview, sendMessage])

  const handleQuickAction = (actionId: string) => {
    console.log('Quick action:', actionId)
    // Handle quick actions
  }

  // If no session is active, show the new Welcome Page
  if (!sessionId) {
    return (
      <div className="flex h-full flex-col bg-background">
        <WelcomePage onInputSubmit={handleSubmit} />
      </div>
    )
  }

  return (
    <div className="flex h-[100vh] flex-col overflow-hidden bg-background">
      {/* Messages Area - scrollable */}
      <div className="min-h-0 flex-1 overflow-y-auto p-4" ref={scrollRef}>
        <div className="mx-auto max-w-3xl space-y-4 pb-4">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isLoading && <ChatMessageLoading />}

          {/* Scroll anchor for auto-scroll */}
          <div ref={messagesEndRef} className="h-px w-full" />
        </div>
      </div>

      {/* Input Area - Fixed at bottom, won't scroll */}
      <div className="flex-shrink-0 border-t border-border bg-background/95 p-4 backdrop-blur-sm">
        <div className="mx-auto max-w-3xl">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={handleSubmit}
            onQuickAction={handleQuickAction}
            disabled={isLoading}
            showQuickActions={false} // Hide quick actions in chat view to keep it clean
            variant="default"
            rows={2}
          />
          <div className="mt-2 text-center text-xs text-muted-foreground">
            MolX Agent can make mistakes. Please verify important information.
          </div>
        </div>
      </div>
    </div>
  )
}

