import { useState, useRef, useEffect, useCallback } from 'react'
import { ChatInput, type ChatInputFile } from './chat-input'
import { WelcomePage } from './welcome-page'
import { Bot, User } from 'lucide-react'
import { cn } from '@/utils'
import { useStreamingChat } from '@/hooks/use-streaming-chat'
import ReactMarkdown from 'react-markdown'

interface ChatPanelProps {
  sessionId: string | null
  onCreateSession?: (initialMessage?: string) => string
}

export function ChatPanel({ sessionId, onCreateSession }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [pendingMessage, setPendingMessage] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

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

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  // Send pending message when session is created
  useEffect(() => {
    if (sessionId && pendingMessage) {
      sendMessage(pendingMessage)
      setPendingMessage(null)
    }
  }, [sessionId, pendingMessage, sendMessage])

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
      onCreateSession(value) // Pass message for title generation
      setInput('')
      return
    }

    await sendMessage(value)
    setInput('')
  }, [sessionId, onCreateSession, sendMessage])

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
    <div className="flex h-full flex-col bg-background">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4" ref={scrollRef}>
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-3 rounded-xl p-3 transition-all",
                message.role === 'user'
                  ? "bg-muted/30"
                  : "bg-background"
              )}
            >
              <div className={cn(
                "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border shadow-sm",
                message.role === 'user'
                  ? "bg-background border-border"
                  : "bg-primary/10 border-primary/20 text-primary"
              )}>
                {message.role === 'user' ? (
                  <User className="h-4 w-4" />
                ) : (
                  <Bot className="h-4 w-4" />
                )}
              </div>

              <div className="flex-1 space-y-1 overflow-hidden">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium">
                    {message.role === 'user' ? 'You' : 'MolX Agent'}
                  </span>
                </div>

                <div className="prose prose-sm dark:prose-invert max-w-none break-words text-[14px] leading-relaxed">
                  {message.role === 'assistant' ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4 rounded-xl p-4">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary shadow-sm">
                <Bot className="h-5 w-5 animate-pulse" />
              </div>
              <div className="flex items-center gap-1">
                <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '0ms' }} />
                <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '150ms' }} />
                <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area (Only shown in active chat) */}
      <div className="border-t border-border bg-background/80 p-4 backdrop-blur-sm">
        <div className="mx-auto max-w-3xl">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={handleSubmit}
            onQuickAction={handleQuickAction}
            disabled={isLoading}
            showQuickActions={false} // Hide quick actions in chat view to keep it clean
            variant="default"
          />
          <div className="mt-2 text-center text-xs text-muted-foreground">
            MolX Agent can make mistakes. Please verify important information.
          </div>
        </div>
      </div>
    </div>
  )
}
