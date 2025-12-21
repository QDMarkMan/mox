import { useRef, useEffect } from 'react'
import { Send, Bot, User, Lightbulb } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { cn } from '@/utils'
import { useStreamingChat } from '@/hooks'

interface ChatPanelProps {
  sessionId: string | null
}

export function ChatPanel({ sessionId }: ChatPanelProps) {
  const {
    messages,
    input,
    setInput,
    isLoading,
    thinking,
    sendMessage,
  } = useStreamingChat({
    api: '/api/v1/agent/stream',
    sessionId,
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, thinking])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const message = input
    setInput('')
    await sendMessage(message)
  }

  if (!sessionId) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <Bot className="mx-auto h-16 w-16 text-muted-foreground/50" />
          <h2 className="mt-4 text-lg font-medium text-muted-foreground">
            Welcome to MolX Agent
          </h2>
          <p className="mt-2 text-sm text-muted-foreground/70">
            Select a conversation or create a new one to start
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center py-20">
              <div className="text-center">
                <Bot className="mx-auto h-12 w-12 text-primary/50" />
                <p className="mt-4 text-muted-foreground">
                  Ask me about SAR analysis, drug design, or molecular properties
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className="space-y-2">
                {/* Thinking Block (for assistant messages) */}
                {message.role === 'assistant' && message.thinking && message.thinking.status === 'complete' && (
                  <div className="ml-11 rounded-lg border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-950/30">
                    <div className="flex items-center gap-2 text-sm text-amber-700 dark:text-amber-400">
                      <Lightbulb className="h-4 w-4" />
                      <span className="font-medium">Thinking</span>
                      {message.thinking.intent && (
                        <span className="rounded bg-amber-200 px-1.5 py-0.5 text-xs dark:bg-amber-800">
                          {message.thinking.intent}
                        </span>
                      )}
                      {message.thinking.confidence !== undefined && (
                        <span className="text-xs text-amber-600 dark:text-amber-500">
                          ({(message.thinking.confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                    {message.thinking.reasoning && (
                      <p className="mt-1 text-sm text-amber-600 dark:text-amber-400">
                        {message.thinking.reasoning}
                      </p>
                    )}
                  </div>
                )}

                {/* Message Content */}
                <div
                  className={cn(
                    'flex gap-3',
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  {message.role === 'assistant' && (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                      <Bot className="h-5 w-5 text-primary-foreground" />
                    </div>
                  )}
                  <div
                    className={cn(
                      'max-w-[80%] rounded-lg px-4 py-2',
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted'
                    )}
                  >
                    {message.role === 'assistant' ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm">{message.content}</p>
                    )}
                  </div>
                  {message.role === 'user' && (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary">
                      <User className="h-5 w-5 text-secondary-foreground" />
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Live Thinking Indicator */}
          {thinking && thinking.status === 'analyzing' && (
            <div className="ml-11 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-950/30">
              <Lightbulb className="h-4 w-4 animate-pulse text-amber-600 dark:text-amber-400" />
              <span className="text-sm text-amber-700 dark:text-amber-400">
                {thinking.message || 'Analyzing...'}
              </span>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && !thinking && messages[messages.length - 1]?.role === 'assistant' &&
            !messages[messages.length - 1]?.content && (
              <div className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                  <Bot className="h-5 w-5 text-primary-foreground" />
                </div>
                <div className="rounded-lg bg-muted px-4 py-2">
                  <div className="flex gap-1">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" style={{ animationDelay: '0ms' }} />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" style={{ animationDelay: '150ms' }} />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Form */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about SAR analysis, drug design..."
              className="flex-1 rounded-lg border border-input bg-background px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="rounded-lg bg-primary px-4 py-2 text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            {thinking ? '🧠 Thinking...' : isLoading ? '🔄 Streaming...' : '✨ Streaming enabled'}
          </p>
        </form>
      </div>
    </div>
  )
}
