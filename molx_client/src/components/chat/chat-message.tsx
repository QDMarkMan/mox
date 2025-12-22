/**
 * ChatMessage - Renders a single chat message (user or assistant).
 * Includes avatar, content, and action buttons for assistant messages.
 */
import { cn } from '@/utils'
import ReactMarkdown from 'react-markdown'
import type { StreamingMessage } from '@/hooks/use-streaming-chat'

// SVG Icons as components
const UserIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
)

const BotIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" />
    <path d="M20 14h2" />
    <path d="M15 13v2" />
    <path d="M9 13v2" />
  </svg>
)

const ThumbsUpIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M7 10v12" />
    <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
  </svg>
)

const ThumbsDownIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 14V2" />
    <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
  </svg>
)

const CopyIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
  </svg>
)

interface ChatMessageProps {
  message: StreamingMessage
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
  }

  return (
    <div
      className={cn(
        "group flex gap-3 rounded-lg p-3 transition-all duration-200 ease-out",
        "hover:bg-muted/30 hover:shadow-sm",
        isUser ? "bg-muted/40 hover:bg-muted/50" : ""
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
        isUser
          ? "bg-muted text-muted-foreground"
          : "bg-primary/10 text-primary"
      )}>
        {isUser ? <UserIcon /> : <BotIcon />}
      </div>

      {/* Content */}
      <div className="flex-1 space-y-1 overflow-hidden pt-0.5">
        <span className={cn(
          "text-xs font-medium",
          isUser ? "text-muted-foreground" : "text-primary"
        )}>
          {isUser ? 'You' : 'MolX Agent'}
        </span>

        <div className="prose prose-sm dark:prose-invert max-w-none break-words text-[14px] leading-relaxed">
          {isUser ? (
            <p className="whitespace-pre-wrap m-0">{message.content}</p>
          ) : (
            <ReactMarkdown>{message.content}</ReactMarkdown>
          )}
        </div>

        {!isUser && message.thinking && (
          <div className="mt-2 rounded-md border border-amber-200/60 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-400/40 dark:bg-amber-950/30 dark:text-amber-200">
            {message.thinking.status === 'analyzing'
              ? (message.thinking.message || 'Analyzing intent…')
              : (
                <span>
                  Intent: <span className="font-semibold">{message.thinking.intent || 'Unknown'}</span>
                  {message.thinking.confidence !== undefined && (
                    <span className="ml-1 text-muted-foreground">({Math.round(message.thinking.confidence * 100)}%)</span>
                  )}
                </span>
              )}
          </div>
        )}

        {!isUser && message.status && message.status.length > 0 && (
          <div className="mt-3 max-h-48 overflow-y-auto rounded-md border border-border/50 bg-muted/40 p-2 font-mono text-[12px] leading-5 text-muted-foreground">
            {message.status.map((line, idx) => (
              <div key={`${message.id}-status-${idx}`} className="whitespace-pre-wrap">
                {line}
              </div>
            ))}
          </div>
        )}

        {/* Action buttons for assistant messages */}
        {!isUser && (
          <div className="flex items-center gap-1 pt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              className="p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title="Like"
            >
              <ThumbsUpIcon />
            </button>
            <button
              className="p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title="Dislike"
            >
              <ThumbsDownIcon />
            </button>
            <button
              className="p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title="Copy"
              onClick={handleCopy}
            >
              <CopyIcon />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * ChatMessageLoading - Loading indicator for when assistant is responding.
 */
export function ChatMessageLoading() {
  return (
    <div className="flex gap-4 rounded-xl p-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary shadow-sm">
        <BotIcon className="animate-pulse" />
      </div>
      <div className="flex items-center gap-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '0ms' }} />
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '150ms' }} />
        <span className="h-2 w-2 animate-bounce rounded-full bg-primary/40" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  )
}
