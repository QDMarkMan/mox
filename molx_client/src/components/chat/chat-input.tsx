import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, Plus, Trash2 } from 'lucide-react'
import { cn } from '@/utils'

// Quick action button configuration
const quickActions = [
  {
    id: 'analyze-sar',
    label: 'Analyze SAR',
    emoji: '🧪',
    color: 'text-emerald-500',
    bgColor: 'hover:bg-emerald-50 dark:hover:bg-emerald-950/30'
  },
  {
    id: 'drug-design',
    label: 'Drug Design',
    emoji: '✨',
    color: 'text-blue-500',
    bgColor: 'hover:bg-blue-50 dark:hover:bg-blue-950/30'
  },
  {
    id: 'predict-properties',
    label: 'Predict Properties',
    emoji: '📊',
    color: 'text-orange-500',
    bgColor: 'hover:bg-orange-50 dark:hover:bg-orange-950/30'
  },
  {
    id: 'summarize-data',
    label: 'Summarize Data',
    emoji: '📝',
    color: 'text-purple-500',
    bgColor: 'hover:bg-purple-50 dark:hover:bg-purple-950/30'
  },
  {
    id: 'generate-report',
    label: 'Generate Report',
    emoji: '📋',
    color: 'text-pink-500',
    bgColor: 'hover:bg-pink-50 dark:hover:bg-pink-950/30'
  },
  {
    id: 'more',
    label: 'More',
    emoji: '⋯',
    color: 'text-gray-500',
    bgColor: 'hover:bg-gray-50 dark:hover:bg-gray-800/30'
  }
]

export interface ChatInputFile {
  id: string
  file: File
  name: string
  size: number
  type: string
}

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (message: string, files?: ChatInputFile[]) => void
  onQuickAction?: (actionId: string) => void
  disabled?: boolean
  placeholder?: string
  // Display configuration
  variant?: 'default' | 'welcome'
  rows?: number
  // Feature toggles
  showQuickActions?: boolean
  showAgentMode?: boolean
  showKnowledgeBase?: boolean
  showFileUpload?: boolean
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  onQuickAction,
  disabled = false,
  placeholder = 'Give MolX a task, let it plan, call tools, and execute for you...',
  variant = 'default',
  rows = 1,
  showQuickActions = false,
  showAgentMode = false,
  showKnowledgeBase = false,
  showFileUpload = true
}: ChatInputProps) {
  const [files, setFiles] = useState<ChatInputFile[]>([])
  const [useKnowledgeBase, setUseKnowledgeBase] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Computed values based on variant
  const isWelcome = variant === 'welcome'
  const textareaRows = isWelcome ? 4 : rows

  // Auto-resize textarea for welcome variant
  useEffect(() => {
    if (textareaRef.current && isWelcome) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [value, isWelcome])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (!selectedFiles) return

    const newFiles: ChatInputFile[] = Array.from(selectedFiles).map((file) => ({
      id: 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9),
      file,
      name: file.name,
      size: file.size,
      type: file.type
    }))

    setFiles((prev) => [...prev, ...newFiles])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [])

  const removeFile = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId))
  }, [])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!value.trim() && files.length === 0) return
    if (disabled) return

    onSubmit(value, files.length > 0 ? files : undefined)
    setFiles([])
  }, [value, files, disabled, onSubmit])

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }, [handleSubmit])

  const handleQuickActionClick = useCallback((actionId: string) => {
    if (onQuickAction) {
      onQuickAction(actionId)
    }
  }, [onQuickAction])

  const canSubmit = (value.trim() || files.length > 0) && !disabled

  return (
    <div className="w-full">
      {/* Quick Actions (shown when enabled) */}
      {showQuickActions && (
        <div className="mb-4 flex flex-wrap justify-center gap-2">
          {quickActions.map((action) => (
            <button
              key={action.id}
              onClick={() => handleQuickActionClick(action.id)}
              disabled={disabled}
              className={cn(
                'flex items-center gap-2 rounded-full border border-border/60 bg-background px-4 py-2 text-sm font-medium transition-all duration-200',
                'hover:border-border hover:shadow-sm',
                action.bgColor,
                disabled && 'cursor-not-allowed opacity-50'
              )}
            >
              <span className="text-base">{action.emoji}</span>
              <span className="text-foreground/80">{action.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* Input Container */}
      <div className={cn(
        "input-gradient-border relative rounded-xl border border-border/60 bg-muted/30 transition-all duration-500",
        "focus-within:border-transparent focus-within:bg-background focus-within:shadow-md focus-within:shadow-primary/5",
        isWelcome ? "p-3 shadow-sm" : "p-2"
      )}>
        {/* Agent Mode Toggle (shown in welcome variant) */}
        {showAgentMode && (
          <div className="mb-1.5 absolute right-2 flex justify-end z-10">
            <button
              // onClick={() => setIsAgentMode(!isAgentMode)}
              // className={cn(
              //   "flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors",
              //   isAgentMode
              //     ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300"
              //     : "bg-muted text-muted-foreground"
              // )}
              className="flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 text-white shadow-sm shadow-purple-500/20"

            >
              <span className="text-sm">🤖</span>
              <span>AGENT MODE</span>
              {/* <span className={cn("ml-1 font-bold", isAgentMode ? "opacity-100" : "opacity-50")}>
                {isAgentMode ? "ON" : "OFF"}
              </span> */}
            </button>
          </div>
        )}

        {/* File Previews (above textarea when files exist) */}
        {files.length > 0 && (
          <div className={cn(
            "flex flex-wrap gap-2 border-b border-border/40",
            isWelcome ? "pb-2 mb-2" : "pb-2 mb-2 px-1"
          )}>
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center gap-1 rounded-md border border-border bg-background px-2 py-0.5 text-[10px]"
              >
                <span className="text-muted-foreground">📎</span>
                <span className="max-w-[80px] truncate">{file.name}</span>
                <button
                  onClick={() => removeFile(file.id)}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          rows={textareaRows}
          className={cn(
            "w-full resize-none bg-transparent text-foreground placeholder:text-muted-foreground/60 focus:outline-none",
            isWelcome ? "text-[15px] leading-relaxed" : "text-sm leading-normal"
          )}
          onKeyDown={handleKeyDown}
        />

        {/* Footer Actions */}
        <div className={cn(
          "flex items-center justify-between",
          isWelcome ? "mt-2" : "mt-1"
        )}>
          {/* Left: File Upload */}
          <div className="flex items-center gap-1">
            {showFileUpload && (
              <>
                <input
                  type="file"
                  multiple
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled}
                  className="group rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  title="Attach files"
                >
                  <Plus className="h-4 w-4 transition-transform group-hover:rotate-90" />
                </button>
              </>
            )}
          </div>

          {/* Right: Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={cn(
              "flex items-center gap-2 rounded-lg font-medium transition-all",
              isWelcome ? "px-3 py-1.5 text-[13px]" : "px-3 py-1 text-sm",
              canSubmit
                ? "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            <span>{isWelcome ? 'Submit' : 'Send'}</span>
            <Send className={cn(isWelcome ? "h-3.5 w-3.5" : "h-4 w-4")} />
          </button>
        </div>
      </div>

      {/* Knowledge Base Toggle (shown when enabled) */}
      {showKnowledgeBase && (
        <div className="mt-2 flex justify-end">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Knowledge Base</span>
            <button
              onClick={() => setUseKnowledgeBase(!useKnowledgeBase)}
              className={cn(
                "relative h-4 w-8 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary/20",
                useKnowledgeBase ? "bg-primary" : "bg-muted"
              )}
            >
              <span
                className={cn(
                  "absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-white transition-transform",
                  useKnowledgeBase ? "translate-x-4" : "translate-x-0"
                )}
              />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
