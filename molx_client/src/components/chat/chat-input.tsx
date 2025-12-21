import { useState, useRef, useCallback, useEffect } from 'react'
import {
  Send,
  Plus,
  Trash2,
  MessageSquare
} from 'lucide-react'
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
  showQuickActions?: boolean
  variant?: 'default' | 'welcome'
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  onQuickAction,
  disabled = false,
  placeholder = 'Give MolX a task, let it plan, call tools, and execute for you...',
  showQuickActions = true,
  variant = 'default'
}: ChatInputProps) {
  const [files, setFiles] = useState<ChatInputFile[]>([])
  const [isAgentMode, setIsAgentMode] = useState(true)
  const [useKnowledgeBase, setUseKnowledgeBase] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight} px`
    }
  }, [value])

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

  const handleQuickActionClick = useCallback((actionId: string) => {
    if (onQuickAction) {
      onQuickAction(actionId)
    }
  }, [onQuickAction])

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  // Welcome Variant (Large Text Area)
  if (variant === 'welcome') {
    return (
      <div className="w-full">
        <div className="relative rounded-xl border border-border/60 bg-muted/30 p-3 shadow-sm transition-all duration-200 focus-within:border-primary/40 focus-within:bg-background focus-within:shadow-md focus-within:shadow-primary/5">
          {/* Header: Agent Mode Toggle */}
          <div className="mb-1.5 absolute right-2 flex justify-end">
            <button
              onClick={() => setIsAgentMode(!isAgentMode)}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors",
                isAgentMode ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300" : "bg-muted text-muted-foreground"
              )}
            >
              <span className="text-sm">🤖</span>
              <span>AGENT MODE</span>
              <span className={cn("ml-1 font-bold", isAgentMode ? "opacity-100" : "opacity-50")}>
                {isAgentMode ? "ON" : "OFF"}
              </span>
            </button>
          </div>

          {/* Text Area */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            disabled={disabled}
            rows={4}
            className="w-full resize-none bg-transparent text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSubmit(e)
              }
            }}
          />

          {/* Footer Actions */}
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-1">
              <input
                type="file"
                multiple
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileSelect}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="group rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                title="Attach files"
              >
                <Plus className="h-4 w-4 transition-transform group-hover:rotate-90" />
              </button>

              {/* File Previews */}
              {files.length > 0 && (
                <div className="flex gap-1 overflow-x-auto">
                  {files.map((file, i) => (
                    <div key={i} className="flex items-center gap-1 rounded-md border border-border bg-background px-2 py-0.5 text-[10px]">
                      <span className="max-w-[80px] truncate">{file.name}</span>
                      <button onClick={() => removeFile(file.id)} className="text-muted-foreground hover:text-destructive">
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              {/* <button className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
                <span className="text-sm">⚙️</span>
              </button> */}
              <button
                onClick={handleSubmit}
                disabled={(!value.trim() && files.length === 0) || disabled}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-3 py-1.5 text-[13px] font-medium transition-all",
                  (value.trim() || files.length > 0) && !disabled
                    ? "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90"
                    : "bg-muted text-muted-foreground cursor-not-allowed"
                )}
              >
                <span>Submit</span>
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
        {/* Knowledge Base Toggle */}
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
      </div>
    )
  }

  // Default Variant (Compact Bar)
  return (
    <div className="w-full">
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

      <div className="relative rounded-2xl border border-border/60 bg-background shadow-sm transition-all duration-200 focus-within:border-primary/40 focus-within:shadow-md focus-within:shadow-primary/5">
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 border-b border-border/40 p-3">
            {files.map((file) => (
              <div key={file.id} className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-1.5 text-sm">
                <span className="text-muted-foreground">📎</span>
                <span className="max-w-[150px] truncate text-foreground/80">{file.name}</span>
                <span className="text-xs text-muted-foreground">({formatFileSize(file.size)})</span>
                <button onClick={() => removeFile(file.id)} className="ml-1 rounded-full p-0.5 hover:bg-destructive/10 hover:text-destructive">
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-center gap-2 p-3">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <Plus className="h-5 w-5" />
          </button>

          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            disabled={disabled}
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
          />

          <button type="button" className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground">
            <MessageSquare className="h-5 w-5" />
          </button>

          <button
            type="submit"
            disabled={disabled || (!value.trim() && files.length === 0)}
            className={cn(
              'flex h-9 items-center gap-2 rounded-lg bg-muted px-4 text-sm font-medium transition-all',
              'hover:bg-primary hover:text-primary-foreground',
              (disabled || (!value.trim() && files.length === 0)) && 'cursor-not-allowed opacity-50'
            )}
          >
            <span>Send</span>
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  )
}
