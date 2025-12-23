import { useState, useCallback, useRef, useEffect } from 'react' 
import { apiUrl } from '@/lib/api'

export interface ThinkingInfo {
  status: 'analyzing' | 'complete'
  intent?: string
  reasoning?: string
  confidence?: number
  message?: string
}

export interface SessionArtifact {
  fileId: string
  fileName: string
  description?: string
  contentType?: string
  sizeBytes?: number
  createdAt?: string
  downloadUrl?: string
  inlineUrl?: string
}

export interface ReportMetadata {
  report_path?: string
  summary?: string
  preview?: string
  created_at?: string
}

export interface StreamingMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  thinking?: ThinkingInfo
  status?: string[]
  artifacts?: SessionArtifact[]
  report?: ReportMetadata
  structuredData?: Record<string, any>
}

interface UseStreamingChatOptions {
  api?: string
  sessionId?: string | null
  onFinish?: (message: StreamingMessage) => void
  onError?: (error: Error) => void
}

interface UseStreamingChatReturn {
  messages: StreamingMessage[]
  input: string
  setInput: (input: string) => void
  isLoading: boolean
  thinking: ThinkingInfo | null
  error: Error | null
  sendMessage: (content: string) => Promise<void>
  clearMessages: () => void
}

/**
 * Custom hook for streaming chat with MolX Server SSE endpoint.
 * 
 * Handles MolX Server SSE format including thinking events.
 */
export function useStreamingChat({
  api = '/api/v1/agent/stream',
  sessionId,
  onFinish,
  onError,
}: UseStreamingChatOptions = {}): UseStreamingChatReturn {
  const [messages, setMessages] = useState<StreamingMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [thinking, setThinking] = useState<ThinkingInfo | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const lastSessionIdRef = useRef<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const isSessionSwitch = lastSessionIdRef.current !== null && 
                            sessionId !== null && 
                            sessionId !== undefined &&
                            lastSessionIdRef.current !== sessionId
    const isGoingToNull = sessionId === null

    if (isSessionSwitch || isGoingToNull) {
      setMessages([])
      setThinking(null)
      setError(null)
      setIsLoading(false)
    }
    
    lastSessionIdRef.current = sessionId ?? null

    if (!sessionId) {
      return
    }

    const controller = new AbortController()
    const loadHistory = async () => {
      try {
        const [historyResponse, metadataResponse] = await Promise.all([
          fetch(apiUrl(`/api/v1/session/${sessionId}/history`), {
            signal: controller.signal,
          }),
          fetch(apiUrl(`/api/v1/session/${sessionId}/data`), {
            signal: controller.signal,
          }),
        ])

        let hydrated: StreamingMessage[] = []
        if (historyResponse.ok) {
          const history = await historyResponse.json()
          hydrated = (history?.messages ?? [])
            .filter((msg: any) => msg.role === 'user' || msg.role === 'agent')
            .map((msg: any, index: number) => ({
              id: `${msg.role}-${index}`,
              role: msg.role === 'agent' ? 'assistant' : 'user',
              content: msg.content ?? '',
            }))
        }

        if (metadataResponse.ok) {
          const metadata = await metadataResponse.json()
          const turns = Array.isArray(metadata?.turns) ? metadata.turns : []
          if (turns.length > 0) {
            const turnMessages: StreamingMessage[] = []
            turns.forEach((turn: any, index: number) => {
              if (turn?.query) {
                turnMessages.push({
                  id: `user-turn-${index}`,
                  role: 'user',
                  content: turn.query,
                })
              }
              turnMessages.push({
                id: `assistant-turn-${index}`,
                role: 'assistant',
                content: turn?.response ?? '',
                artifacts: normalizeArtifacts(turn?.artifacts, sessionId),
                report: turn?.report ?? undefined,
                structuredData: turn?.structured_data ?? undefined,
              })
            })
            hydrated = turnMessages
          }
        }

        setMessages(prev => {
          // If we already have messages (e.g. from a just-started sendMessage),
          // merge history with them.
          if (prev.length > 0) {
            // Filter out any messages from prev that are already in hydrated (by content/role if ID is different)
            // But usually, we just want to append the new ones.
            const newMessages = prev.filter(m => !hydrated.some(h => h.role === m.role && h.content === m.content))
            return [...hydrated, ...newMessages]
          }
          return hydrated
        })
      } catch (historyError) {
        if ((historyError as Error).name !== 'AbortError') {
          console.warn('Failed to load session history', historyError)
        }
      }
    }

    loadHistory()
    return () => controller.abort()
  }, [sessionId])

  const appendStatus = useCallback((existing: string[] | undefined, value: string) => {
    const bucket = [...(existing ?? []), value]
    if (bucket.length > 40) {
      bucket.shift()
    }
    return bucket
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    // Add user message
    const userMessage: StreamingMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    setThinking(null)
    setError(null)

    // Create assistant message placeholder
    const assistantId = `assistant-${Date.now()}`
    const assistantMessage: StreamingMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      abortControllerRef.current = new AbortController()

      const endpoint = apiUrl(api)
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          query: content.trim(),
          session_id: sessionId,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let accumulatedContent = ''
      let buffer = ''
      let capturedThinking: ThinkingInfo | null = null
      let capturedArtifacts: SessionArtifact[] | undefined
      let capturedReport: ReportMetadata | undefined
      let capturedStructured: Record<string, any> | undefined

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        
        // Parse SSE events from buffer
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''

        for (const eventBlock of events) {
          if (!eventBlock.trim()) continue

          const lines = eventBlock.split('\n')
          let eventType = ''
          let eventData = ''

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              eventData = line.slice(5).trim()
            }
          }

          if (!eventType || !eventData) continue

          try {
            const data = JSON.parse(eventData)

            switch (eventType) {
              case 'start':
                console.log('[SSE] Start:', data.query)
                break
              
              case 'thinking':
                // Update thinking state
                const thinkingInfo: ThinkingInfo = {
                  status: data.status || 'analyzing',
                  intent: data.intent,
                  reasoning: data.reasoning,
                  confidence: data.confidence,
                  message: data.message,
                }
                setThinking(thinkingInfo)
                capturedThinking = thinkingInfo
                
                // Store thinking in message
                if (data.status === 'complete') {
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === assistantId
                        ? { ...msg, thinking: thinkingInfo }
                        : msg
                    )
                  )
                }
                break
              
              case 'token':
                if (data.content) {
                  accumulatedContent += data.content
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === assistantId
                        ? { ...msg, content: accumulatedContent, thinking: capturedThinking || undefined }
                        : msg
                    )
                  )
                }
                break

              case 'complete':
                const finalContent = data.result || accumulatedContent
                setMessages(prev =>
                  prev.map(msg =>
                    msg.id === assistantId
                      ? {
                          ...msg,
                          content: finalContent,
                          thinking: capturedThinking || undefined,
                          artifacts: capturedArtifacts,
                          report: capturedReport,
                          structuredData: capturedStructured,
                        }
                      : msg
                  )
                )
                accumulatedContent = finalContent
                setThinking(null)
                break

              case 'artifacts':
                if (data.artifacts || data.report || data.structured_data) {
                  const artifacts = normalizeArtifacts(data.artifacts, sessionId)
                  capturedArtifacts = artifacts.length > 0 ? artifacts : undefined
                  capturedReport = data.report ?? undefined
                  capturedStructured = data.structured_data ?? undefined
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === assistantId
                        ? {
                            ...msg,
                            artifacts: capturedArtifacts,
                            report: capturedReport,
                            structuredData: capturedStructured,
                          }
                        : msg
                    )
                  )
                }
                break
              
              case 'status':
                if (data.message) {
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === assistantId
                        ? { ...msg, status: appendStatus(msg.status, data.message) }
                        : msg
                    )
                  )
                }
                break
              
              case 'error':
                throw new Error(data.message || 'Stream error')
              
              case 'tool_start':
                console.log('[SSE] Tool start:', data.content)
                break
              
              case 'tool_end':
                console.log('[SSE] Tool end:', data.content)
                break
            }
          } catch (parseError) {
            console.warn('[SSE] Parse error:', parseError, eventData)
          }
        }
      }

      // Final message
      const finalMessage: StreamingMessage = {
        id: assistantId,
        role: 'assistant',
        content: accumulatedContent || 'No response received.',
        thinking: capturedThinking || undefined,
        artifacts: capturedArtifacts,
        report: capturedReport,
        structuredData: capturedStructured,
      }

      onFinish?.(finalMessage)
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))

      if (error.name !== 'AbortError') {
        setError(error)
        onError?.(error)
        setThinking(null)

        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: `Error: ${error.message}` }
              : msg
          )
        )
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }, [api, sessionId, isLoading, onFinish, onError, appendStatus])

  const clearMessages = useCallback(() => {
    setMessages([])
    setThinking(null)
    setError(null)
  }, [])

  return {
    messages,
    input,
    setInput,
    isLoading,
    thinking,
    error,
    sendMessage,
    clearMessages,
  }
}

function normalizeArtifacts(rawArtifacts: any, sessionId?: string | null): SessionArtifact[] {
  if (!Array.isArray(rawArtifacts)) {
    return []
  }

  const artifacts: SessionArtifact[] = []
  
  for (const artifact of rawArtifacts) {
    if (!artifact?.file_id || !artifact?.file_name) {
      continue
    }
    const basePath = sessionId
      ? `/api/v1/session/${sessionId}/files/${artifact.file_id}`
      : undefined
    
    artifacts.push({
      fileId: String(artifact.file_id),
      fileName: String(artifact.file_name),
      description: artifact.description ? String(artifact.description) : undefined,
      contentType: artifact.content_type ? String(artifact.content_type) : undefined,
      sizeBytes: typeof artifact.size_bytes === 'number' ? artifact.size_bytes : undefined,
      createdAt: artifact.created_at ? String(artifact.created_at) : undefined,
      downloadUrl: basePath ? apiUrl(basePath) : undefined,
      inlineUrl: basePath ? apiUrl(`${basePath}?inline=1`) : undefined,
    })
  }

  return artifacts
}
