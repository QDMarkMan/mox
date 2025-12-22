import { useState, useCallback, useRef, useEffect } from 'react'
import { apiUrl } from '@/lib/api'

export interface ThinkingInfo {
  status: 'analyzing' | 'complete'
  intent?: string
  reasoning?: string
  confidence?: number
  message?: string
}

export interface StreamingMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  thinking?: ThinkingInfo
  status?: string[]
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
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!sessionId) {
      setMessages([])
      return
    }

    const controller = new AbortController()
    const loadHistory = async () => {
      try {
        const response = await fetch(apiUrl(`/api/v1/session/${sessionId}/history`), {
          signal: controller.signal,
        })
        if (!response.ok) {
          return
        }
        const data = await response.json()
        const hydrated: StreamingMessage[] = (data?.messages ?? [])
          .filter((msg: any) => msg.role === 'user' || msg.role === 'agent')
          .map((msg: any, index: number) => ({
            id: `${msg.role}-${index}`,
            role: msg.role === 'agent' ? 'assistant' : 'user',
            content: msg.content ?? '',
          }))
        setMessages(hydrated)
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
                      ? { ...msg, content: finalContent, thinking: capturedThinking || undefined }
                      : msg
                  )
                )
                accumulatedContent = finalContent
                setThinking(null)
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
