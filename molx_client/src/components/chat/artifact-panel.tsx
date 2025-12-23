import { useState, useEffect } from 'react'
import * as Lucide from 'lucide-react'
const { Maximize, Download, Eye, EyeOff, FileText, Table, FileJson, X } = Lucide as any
import type { SessionArtifact, ReportMetadata } from '@/hooks/use-streaming-chat'
import { cn } from '@/utils'
import { Button } from '@/components/ui/button'

interface DataPreviewProps {
  url: string
  type: 'json' | 'csv'
  className?: string
  fullScreen?: boolean
}

function DataPreview({ url, type, className, fullScreen }: DataPreviewProps) {
  const [data, setData] = useState<string | string[][] | any | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(url)
        if (!response.ok) throw new Error('Failed to fetch data')
        const text = await response.text()

        if (type === 'json') {
          try {
            setData(JSON.parse(text))
          } catch {
            setData(text)
          }
        } else if (type === 'csv') {
          const rows = text.split('\n').filter(row => row.trim()).map(row => row.split(','))
          setData(rows)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [url, type])

  if (loading) return <div className="flex items-center justify-center p-8 text-muted-foreground animate-pulse">Loading preview...</div>
  if (error) return <div className="p-4 text-destructive text-sm">Error: {error}</div>
  if (!data) return null

  if (type === 'json') {
    // Syntax highlighting for JSON
    const highlightJson = (json: any): React.ReactNode => {
      const jsonString = typeof json === 'string' ? json : JSON.stringify(json, null, 2)

      // Split into lines and highlight each token
      const highlightLine = (line: string): React.ReactNode => {
        // Pattern to match JSON tokens
        const parts: React.ReactNode[] = []
        let remaining = line
        let keyIndex = 0

        // Match key-value patterns
        const keyMatch = remaining.match(/^(\s*)("(?:[^"\\]|\\.)*")(\s*:\s*)/)
        if (keyMatch) {
          parts.push(<span key={`space-${keyIndex}`}>{keyMatch[1]}</span>)
          parts.push(<span key={`key-${keyIndex}`} className="text-purple-600 dark:text-purple-400">{keyMatch[2]}</span>)
          parts.push(<span key={`colon-${keyIndex}`}>{keyMatch[3]}</span>)
          remaining = remaining.slice(keyMatch[0].length)
          keyIndex++
        }

        // Highlight the value part
        const valuePatterns = [
          { pattern: /^"(?:[^"\\]|\\.)*"/, className: "text-green-600 dark:text-green-400" }, // strings
          { pattern: /^-?\d+\.?\d*(?:[eE][+-]?\d+)?/, className: "text-blue-600 dark:text-blue-400" }, // numbers
          { pattern: /^(true|false)/, className: "text-orange-600 dark:text-orange-400" }, // booleans
          { pattern: /^null/, className: "text-red-500 dark:text-red-400" }, // null
        ]

        for (const { pattern, className } of valuePatterns) {
          const match = remaining.match(pattern)
          if (match) {
            parts.push(<span key={`val-${keyIndex}`} className={className}>{match[0]}</span>)
            remaining = remaining.slice(match[0].length)
            break
          }
        }

        // Add any remaining text (brackets, commas, etc.)
        if (remaining) {
          parts.push(<span key={`rest-${keyIndex}`} className="text-foreground/80">{remaining}</span>)
        }

        return parts.length > 0 ? parts : line
      }

      return jsonString.split('\n').map((line, i) => (
        <div key={i}>{highlightLine(line)}</div>
      ))
    }

    return (
      <pre className={cn(
        "p-4 text-xs font-mono overflow-auto bg-slate-50 dark:bg-slate-900/50 rounded-md",
        fullScreen ? "h-full" : "max-h-[32rem]",
        className
      )}>
        {highlightJson(data)}
      </pre>
    )
  }

  if (type === 'csv' && Array.isArray(data)) {
    return (
      <div className={cn(
        "overflow-auto bg-background rounded-md border border-border",
        fullScreen ? "h-full" : "max-h-[32rem]",
        className
      )}>
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-gradient-to-r from-primary/10 to-primary/5 sticky top-0">
              {data[0]?.map((header: string, i: number) => (
                <th key={i} className="p-2 border-b border-border font-bold text-primary whitespace-nowrap">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(1).map((row: string[], i: number) => (
              <tr
                key={i}
                className={cn(
                  "hover:bg-primary/5 transition-colors",
                  i % 2 === 0 ? "bg-background" : "bg-muted/20"
                )}
              >
                {row.map((cell: string, j: number) => {
                  // Highlight numeric values
                  const isNumber = !isNaN(Number(cell)) && cell.trim() !== ''
                  return (
                    <td
                      key={j}
                      className={cn(
                        "p-2 border-b border-border/50 whitespace-nowrap",
                        isNumber && "text-blue-600 dark:text-blue-400 font-medium tabular-nums"
                      )}
                    >
                      {cell}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return null
}

interface ArtifactPanelProps {
  artifacts?: SessionArtifact[]
  report?: ReportMetadata
}

export function ArtifactPanel({ artifacts, report }: ArtifactPanelProps) {
  if ((!artifacts || artifacts.length === 0) && !report) {
    return null
  }

  const summary = report?.preview || report?.summary

  return (
    <div className="mt-3 space-y-3">
      {summary && (
        <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm text-primary">
          <div className="text-xs font-semibold uppercase tracking-wide text-primary/80">Report Summary</div>
          <p className="mt-1 text-sm text-primary/90">{summary}</p>
        </div>
      )}

      {artifacts && artifacts.length > 0 && (
        <div className="space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Generated Reports & Artifacts</div>
          {artifacts.map(artifact => (
            <ArtifactCard key={artifact.fileId} artifact={artifact} />
          ))}
        </div>
      )}
    </div>
  )
}

function ArtifactCard({ artifact }: { artifact: SessionArtifact }) {
  const [expanded, setExpanded] = useState(false)
  const [fullScreen, setFullScreen] = useState(false)
  const previewable = isPreviewable(artifact)

  return (
    <>
      <div className="rounded-md border border-border bg-background/60 p-3 shadow-sm transition-all hover:shadow-md">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/50 text-muted-foreground">
              {getIcon(artifact)}
            </div>
            <div>
              <div className="text-sm font-medium text-foreground">{artifact.fileName}</div>
              <div className="text-xs text-muted-foreground">
                {artifact.description || 'Generated artifact'}
                {artifact.sizeBytes ? ` • ${formatFileSize(artifact.sizeBytes)}` : ''}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {artifact.downloadUrl && (
              <Button variant="ghost" size="sm" asChild className="h-8 px-2">
                <a href={artifact.downloadUrl} target="_blank" rel="noreferrer">
                  <Download className="mr-1 h-3.5 w-3.5" />
                  Download
                </a>
              </Button>
            )}
            {previewable && artifact.inlineUrl && (
              <Button
                variant={expanded ? "secondary" : "outline"}
                size="sm"
                onClick={() => setExpanded(prev => !prev)}
                className="h-8 px-2"
              >
                {expanded ? <EyeOff className="mr-1 h-3.5 w-3.5" /> : <Eye className="mr-1 h-3.5 w-3.5" />}
                {expanded ? 'Hide' : 'Preview'}
              </Button>
            )}
          </div>
        </div>

        {expanded && artifact.inlineUrl && (
          <div className="mt-3 relative group">
            <div className="absolute right-2 top-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button
                variant="secondary"
                size="icon"
                className="h-8 w-8 bg-background/80 backdrop-blur-sm shadow-sm"
                onClick={() => setFullScreen(true)}
              >
                <Maximize className="h-4 w-4" />
              </Button>
            </div>
            <div className="overflow-hidden rounded-md border border-border bg-muted/20">
              {renderPreview(artifact)}
            </div>
          </div>
        )}
      </div>

      {/* Custom Full Screen Modal */}
      {fullScreen && (
        <div className="fixed inset-0 z-[100] flex flex-col bg-background animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-border p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/50 text-muted-foreground">
                {getIcon(artifact)}
              </div>
              <h2 className="text-lg font-semibold">{artifact.fileName}</h2>
            </div>
            <div className="flex items-center gap-2">
              {artifact.downloadUrl && (
                <Button variant="outline" size="sm" asChild>
                  <a href={artifact.downloadUrl} target="_blank" rel="noreferrer">
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </a>
                </Button>
              )}
              <Button variant="ghost" size="icon" onClick={() => setFullScreen(false)}>
                <X className="h-5 w-5" />
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-6 bg-muted/5">
            {renderPreview(artifact, true)}
          </div>
        </div>
      )}
    </>
  )
}

function getIcon(artifact: SessionArtifact) {
  const type = artifact.contentType || ''
  if (type.startsWith('image/')) return <Eye className="h-4 w-4" />
  if (type === 'text/csv') return <Table className="h-4 w-4" />
  if (type === 'application/json') return <FileJson className="h-4 w-4" />
  return <FileText className="h-4 w-4" />
}

function isPreviewable(artifact: SessionArtifact): boolean {
  if (!artifact.contentType) return false
  if (!artifact.inlineUrl) return false
  const type = artifact.contentType
  return (
    type.startsWith('image/') ||
    type === 'text/html' ||
    type === 'application/pdf' ||
    type.startsWith('text/') ||
    type === 'application/json' ||
    type === 'text/csv'
  )
}

function renderPreview(artifact: SessionArtifact, fullScreen = false) {
  if (!artifact.inlineUrl) return null

  const type = artifact.contentType || ''

  if (type.startsWith('image/')) {
    return (
      <div className={cn("flex items-center justify-center bg-background", fullScreen ? "h-full" : "max-h-[32rem]")}>
        <img src={artifact.inlineUrl} alt={artifact.fileName} className="max-h-full max-w-full object-contain" />
      </div>
    )
  }

  if (type === 'application/json') {
    return <DataPreview url={artifact.inlineUrl} type="json" fullScreen={fullScreen} />
  }

  if (type === 'text/csv') {
    return <DataPreview url={artifact.inlineUrl} type="csv" fullScreen={fullScreen} />
  }

  return (
    <iframe
      src={artifact.inlineUrl}
      title={artifact.fileName}
      className={cn("w-full border-none", fullScreen ? "h-full" : "h-[32rem]")}
      sandbox="allow-scripts allow-downloads allow-same-origin"
    />
  )
}

function formatFileSize(size: number): string {
  if (size < 1024) return `${size} B`
  const units = ['KB', 'MB', 'GB']
  let value = size / 1024
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`
}
