import { useState } from 'react'
import type { SessionArtifact, ReportMetadata } from '@/hooks/use-streaming-chat'
import { cn } from '@/utils'

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
  const previewable = isPreviewable(artifact)

  return (
    <div className="rounded-md border border-border bg-background/60 p-3 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm font-medium text-foreground">{artifact.fileName}</div>
          <div className="text-xs text-muted-foreground">
            {artifact.description || 'Generated artifact'}
            {artifact.sizeBytes ? ` • ${formatFileSize(artifact.sizeBytes)}` : ''}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {artifact.downloadUrl && (
            <a
              href={artifact.downloadUrl}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-border px-2 py-1 text-foreground transition hover:bg-muted"
            >
              Download
            </a>
          )}
          {previewable && artifact.inlineUrl && (
            <button
              onClick={() => setExpanded(prev => !prev)}
              className={cn(
                'rounded-md border px-2 py-1 transition',
                expanded ? 'border-primary bg-primary/10 text-primary' : 'border-border text-foreground hover:bg-muted'
              )}
            >
              {expanded ? 'Hide Preview' : 'Preview'}
            </button>
          )}
        </div>
      </div>

      {expanded && artifact.inlineUrl && (
        <div className="mt-3 overflow-hidden rounded-md border border-border bg-muted/50">
          {renderPreview(artifact)}
        </div>
      )}
    </div>
  )
}

function isPreviewable(artifact: SessionArtifact): boolean {
  if (!artifact.contentType) return false
  if (!artifact.inlineUrl) return false
  return (
    artifact.contentType.startsWith('image/') ||
    artifact.contentType === 'text/html' ||
    artifact.contentType === 'application/pdf' ||
    artifact.contentType.startsWith('text/')
  )
}

function renderPreview(artifact: SessionArtifact) {
  if (!artifact.inlineUrl) return null

  if (artifact.contentType?.startsWith('image/')) {
    return <img src={artifact.inlineUrl} alt={artifact.fileName} className="max-h-[32rem] w-full object-contain bg-background" />
  }

  if (artifact.contentType === 'application/pdf') {
    return (
      <iframe
        src={artifact.inlineUrl}
        title={artifact.fileName}
        className="h-[32rem] w-full"
        sandbox="allow-scripts allow-downloads allow-same-origin"
      />
    )
  }

  return (
    <iframe
      src={artifact.inlineUrl}
      title={artifact.fileName}
      className="h-[32rem] w-full"
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
