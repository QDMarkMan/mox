import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MessageContentProps {
  content: string
  isUser: boolean
}

export function MessageContent({ content, isUser }: MessageContentProps) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words text-[14px] leading-relaxed">
      {isUser ? (
        <p className="whitespace-pre-wrap m-0">{content}</p>
      ) : (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      )}
    </div>
  )
}
