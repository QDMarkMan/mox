import { useState } from 'react'
import { ChatInput } from './chat-input'
import { MolecularBackground } from './molecular-background'
import { CapabilitiesSection } from './capabilities-section'

interface WelcomePageProps {
  onInputSubmit: (message: string, files?: any[]) => void
}

export function WelcomePage({ onInputSubmit }: WelcomePageProps) {
  const [input, setInput] = useState('')

  return (
    <div className="relative flex h-full w-full flex-col items-center overflow-auto bg-background/50 p-4">
      {/* Decorative Molecular Background */}
      <MolecularBackground />

      <div className="relative z-10 w-full max-w-4xl space-y-5 pt-8">
        {/* Welcome Header */}
        <div className="space-y-1.5">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            Welcome to MolX Agent
          </h1>
          <p className="text-[15px] text-muted-foreground">
            Your intelligent molecular data analysis assistant for drug discovery
          </p>
        </div>

        {/* Large Input Area */}
        <div className="w-full">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={onInputSubmit}
            variant="welcome"
            placeholder="Describe your task, e.g., Analyze the SAR of these molecules..."
            showQuickActions={false}
            showAgentMode={true}
            showKnowledgeBase={true}
          />
        </div>

        {/* Capabilities Section */}
        <CapabilitiesSection />
      </div>
    </div>
  )
}
