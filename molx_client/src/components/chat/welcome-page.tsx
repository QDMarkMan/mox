import { useState } from 'react'
import { ChatInput } from './chat-input'
import { MolecularBackground } from './molecular-background'
import { CapabilitiesSection } from './capabilities-section'
import MoxLogo from '@/assets/logo.png'

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
        <div className="space-y-4">
          <div className="space-y-1.5">
            <h1 className="inline-flex items-center gap-2 text-2xl font-semibold tracking-tight text-foreground">
              {/* <img src={MoxLogo} alt="MolX Logo" className="h-8 w-8 object-contain" /> */}
              Welcome to{' '}
              <span
                className="bg-clip-text text-transparent bg-[length:200%_100%] animate-gradient"
                style={{ backgroundImage: 'linear-gradient(90deg, #B5FF00, #FDBC85, #B5FF00)' }}
              >
                Molx Agent
              </span>
            </h1>
            <p className="text-[15px] text-muted-foreground">
              Your intelligent assistant agent for drug discovery
            </p>
          </div>
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
          />
        </div>

        {/* Capabilities Section */}
        <CapabilitiesSection />
      </div>
    </div>
  )
}
