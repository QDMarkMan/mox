import { Beaker } from 'lucide-react'

export function Header() {
  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-2">
        <Beaker className="h-6 w-6 text-primary" />
        <h1 className="text-lg font-semibold">MolX Agent</h1>
      </div>
      <nav className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">
          Drug Design AI Assistant
        </span>
      </nav>
    </header>
  )
}
