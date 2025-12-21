/**
 * Decorative molecular background SVG elements with subtle animations.
 * Used as a visual enhancement for the welcome page.
 */
export function MolecularBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden opacity-[0.04]">
      {/* Molecule 1 - Top Right */}
      <svg className="absolute -right-10 -top-10 h-64 w-64 text-primary animate-[float_20s_ease-in-out_infinite]" viewBox="0 0 200 200" fill="none">
        <circle cx="100" cy="100" r="12" fill="currentColor" />
        <circle cx="50" cy="60" r="8" fill="currentColor" />
        <circle cx="150" cy="60" r="8" fill="currentColor" />
        <circle cx="60" cy="140" r="10" fill="currentColor" />
        <circle cx="140" cy="140" r="10" fill="currentColor" />
        <line x1="100" y1="100" x2="50" y2="60" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="150" y2="60" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="60" y2="140" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="140" y2="140" stroke="currentColor" strokeWidth="3" />
      </svg>

      {/* Molecule 2 - Bottom Left */}
      <svg className="absolute -bottom-20 -left-20 h-80 w-80 text-primary animate-[float_25s_ease-in-out_infinite_reverse]" viewBox="0 0 200 200" fill="none">
        <circle cx="100" cy="100" r="15" fill="currentColor" />
        <circle cx="40" cy="100" r="10" fill="currentColor" />
        <circle cx="160" cy="100" r="10" fill="currentColor" />
        <circle cx="100" cy="40" r="8" fill="currentColor" />
        <circle cx="100" cy="160" r="8" fill="currentColor" />
        <line x1="100" y1="100" x2="40" y2="100" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="160" y2="100" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="100" y2="40" stroke="currentColor" strokeWidth="3" />
        <line x1="100" y1="100" x2="100" y2="160" stroke="currentColor" strokeWidth="3" />
      </svg>

      {/* Hexagon Ring - Center Right */}
      <svg className="absolute right-1/4 top-1/3 h-48 w-48 text-primary animate-[spin_60s_linear_infinite]" viewBox="0 0 200 200" fill="none">
        <polygon points="100,20 170,60 170,140 100,180 30,140 30,60" stroke="currentColor" strokeWidth="2" fill="none" />
        <circle cx="100" cy="20" r="6" fill="currentColor" />
        <circle cx="170" cy="60" r="6" fill="currentColor" />
        <circle cx="170" cy="140" r="6" fill="currentColor" />
        <circle cx="100" cy="180" r="6" fill="currentColor" />
        <circle cx="30" cy="140" r="6" fill="currentColor" />
        <circle cx="30" cy="60" r="6" fill="currentColor" />
      </svg>

      {/* Small molecules scattered */}
      <svg className="absolute left-1/4 top-1/4 h-32 w-32 text-muted-foreground animate-[pulse_8s_ease-in-out_infinite]" viewBox="0 0 100 100" fill="none">
        <circle cx="50" cy="50" r="8" fill="currentColor" />
        <circle cx="20" cy="30" r="5" fill="currentColor" />
        <circle cx="80" cy="70" r="5" fill="currentColor" />
        <line x1="50" y1="50" x2="20" y2="30" stroke="currentColor" strokeWidth="2" />
        <line x1="50" y1="50" x2="80" y2="70" stroke="currentColor" strokeWidth="2" />
      </svg>

      <svg className="absolute bottom-1/4 right-1/3 h-24 w-24 text-muted-foreground animate-[pulse_10s_ease-in-out_infinite_2s]" viewBox="0 0 100 100" fill="none">
        <circle cx="50" cy="50" r="6" fill="currentColor" />
        <circle cx="30" cy="70" r="4" fill="currentColor" />
        <circle cx="70" cy="30" r="4" fill="currentColor" />
        <line x1="50" y1="50" x2="30" y2="70" stroke="currentColor" strokeWidth="2" />
        <line x1="50" y1="50" x2="70" y2="30" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  )
}
