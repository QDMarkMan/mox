/**
 * Molx Agent Capabilities data and display component.
 * Shows the three core capabilities: Data Organization, SAR Analysis, and Molecular Analysis.
 */

// Molx Agent Capabilities
const capabilities = [
  {
    id: 'data-organization',
    emoji: '📊',
    title: 'Data Organization',
    description: 'Intelligently organize and clean molecular data. Automatically identify data formats, extract key information, and build structured datasets.',
    features: ['Format Detection', 'Data Cleaning', 'Structured Output']
  },
  {
    id: 'sar-analysis',
    emoji: '🔬',
    title: 'SAR Analysis',
    description: 'Deep structure-activity relationship analysis. Identify key pharmacophores, generate visual reports, and support multi-dimensional activity prediction.',
    features: ['Activity Prediction', 'Pharmacophore ID', 'Visual Reports']
  },
  {
    id: 'molecular-analysis',
    emoji: '🧬',
    title: 'Molecular Analysis',
    description: 'Comprehensive molecular property analysis including physicochemical calculations, ADMET prediction, and molecular fingerprint comparison.',
    features: ['Physicochemical', 'ADMET Prediction', 'Fingerprints']
  }
]

interface CapabilityCardProps {
  emoji: string
  title: string
  description: string
  features: string[]
}

function CapabilityCard({ emoji, title, description, features }: CapabilityCardProps) {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-border/50 bg-card p-4 transition-all duration-300 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5">
      {/* Header */}
      <div className="mb-3 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-xl">
          {emoji}
        </div>
        <h3 className="font-medium text-foreground">{title}</h3>
      </div>

      {/* Content */}
      <p className="mb-4 text-xs leading-relaxed text-muted-foreground">{description}</p>

      {/* Features */}
      <div className="flex flex-wrap gap-1.5">
        {features.map((feature, i) => (
          <span key={i} className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
            {feature}
          </span>
        ))}
      </div>
    </div>
  )
}

/**
 * Capabilities section component displaying all Molx Agent capabilities.
 */
export function CapabilitiesSection() {
  return (
    <div className="w-full space-y-3">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground"><span className="uppercase">Capabilities</span></span>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {capabilities.map((capability) => (
          <CapabilityCard
            key={capability.id}
            emoji={capability.emoji}
            title={capability.title}
            description={capability.description}
            features={capability.features}
          />
        ))}
      </div>
    </div>
  )
}
