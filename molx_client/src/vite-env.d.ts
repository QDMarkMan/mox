/// <reference types="vite/client" />

declare module 'lucide-react' {
  import { ComponentType, SVGProps } from 'react'
  export const Beaker: ComponentType<SVGProps<SVGSVGElement>>
  export const Plus: ComponentType<SVGProps<SVGSVGElement>>
  export const MessageSquare: ComponentType<SVGProps<SVGSVGElement>>
  export const Trash2: ComponentType<SVGProps<SVGSVGElement>>
  export const Send: ComponentType<SVGProps<SVGSVGElement>>
  export const Bot: ComponentType<SVGProps<SVGSVGElement>>
  export const User: ComponentType<SVGProps<SVGSVGElement>>
  export const Brain: ComponentType<SVGProps<SVGSVGElement>>
  export const Lightbulb: ComponentType<SVGProps<SVGSVGElement>>
}

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
