export interface SiteNode {
  id: number;
  title: string;
  href: string;
  x: number;
  y: number;
}

export interface SiteEdge {
  from: number;
  to: number;
}

export const SITE_NODES: SiteNode[] = [
  // Root (Overview)
  { id: 0, title: "Overview", href: "/", x: 50, y: 40 },
  
  // Documentation Pages (Vertical Stack)
  { id: 1, title: "How GNNs work", href: "/docs/gnns", x: 50, y: 100 },
  { id: 2, title: "Architectures", href: "/docs/architecture", x: 50, y: 160 },
  { id: 3, title: "Degree prediction", href: "/docs/module-degree", x: 50, y: 220 },
  { id: 4, title: "Cycle prediction", href: "/docs/module-min-cycle", x: 50, y: 280 },
  { id: 5, title: "Assessment", href: "/docs/module-assessment", x: 50, y: 340 },
  { id: 6, title: "Cage generation", href: "/docs/module-cage", x: 50, y: 400 },
  { id: 7, title: "Try it yourself", href: "/docs/training", x: 50, y: 460 },

  // Interactive Editors (Side branches - Shifted down to avoid text overlap)
  { id: 8, title: "Degree Predictor", href: "/degree", x: 120, y: 240 },
  { id: 9, title: "Cycle Predictor", href: "/min_cycle", x: 120, y: 300 },
  { id: 10, title: "Cage Generator", href: "/cage", x: 120, y: 420 }
];

export const SITE_EDGES: SiteEdge[] = [
  // Row 1 -> Row 2
  { from: 0, to: 1 },
  { from: 0, to: 2 },
  { from: 0, to: 3 },
  { from: 0, to: 4 },
  { from: 0, to: 5 },
  { from: 0, to: 6 },
  { from: 0, to: 7 },

  // Sequential flow within Row 2
  { from: 1, to: 2 },
  { from: 2, to: 3 },
  { from: 3, to: 4 },
  { from: 4, to: 5 },
  { from: 5, to: 6 },
  { from: 6, to: 7 },

  // Row 2 -> Row 3 (Interactive Editors)
  { from: 3, to: 8 },  // Degree prediction -> Degree Predictor
  { from: 4, to: 9 },  // Cycle prediction -> Cycle Predictor
  { from: 6, to: 10 }  // Cage generation -> Cage Generator
];
