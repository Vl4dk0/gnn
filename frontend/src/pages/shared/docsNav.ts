export interface DocsItem {
  href: string;
  title: string;
  short: string;
}

export const DOCS_SEQUENCE: DocsItem[] = [
  { href: "/docs/gnns", title: "How GNNs work", short: "GNNs" },
  { href: "/docs/architecture", title: "Architectures", short: "Arch" },
  { href: "/docs/module-degree", title: "Degree prediction", short: "Degree" },
  { href: "/docs/module-min-cycle", title: "Cycle prediction", short: "Cycle" },
  { href: "/docs/module-assessment", title: "Assessment", short: "Assess" },
  { href: "/docs/module-cage", title: "Cage generation", short: "Cage" },
  { href: "/docs/training", title: "Try it yourself", short: "Try" }
];

export const getNextDocsItem = (currentHref: string): DocsItem | null => {
  const currentIndex = DOCS_SEQUENCE.findIndex((item) => item.href === currentHref);
  if (currentIndex === -1 || currentIndex === DOCS_SEQUENCE.length - 1) {
    return null;
  }
  return DOCS_SEQUENCE[currentIndex + 1];
};
