export interface DocsItem {
  href: string;
  title: string;
  short: string;
}

export const DOCS_SEQUENCE: DocsItem[] = [
  { href: "/degree-task", title: "Degree subproblem", short: "Degree" },
  { href: "/min-cycle-task", title: "Min-cycle subproblem", short: "Cycle" },
  { href: "/cage/astar", title: "A* + backtracking", short: "A*" },
  { href: "/cage/rl", title: "Reinforcement learning", short: "RL" },
  { href: "/cage/voltage", title: "Voltage lifts", short: "Voltage" },
  { href: "/excision", title: "Excision", short: "Excision" }
];

export const getNextDocsItem = (currentHref: string): DocsItem | null => {
  const currentIndex = DOCS_SEQUENCE.findIndex((item) => item.href === currentHref);
  if (currentIndex === -1 || currentIndex === DOCS_SEQUENCE.length - 1) {
    return null;
  }
  return DOCS_SEQUENCE[currentIndex + 1];
};
