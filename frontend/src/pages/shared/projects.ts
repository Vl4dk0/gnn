import type { FeatureKey } from "../../types/app";

export interface ProjectLink {
  key: FeatureKey;
  href: string;
  title: string;
  description: string;
}

export const PROJECT_LINKS: ProjectLink[] = [
  {
    key: "degree",
    href: "/degree",
    title: "Degree Predictor",
    description: "Interactive node-degree prediction"
  },
  {
    key: "min_cycle",
    href: "/min_cycle",
    title: "Cycle Predictor",
    description: "Interactive shortest-cycle prediction"
  },
  {
    key: "cage",
    href: "/cage",
    title: "Cage Generator",
    description: "Experimental search and PPO generation"
  }
];
