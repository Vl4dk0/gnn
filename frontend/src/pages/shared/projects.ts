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
    description: "Predicting vertex degree"
  },
  {
    key: "min_cycle",
    href: "/min_cycle",
    title: "Cycle Predictor",
    description: "Smallest cycle containing vertex"
  },
  {
    key: "cage",
    href: "/cage",
    title: "Cage Generator",
    description: "A* & PPO cage generation"
  }
];
