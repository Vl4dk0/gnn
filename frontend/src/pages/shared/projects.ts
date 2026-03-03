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
    href: "/degree/index.html",
    title: "Degree Predictor",
    description: "Predicting vertex degree"
  },
  {
    key: "min_cycle",
    href: "/min_cycle/index.html",
    title: "Cycle Predictor",
    description: "Smallest cycle containing vertex"
  },
  {
    key: "cage",
    href: "/cage/index.html",
    title: "Cage Generator",
    description: "A* & PPO cage generation"
  }
];
