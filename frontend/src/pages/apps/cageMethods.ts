import type { CageMethodId, GeneratorType } from "../../types/api";

export type CageMethodGroup = "classical" | "rl" | "voltage";

export interface CageMethod {
  id: CageMethodId;
  label: string;
  group: CageMethodGroup;
  generator: GeneratorType;
  /** For voltage_girth and voltage_tabu, the kind of voltage model to resolve. */
  voltageKind?: "girth" | "tabu_cost";
}

export const CAGE_METHODS: CageMethod[] = [
  {
    id: "astar",
    label: "A* search (best-first)",
    group: "classical",
    generator: "astar"
  },
  {
    id: "randomwalk",
    label: "Random walk",
    group: "classical",
    generator: "randomwalk"
  },
  {
    id: "bruteforce",
    label: "Bruteforce (backtracking)",
    group: "classical",
    generator: "bruteforce"
  },
  {
    id: "rl",
    label: "RL agent (GNN-guided)",
    group: "rl",
    generator: "rl"
  },
  {
    id: "voltage_algebraic",
    label: "Voltage algebraic (tabu search)",
    group: "voltage",
    generator: "voltage"
  },
  {
    id: "voltage_tabu",
    label: "Voltage tabu-predictor",
    group: "voltage",
    generator: "voltage",
    voltageKind: "tabu_cost"
  },
  {
    id: "voltage_girth",
    label: "Voltage girth-predictor",
    group: "voltage",
    generator: "voltage",
    voltageKind: "girth"
  },
  {
    id: "voltage_rl",
    label: "Voltage RL (learned assignments)",
    group: "voltage",
    generator: "voltage_rl"
  },
  {
    id: "forge",
    label: "Forge (lift, refine, excision)",
    group: "voltage",
    generator: "forge"
  }
];

export const GROUP_LABELS: Record<CageMethodGroup, string> = {
  classical: "Classical search",
  rl: "Reinforcement learning",
  voltage: "Voltage methods"
};

export const GROUP_DEFAULTS: Record<CageMethodGroup, CageMethodId> = {
  classical: "astar",
  rl: "rl",
  voltage: "voltage_algebraic"
};

export const FROM_PARAM_TO_GROUP: Record<string, CageMethodGroup> = {
  astar: "classical",
  rl: "rl",
  voltage: "voltage"
};

export function getMethod(id: CageMethodId): CageMethod {
  const found = CAGE_METHODS.find((m) => m.id === id);
  if (!found) {
    throw new Error(`Unknown cage method: ${id}`);
  }
  return found;
}

export function methodsForGroup(group: CageMethodGroup): CageMethod[] {
  return CAGE_METHODS.filter((m) => m.group === group);
}
