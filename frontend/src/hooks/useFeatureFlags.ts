import { useEffect, useMemo, useState } from "react";

import { fetchFeatureConfig } from "../services/features";
import type { FeatureKey } from "../types/app";

const ALL_ACTIVE: Record<FeatureKey, boolean> = {
  degree: true,
  min_cycle: true,
  cage: true
};

export const useFeatureFlags = () => {
  const [flags, setFlags] = useState<Record<FeatureKey, boolean>>(ALL_ACTIVE);

  useEffect(() => {
    let isActive = true;

    fetchFeatureConfig()
      .then((config) => {
        if (!isActive) return;
        setFlags({
          degree: Boolean(config.features.degree?.active),
          min_cycle: Boolean(config.features.min_cycle?.active),
          cage: Boolean(config.features.cage?.active)
        });
      })
      .catch(() => {
        if (!isActive) return;
        setFlags(ALL_ACTIVE);
      });

    return () => {
      isActive = false;
    };
  }, []);

  return useMemo(() => flags, [flags]);
};
