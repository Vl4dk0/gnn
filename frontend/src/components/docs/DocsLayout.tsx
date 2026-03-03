import type { PropsWithChildren } from "react";

import type { FeatureKey } from "../../types/app";

interface DocsLayoutProps {
  featureActive: Record<FeatureKey, boolean>;
  currentPath: string;
}

export const DocsLayout = ({
  featureActive: _featureActive,
  currentPath,
  children
}: PropsWithChildren<DocsLayoutProps>) => {
  const showOverviewButton = currentPath !== "/";

  return (
    <div className="h-dvh overflow-y-auto bg-bg1">
      <main className="page mx-auto w-full max-w-[1100px] p-10 pb-[60px] max-[760px]:w-full max-[760px]:p-3 max-[760px]:pt-3">
        {showOverviewButton ? (
          <div className="mb-4 flex justify-end">
            <a
              href="/"
              className="inline-block rounded-md bg-accent px-5 py-2.5 text-xs font-bold uppercase tracking-[0.8px] text-textMain no-underline shadow-card transition-all duration-300 hover:bg-accentHover hover:shadow-cardHover"
            >
              Overview
            </a>
          </div>
        ) : null}
        {children}
      </main>
    </div>
  );
};
