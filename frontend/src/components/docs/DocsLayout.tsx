import type { PropsWithChildren } from "react";

import { DOCS_SEQUENCE } from "../../pages/shared/docsNav";
import type { FeatureKey } from "../../types/app";
import { SiteGraphNav } from "../layout/SiteGraphNav";

interface DocsLayoutProps {
  featureActive: Record<FeatureKey, boolean>;
  currentPath: string;
}

interface NavItem {
  href: string;
  title: string;
}

type MobileNavToken =
  | { type: "item"; item: NavItem; index: number }
  | { type: "ellipsis"; key: string };

export const DocsLayout = ({
  featureActive: _featureActive,
  currentPath,
  children
}: PropsWithChildren<DocsLayoutProps>) => {
  const navItems: NavItem[] = [{ href: "/", title: "Overview" }, ...DOCS_SEQUENCE];
  const currentIndex = Math.max(
    0,
    navItems.findIndex((item) => item.href === currentPath)
  );

  const mobileTokens: MobileNavToken[] = (() => {
    const lastIndex = navItems.length - 1;

    if (navItems.length <= 5) {
      return navItems.map((item, index) => ({ type: "item", item, index }));
    }

    if (currentIndex <= 2) {
      return [
        ...navItems.slice(0, 4).map((item, index) => ({ type: "item" as const, item, index })),
        { type: "ellipsis" as const, key: "tail" }
      ];
    }

    if (currentIndex >= lastIndex - 2) {
      return [
        { type: "ellipsis" as const, key: "head" },
        ...navItems.slice(lastIndex - 3).map((item, offset) => ({
          type: "item" as const,
          item,
          index: lastIndex - 3 + offset
        }))
      ];
    }

    return [
      { type: "item" as const, item: navItems[0], index: 0 },
      { type: "ellipsis" as const, key: "head" },
      {
        type: "item" as const,
        item: navItems[currentIndex - 1],
        index: currentIndex - 1
      },
      { type: "item" as const, item: navItems[currentIndex], index: currentIndex },
      {
        type: "item" as const,
        item: navItems[currentIndex + 1],
        index: currentIndex + 1
      },
      { type: "ellipsis" as const, key: "tail" }
    ];
  })();

  return (
    <div className="h-dvh overflow-y-auto bg-bg1">
      <main className="page mx-auto w-full max-w-[920px] p-10 pb-[60px] pt-12 max-[760px]:w-full max-[760px]:p-3 max-[760px]:pt-3">
        {children}
      </main>
      <SiteGraphNav currentPath={currentPath} />
    </div>
  );
};
