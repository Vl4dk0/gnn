import type { PropsWithChildren } from "react";
import { Link, useLocation } from "react-router-dom";

import { DOCS_SEQUENCE } from "../../pages/shared/docsNav";

interface NavItem {
  href: string;
  title: string;
}

type MobileNavToken =
  | { type: "item"; item: NavItem; index: number }
  | { type: "ellipsis"; key: string };

export const DocsLayout = ({
  children
}: PropsWithChildren<unknown>) => {
  const location = useLocation();
  const currentPath = location.pathname;

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
    <div className="relative z-10 h-dvh overflow-y-auto">
      {/* Mobile Navigation (Bullets) */}
      <div className="sticky top-0 z-30 flex w-full items-center justify-center gap-1 border-b border-line bg-bg1/90 p-3 backdrop-blur md:hidden">
        {mobileTokens.map((token, i) => {
          const isLast = i === mobileTokens.length - 1;
          const content =
            token.type === "ellipsis" ? (
              <span className="text-textDim">...</span>
            ) : (
              <Link
                to={token.item.href}
                className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-colors ${
                  token.item.href === currentPath
                    ? "bg-textMain text-bg0 font-bold shadow-md"
                    : "bg-bg2 text-textDim hover:bg-bg2/80 hover:text-textMuted"
                }`}
              >
                {token.index}
              </Link>
            );

          return (
            <div key={token.type === "item" ? token.item.href : `ellipsis-${token.key}`} className="flex items-center gap-1">
              {content}
              {!isLast && <span className="text-[10px] text-line2">→</span>}
            </div>
          );
        })}
      </div>

      <main className="page mx-auto w-full max-w-[920px] p-10 pb-[60px] pt-12 max-[760px]:w-full max-[760px]:p-3 max-[760px]:pt-3">
        {children}
      </main>
    </div>
  );
};
