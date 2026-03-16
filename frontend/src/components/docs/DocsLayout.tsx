import type { PropsWithChildren } from "react";

import { DOCS_SEQUENCE } from "../../pages/shared/docsNav";
import type { FeatureKey } from "../../types/app";

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
      <main className="page mx-auto w-full max-w-[920px] p-10 pb-[60px] max-[760px]:w-full max-[760px]:p-3 max-[760px]:pt-3">
        <nav
          className="mb-5 flex h-12 items-center justify-center max-[760px]:mb-4 max-[760px]:h-10"
          aria-label="Docs sequence"
        >
          <div className="hidden max-w-full items-center gap-2 px-2 min-[761px]:flex">
            {navItems.map((item, index) => {
              const isActive = item.href === currentPath;

              return (
                <div key={item.href} className="flex items-center gap-2">
                  <a
                    href={item.href}
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-sm font-bold no-underline transition-colors ${
                      isActive
                        ? "border-textMain bg-textMain text-bg1"
                        : "border-line2 bg-transparent text-textMuted hover:border-textDim hover:text-textMain"
                    }`}
                    aria-label={item.title}
                    title={item.title}
                  >
                    {index}
                  </a>
                  {index < navItems.length - 1 && (
                    <span className="text-sm font-semibold text-textDim" aria-hidden="true">
                      →
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          <div className="flex max-w-full items-center gap-2 overflow-hidden px-2 min-[761px]:hidden">
            {mobileTokens.map((token, tokenIndex) => {
              const isLast = tokenIndex === mobileTokens.length - 1;

              return (
                <div
                  key={token.type === "item" ? token.item.href : token.key}
                  className="flex items-center gap-2"
                >
                  {token.type === "item" ? (
                    <a
                      href={token.item.href}
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-xs font-bold no-underline transition-colors ${
                        token.item.href === currentPath
                          ? "border-textMain bg-textMain text-bg1"
                          : "border-line2 bg-transparent text-textMuted"
                      }`}
                      aria-label={token.item.title}
                      title={token.item.title}
                    >
                      {token.index}
                    </a>
                  ) : (
                    <span className="px-0.5 text-sm font-bold tracking-[0.2em] text-textDim">
                      ...
                    </span>
                  )}
                  {!isLast && (
                    <span className="text-xs font-semibold text-textDim" aria-hidden="true">
                      →
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </nav>
        {children}
      </main>
    </div>
  );
};
