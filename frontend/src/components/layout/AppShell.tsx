import type { PropsWithChildren, ReactNode } from "react";

import { MobileNavToggle } from "./MobileNavToggle";
import { MobileOverlay } from "./MobileOverlay";

interface AppShellProps {
  sidebar: ReactNode;
  navOpen: boolean;
  onToggleNav: () => void;
  onCloseNav: () => void;
  sidebarClassName?: string;
  contentClassName?: string;
  enableMobileDrawer?: boolean;
  mobileTopPanel?: ReactNode;
}

export const AppShell = ({
  sidebar,
  navOpen,
  onToggleNav,
  onCloseNav,
  sidebarClassName = "",
  contentClassName = "",
  enableMobileDrawer = true,
  mobileTopPanel,
  children
}: PropsWithChildren<AppShellProps>) => {
  return (
    <div className="h-dvh overflow-hidden bg-bg1 min-[901px]:grid min-[901px]:grid-cols-[350px_1fr] max-[900px]:flex max-[900px]:flex-col">
      {enableMobileDrawer && (
        <>
          <MobileOverlay open={navOpen} onClick={onCloseNav} />
          <MobileNavToggle hidden={navOpen} onClick={onToggleNav} />
        </>
      )}

      <aside
        className={`min-h-0 overflow-y-auto border-r border-line bg-bg2 px-[30px] py-10 ${
          enableMobileDrawer
            ? `max-[900px]:fixed max-[900px]:bottom-0 max-[900px]:left-0 max-[900px]:top-0 max-[900px]:z-[1000] max-[900px]:w-[320px] max-[900px]:max-w-[85vw] max-[900px]:translate-x-[-100%] max-[900px]:border-b-0 max-[900px]:px-5 max-[900px]:py-8 max-[900px]:transition-transform max-[900px]:duration-300 ${
                navOpen ? "max-[900px]:translate-x-0" : ""
              }`
            : "max-[900px]:hidden"
        } ${sidebarClassName}`}
      >
        {sidebar}
      </aside>

      {mobileTopPanel && <div className="hidden max-[900px]:block">{mobileTopPanel}</div>}

      <main
        className={`min-h-0 overflow-hidden bg-bg0 p-[15px] max-[900px]:min-h-0 max-[900px]:w-screen max-[900px]:flex-1 max-[900px]:p-0 ${contentClassName}`}
      >
        {children}
      </main>
    </div>
  );
};
