import type { PropsWithChildren, ReactNode } from "react";

import { AppShell } from "../layout/AppShell";

interface GraphPageLayoutProps {
  sidebar: ReactNode;
  navOpen: boolean;
  onToggleNav: () => void;
  onCloseNav: () => void;
}

export const GraphPageLayout = ({
  sidebar,
  navOpen,
  onToggleNav,
  onCloseNav,
  children
}: PropsWithChildren<GraphPageLayoutProps>) => {
  return (
    <AppShell
      sidebar={sidebar}
      navOpen={navOpen}
      onToggleNav={onToggleNav}
      onCloseNav={onCloseNav}
      sidebarClassName="flex flex-col gap-[25px]"
      contentClassName="flex items-center justify-center"
    >
      {children}
    </AppShell>
  );
};
