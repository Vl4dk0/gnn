import type { PropsWithChildren } from "react";

interface DocsCardProps {
  className?: string;
}

export const DocsCard = ({ children, className = "" }: PropsWithChildren<DocsCardProps>) => {
  return (
    <section className={`mb-4 rounded-[10px] border-2 border-line2 bg-bg2 p-[22px] ${className}`}>
      {children}
    </section>
  );
};

export const DocsHero = ({ children, className = "" }: PropsWithChildren<DocsCardProps>) => {
  return (
    <section className={`mb-4 rounded-[10px] border-2 border-line2 bg-bg2 p-[22px] ${className}`}>
      {children}
    </section>
  );
};
