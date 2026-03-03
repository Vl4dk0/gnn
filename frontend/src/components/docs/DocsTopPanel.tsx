import { useState } from "react";

import { DOCS_SEQUENCE } from "../../pages/shared/docsNav";
import { PROJECT_LINKS } from "../../pages/shared/projects";
import type { FeatureKey } from "../../types/app";

interface DocsTopPanelProps {
  featureActive: Record<FeatureKey, boolean>;
  currentPath: string;
}

export const DocsTopPanel = ({ featureActive, currentPath }: DocsTopPanelProps) => {
  const [mobileToolsOpen, setMobileToolsOpen] = useState(false);

  const currentDocTitle =
    DOCS_SEQUENCE.find((item) => item.href === currentPath)?.title ?? "Project Overview";

  return (
    <section className="sticky top-0 z-30 border-b border-line bg-bg1/95 px-3 py-2 backdrop-blur">
      <div className="mx-auto max-w-[1100px]">
        <div className="flex items-end justify-between gap-3 max-[760px]:items-center">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
              GNN Thesis
            </p>
            <h1 className="text-[1.05rem] font-extrabold tracking-[0.1px] text-textMain">
              {currentDocTitle}
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setMobileToolsOpen((open) => !open)}
              className="hidden rounded-md border border-line2 bg-bg2 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.7px] text-textMain max-[760px]:inline-block"
            >
              {mobileToolsOpen ? "Hide tools" : "Tools"}
            </button>
            <a
              href="/"
              className="rounded-md border border-line2 bg-bg2 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.7px] text-textMain no-underline transition-colors hover:border-textDim"
            >
              Overview
            </a>
          </div>
        </div>

        <div className="mt-2 grid grid-cols-3 gap-2 max-[760px]:hidden">
          {PROJECT_LINKS.map((project) => {
            const active = featureActive[project.key];
            return (
              <a
                key={project.key}
                href={active ? project.href : "#"}
                onClick={(event) => {
                  if (!active) event.preventDefault();
                }}
                className={`rounded-md border px-3 py-2 text-inherit no-underline transition-colors ${
                  active
                    ? "border-line2 bg-bg2 text-textMain hover:border-textDim"
                    : "cursor-not-allowed border-[#333333] bg-[#161616] text-textDim opacity-65"
                }`}
              >
                <p className="text-sm font-bold leading-tight">{project.title}</p>
                <p className="mt-0.5 text-[11px] leading-[1.35] text-textMuted">
                  {project.description}
                </p>
                <p className="mt-1 text-[10px] font-semibold uppercase tracking-[0.7px] text-[#d8d8d8]">
                  {active ? "Open interactive graph editor" : "Unavailable"}
                </p>
              </a>
            );
          })}
        </div>

        {mobileToolsOpen ? (
          <div className="mt-2 hidden space-y-2 max-[760px]:block">
            {PROJECT_LINKS.map((project) => {
              const active = featureActive[project.key];
              return (
                <a
                  key={project.key}
                  href={active ? project.href : "#"}
                  onClick={(event) => {
                    if (!active) event.preventDefault();
                  }}
                  className={`block rounded-md border px-3 py-2 text-inherit no-underline transition-colors ${
                    active
                      ? "border-line2 bg-bg2 text-textMain hover:border-textDim"
                      : "cursor-not-allowed border-[#333333] bg-[#161616] text-textDim opacity-65"
                  }`}
                >
                  <p className="text-sm font-bold leading-tight">{project.title}</p>
                  <p className="mt-0.5 text-[11px] leading-[1.35] text-textMuted">
                    Open interactive graph editor
                  </p>
                </a>
              );
            })}
          </div>
        ) : null}
      </div>
    </section>
  );
};
