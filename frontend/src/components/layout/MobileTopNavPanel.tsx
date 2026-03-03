import type { FeatureKey } from "../../types/app";
import { PROJECT_LINKS } from "../../pages/shared/projects";

interface MobileTopNavPanelProps {
  featureActive: Record<FeatureKey, boolean>;
}

export const MobileTopNavPanel = ({ featureActive }: MobileTopNavPanelProps) => {
  return (
    <section className="border-b border-line bg-bg2 px-3 py-3">
      <h2 className="text-sm font-bold uppercase tracking-[0.7px] text-textMain">
        GNN Thesis Modules
      </h2>
      <p className="mt-1 text-xs text-textDim">Quick access panel</p>

      <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
        {PROJECT_LINKS.map((project) => {
          const active = featureActive[project.key];

          return (
            <a
              key={project.key}
              href={active ? project.href : "#"}
              onClick={(event) => {
                if (!active) {
                  event.preventDefault();
                }
              }}
              className={`min-w-fit shrink-0 rounded-md border px-3 py-2 text-xs font-semibold uppercase tracking-[0.6px] no-underline transition-colors ${
                active
                  ? "border-line2 bg-bg1 text-textMain hover:border-textDim hover:bg-[#1e1e1e]"
                  : "cursor-not-allowed border-[#333333] bg-[#161616] text-textDim opacity-70"
              }`}
              title={project.description}
            >
              {project.title}
            </a>
          );
        })}
      </div>
    </section>
  );
};
