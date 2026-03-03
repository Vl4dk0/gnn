import type { FeatureKey } from "../../types/app";
import { PROJECT_LINKS } from "../../pages/shared/projects";
import { ProjectNavCard } from "./ProjectNavCard";

interface SidebarNavProps {
  featureActive: Record<FeatureKey, boolean>;
  onNavigate: () => void;
}

export const SidebarNav = ({ featureActive, onNavigate }: SidebarNavProps) => {
  return (
    <>
      <div className="mb-5 shrink-0">
        <h1 className="mb-3 text-[2.2rem] font-extrabold leading-[1.1] tracking-[-0.5px] text-textMain">
          Bachelor Thesis
        </h1>
        <h2 className="text-[1.2rem] font-medium leading-[1.5] text-textDim">
          GNNs for Algebraic Graph Theory
        </h2>
      </div>

      <div className="shrink-0 space-y-5">
        {PROJECT_LINKS.map((project) => (
          <ProjectNavCard
            key={project.key}
            href={project.href}
            title={project.title}
            description={project.description}
            active={featureActive[project.key]}
            onNavigate={onNavigate}
          />
        ))}
      </div>
    </>
  );
};
