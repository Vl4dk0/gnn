import { Link } from "react-router-dom";

interface EditorLink {
  href: string;
  label: string;
  description: string;
}

interface EditorLinksProps {
  links: EditorLink[];
}

export const EditorLinks = ({ links }: EditorLinksProps) => {
  return (
    <div className="flex flex-col gap-3">
      {links.map(({ href, label, description }) => (
        <div key={href} className="flex flex-wrap items-center gap-x-4 gap-y-1 sm:flex-row">
          <Link to={href} className="ui-button-solid ui-surface-link whitespace-nowrap">
            {label}
          </Link>
          <span className="text-base leading-[1.7] text-textMuted">{description}</span>
        </div>
      ))}
    </div>
  );
};
