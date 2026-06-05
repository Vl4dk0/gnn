import { Link } from "react-router-dom";

interface EditorLink {
  href: string;
  label: string;
}

interface EditorLinksProps {
  links: EditorLink[];
}

export const EditorLinks = ({ links }: EditorLinksProps) => {
  return (
    <div className="mt-4 flex flex-col gap-3">
      {links.map(({ href, label }) => (
        <Link
          key={href}
          to={href}
          className="ui-button-solid ui-surface-link w-fit whitespace-nowrap"
        >
          {label}
        </Link>
      ))}
    </div>
  );
};
