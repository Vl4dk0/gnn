import { Link } from "react-router-dom";

interface StoryCardProps {
  href: string;
  title: string;
  description: string;
  number?: number;
}

export const StoryCard = ({ href, title, description, number }: StoryCardProps) => {
  return (
    <Link to={href} className="block text-inherit no-underline">
      <article className="relative h-full rounded-[10px] border-2 border-line2 bg-bg2 p-5 transition-all duration-200 hover:border-textDim hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.35)]">
        {number && (
          <div className="absolute right-3 top-3 flex h-6 w-6 items-center justify-center rounded-full bg-line text-xs font-bold text-textDim">
            {number}
          </div>
        )}
        <h3 className="mb-2 pr-6 text-[1.08rem] font-bold text-textMain">{title}</h3>
        <p className="text-base leading-[1.7] text-textMuted">{description}</p>
      </article>
    </Link>
  );
};
