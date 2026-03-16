import { Link } from "react-router-dom";

interface StoryCardProps {
  href: string;
  title: string;
  description: string;
}

export const StoryCard = ({ href, title, description }: StoryCardProps) => {
  return (
    <Link to={href} className="block text-inherit no-underline">
      <article className="h-full rounded-[10px] border-2 border-line2 bg-bg2 p-5 transition-all duration-200 hover:border-textDim hover:shadow-[0_8px_24px_rgba(0,0,0,0.35)]">
        <h3 className="mb-2 text-[1.08rem] font-bold text-textMain">{title}</h3>
        <p className="text-base leading-[1.7] text-textMuted">{description}</p>
      </article>
    </Link>
  );
};
