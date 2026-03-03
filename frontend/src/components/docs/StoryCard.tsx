interface StoryCardProps {
  href: string;
  index: string;
  title: string;
  description: string;
}

export const StoryCard = ({ href, index, title, description }: StoryCardProps) => {
  return (
    <a href={href} className="block text-inherit no-underline">
      <article className="h-full rounded-[10px] border-2 border-line2 bg-bg2 p-5 transition-all duration-200 hover:border-textDim hover:shadow-[0_8px_24px_rgba(0,0,0,0.35)]">
        <span className="mb-2.5 inline-block rounded border border-accent bg-bg1 px-2 py-1 text-[0.74rem] font-bold uppercase tracking-[0.7px] text-textMuted">
          {index}
        </span>
        <h3 className="mb-2 text-[1.08rem] font-bold text-textMain">{title}</h3>
        <p className="text-base leading-[1.7] text-textMuted">{description}</p>
      </article>
    </a>
  );
};
