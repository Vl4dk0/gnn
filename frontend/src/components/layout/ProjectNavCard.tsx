interface ProjectNavCardProps {
  href: string;
  title: string;
  description: string;
  active: boolean;
  onNavigate: () => void;
}

export const ProjectNavCard = ({
  href,
  title,
  description,
  active,
  onNavigate
}: ProjectNavCardProps) => {
  return (
    <a
      href={active ? href : "#"}
      onClick={(event) => {
        if (!active) {
          event.preventDefault();
          return;
        }

        onNavigate();
      }}
      className={`ui-surface-link block rounded-lg border-2 p-5 text-inherit no-underline transition-all duration-200 ${
        active
          ? "cursor-pointer border-line bg-bg1"
          : "cursor-not-allowed border-[#d0d0d0] bg-[#e8e8e8] dark:border-[#333333] dark:bg-[#161616] opacity-60"
      }`}
    >
      <div className="flex flex-col gap-2">
        <h3 className="m-0 text-[1.3rem] font-bold tracking-[0.3px] text-textMain">{title}</h3>
        <p className="m-0 text-[0.95rem] leading-[1.4] text-[#777777] dark:text-[#aaaaaa]">{description}</p>
      </div>
    </a>
  );
};
