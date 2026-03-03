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
      className={`block rounded-lg border-2 p-5 text-inherit no-underline transition-all duration-200 ${
        active
          ? "cursor-pointer border-line bg-bg1 hover:border-textDim hover:bg-[#1e1e1e]"
          : "cursor-not-allowed border-[#333333] bg-[#161616] opacity-60"
      }`}
    >
      <div className="flex flex-col gap-2">
        <h3 className="m-0 text-[1.3rem] font-bold tracking-[0.3px] text-textMain">{title}</h3>
        <p className="m-0 text-[0.95rem] leading-[1.4] text-[#aaaaaa]">{description}</p>
      </div>
    </a>
  );
};
