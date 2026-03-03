interface DocsNextButtonProps {
  href: string;
  label: string;
  direction?: "next" | "back";
}

export const DocsNextButton = ({ href, label, direction = "next" }: DocsNextButtonProps) => {
  const isNext = direction === "next";
  return (
    <a
      href={href}
      className="inline-block rounded-md bg-accent px-5 py-3 text-sm font-bold uppercase tracking-[0.8px] text-textMain no-underline shadow-card transition-all duration-300 hover:bg-accentHover hover:shadow-cardHover"
    >
      {isNext ? `Next: ${label} →` : `← Back: ${label}`}
    </a>
  );
};
