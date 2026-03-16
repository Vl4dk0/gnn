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
      className="ui-button-solid ui-surface-link max-[900px]:mx-auto max-[900px]:flex max-[900px]:w-full max-[900px]:max-w-[540px] max-[900px]:justify-center max-[900px]:text-center"
    >
      {isNext ? `Next: ${label} →` : `← Back: ${label}`}
    </a>
  );
};
