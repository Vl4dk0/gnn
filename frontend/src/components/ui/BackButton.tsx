interface BackButtonProps {
  href?: string;
  label?: string;
}

const resolveBackHref = () => {
  if (typeof window === "undefined") {
    return "/";
  }

  const referrer = document.referrer;
  if (!referrer) {
    return "/";
  }

  try {
    const url = new URL(referrer);
    if (url.origin === window.location.origin) {
      return `${url.pathname}${url.search}${url.hash}`;
    }
  } catch {
    return "/";
  }

  return "/";
};

export const BackButton = ({ href = "/", label = "Back" }: BackButtonProps) => {
  const backHref = href === "/" ? resolveBackHref() : href;

  return (
    <a
      href={backHref}
      className="mt-4 block w-full rounded-md bg-accent px-7 py-3.5 text-center text-base font-bold uppercase tracking-[1px] text-textMain no-underline shadow-card transition-all duration-300 hover:bg-accentHover hover:shadow-cardHover"
      title={label}
    >
      {label}
    </a>
  );
};
