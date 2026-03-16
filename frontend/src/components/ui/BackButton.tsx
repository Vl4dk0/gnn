interface BackButtonProps {
  href?: string;
  label?: string;
  className?: string;
  iconOnly?: boolean;
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

export const BackButton = ({
  href = "/",
  label = "Back",
  className = "",
  iconOnly = false
}: BackButtonProps) => {
  const backHref = href === "/" ? resolveBackHref() : href;

  return (
    <a
      href={backHref}
      className={
        iconOnly
          ? `ui-surface-link flex h-11 w-11 items-center justify-center rounded-full border border-line2 bg-bg1/92 text-textMain no-underline shadow-card backdrop-blur-sm transition-all duration-300 ${className}`
          : `ui-surface-link mt-4 block w-full rounded-md bg-accent px-7 py-3.5 text-center text-base font-bold uppercase tracking-[1px] text-textMain no-underline shadow-card transition-all duration-300 hover:shadow-cardHover ${className}`
      }
      title={label}
      aria-label={label}
    >
      {iconOnly ? (
        <svg
          className="h-5 w-5"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M15 18L9 12L15 6"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        label
      )}
    </a>
  );
};
