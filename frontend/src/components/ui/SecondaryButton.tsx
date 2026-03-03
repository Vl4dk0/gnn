import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

interface SecondaryButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  fullWidth?: boolean;
}

export const SecondaryButton = ({
  children,
  className = "",
  fullWidth = true,
  ...props
}: PropsWithChildren<SecondaryButtonProps>) => {
  return (
    <button
      type="button"
      className={`${fullWidth ? "w-full rounded-md bg-[#4a4a4a] px-7 py-3.5 text-base font-bold uppercase tracking-[1px] text-textMain shadow-card transition-all duration-300 hover:bg-[#555555] hover:shadow-cardHover" : "rounded-md bg-[#4a4a4a] px-5 py-2.5 text-sm font-semibold uppercase tracking-[1px] text-textMain transition-all duration-300 hover:bg-[#555555]"} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
