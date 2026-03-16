import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

interface PrimaryButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  fullWidth?: boolean;
}

export const PrimaryButton = ({
  children,
  className = "",
  fullWidth = true,
  ...props
}: PropsWithChildren<PrimaryButtonProps>) => {
  return (
    <button
      type="button"
      className={`${fullWidth ? "primary-action ui-action" : "ui-action rounded-md bg-accent px-5 py-2.5 text-sm font-semibold uppercase tracking-[1px] text-textMain transition-all duration-300"} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
