import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  positionClassName: string;
}

export const IconButton = ({
  children,
  positionClassName,
  className = "",
  ...props
}: PropsWithChildren<IconButtonProps>) => {
  return (
    <button type="button" className={`icon-button ${positionClassName} ${className}`} {...props}>
      {children}
    </button>
  );
};
