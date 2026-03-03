import type { PropsWithChildren, SelectHTMLAttributes } from "react";

interface SelectFieldProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  id: string;
}

export const SelectField = ({
  label,
  id,
  children,
  className = "",
  ...props
}: PropsWithChildren<SelectFieldProps>) => {
  return (
    <div className="mb-6">
      <label htmlFor={id} className="label-base mb-2.5 block normal-case tracking-normal">
        {label}
      </label>
      <select
        id={id}
        className={`w-full cursor-pointer rounded-md border-2 border-line2 bg-bg1 px-3 py-2.5 pr-9 text-[0.95em] text-textMuted outline-none transition-colors hover:border-[#666666] focus:border-textDim ${className}`}
        {...props}
      >
        {children}
      </select>
    </div>
  );
};
