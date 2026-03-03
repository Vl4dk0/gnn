import type { InputHTMLAttributes } from "react";

interface InputFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  id: string;
}

export const InputField = ({ label, id, className = "", ...props }: InputFieldProps) => {
  return (
    <div className="flex flex-col gap-2.5">
      <label htmlFor={id} className="label-base">
        {label}
      </label>
      <input id={id} className={`input-base ${className}`} {...props} />
    </div>
  );
};
