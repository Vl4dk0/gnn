import type { TextareaHTMLAttributes } from "react";

interface TextAreaFieldProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label: string;
  id: string;
}

export const TextAreaField = ({ label, id, className = "", ...props }: TextAreaFieldProps) => {
  return (
    <div className="flex flex-col gap-2.5">
      <label htmlFor={id} className="label-base">
        {label}
      </label>
      <textarea
        id={id}
        className={`input-base h-[200px] resize-none overflow-y-auto font-['Courier_New',monospace] ${className}`}
        {...props}
      />
    </div>
  );
};
