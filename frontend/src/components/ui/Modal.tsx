import type { PropsWithChildren } from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
}

export const Modal = ({ open, onClose, title, children }: PropsWithChildren<ModalProps>) => {
  return (
    <div
      className={`modal-overlay ${open ? "show" : ""}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="modal-content">
        <h2 className="mb-6 border-b-2 border-line pb-4 text-2xl text-textMain">{title}</h2>
        {children}
      </div>
    </div>
  );
};
