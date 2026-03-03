import type { PropsWithChildren } from "react";

import { Modal } from "../ui/Modal";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
}

export const SettingsModal = ({
  open,
  onClose,
  title,
  children
}: PropsWithChildren<SettingsModalProps>) => {
  return (
    <Modal open={open} onClose={onClose} title={title}>
      {children}
    </Modal>
  );
};
