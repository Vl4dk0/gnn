import { useCallback, useState } from "react";

export const useSidebarDrawer = () => {
  const [isOpen, setIsOpen] = useState(false);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback(() => {
    setIsOpen((value) => !value);
  }, []);

  return {
    isOpen,
    close,
    toggle
  };
};
