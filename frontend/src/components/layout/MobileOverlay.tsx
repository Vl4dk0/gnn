interface MobileOverlayProps {
  open: boolean;
  onClick: () => void;
}

export const MobileOverlay = ({ open, onClick }: MobileOverlayProps) => {
  return (
    <div
      className={`pointer-events-none fixed inset-0 z-[999] hidden bg-black/60 opacity-0 backdrop-blur-[2px] transition-opacity duration-300 max-[900px]:block ${open ? "pointer-events-auto opacity-100" : ""}`}
      onClick={onClick}
    />
  );
};
