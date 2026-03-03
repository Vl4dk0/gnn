interface MobileNavToggleProps {
  hidden: boolean;
  onClick: () => void;
}

export const MobileNavToggle = ({ hidden, onClick }: MobileNavToggleProps) => {
  return (
    <button
      type="button"
      className={`fixed left-0 top-10 z-[990] hidden h-16 w-8 items-center justify-center rounded-r-lg border border-l-0 border-line2 bg-bg2 shadow-[2px_2px_8px_rgba(0,0,0,0.3)] transition-transform duration-300 max-[900px]:flex ${hidden ? "-translate-x-full" : ""}`}
      onClick={onClick}
      aria-label="Open sidebar"
    >
      <span className="block h-2 w-2 -translate-x-0.5 rotate-[-45deg] border-b-2 border-r-2 border-[#aaaaaa]" />
    </button>
  );
};
