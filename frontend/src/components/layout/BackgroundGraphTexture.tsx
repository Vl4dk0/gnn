import { useMemo } from "react";

export const BackgroundGraphTexture = () => {
  // Use a unique ID for the pattern to avoid conflicts
  const patternId = "graph-texture-pattern";

  // Generate the pattern content once
  // We'll create a tile that repeats seamlesssly
  const tileContent = useMemo(() => {
    const tileSize = 400; // 400x400 tile
    const fontSize = 11;
    const lineHeight = 20; // Vertical spacing
    const colWidth = 50; // Horizontal spacing

    const rows = Math.ceil(tileSize / lineHeight);
    const cols = Math.ceil(tileSize / colWidth);

    const words = ["GRAPH", "NEURL", "NTWRK"];
    const elements = [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Strict grid, no randomness
        const x = c * colWidth;
        const y = r * lineHeight;

        // Deterministic word choice based on position to create a pattern
        const wordIndex = (r + c) % words.length;
        const word = words[wordIndex];

        elements.push(
          <text
            key={`${r}-${c}`}
            x={x + colWidth / 2} // Center in cell
            y={y + lineHeight / 2} // Center in cell
            className="fill-black dark:fill-white"
            fontSize={fontSize}
            fontFamily="monospace"
            dominantBaseline="middle"
            textAnchor="middle"
            opacity={0.66} // Increased visibility
          >
            {word}
          </text>
        );
      }
    }

    return { elements, tileSize };
  }, []);

  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden select-none opacity-[0.08]">
      <svg width="100%" height="100%">
        <defs>
          <pattern
            id={patternId}
            x="0"
            y="0"
            width={tileContent.tileSize}
            height={tileContent.tileSize}
            patternUnits="userSpaceOnUse"
          >
            {tileContent.elements}
          </pattern>
        </defs>
        <rect x="0" y="0" width="100%" height="100%" fill={`url(#${patternId})`} />
      </svg>
    </div>
  );
};
