import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { SITE_NODES, SITE_EDGES } from "../../pages/shared/siteGraph";

interface SiteGraphNavProps {
  currentPath: string;
}

export const SiteGraphNav = ({ currentPath }: SiteGraphNavProps) => {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);
  const NODE_RADIUS = 5;
  const ACTIVE_RADIUS = 7;
  const VIEW_WIDTH = 250;
  const VIEW_HEIGHT = 500;

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const handleNavigation = (href: string) => {
    // Use View Transitions API if available
    if (document.startViewTransition) {
      document.startViewTransition(() => {
        navigate(href);
      });
    } else {
      navigate(href);
    }
  };

  return (
    <div className={`pointer-events-none fixed left-1/2 top-1/2 z-40 hidden -translate-y-1/2 min-[1380px]:block transition-opacity duration-300 ${
      isVisible ? "opacity-100" : "opacity-0"
    }`}
         style={{ marginLeft: '420px' }}>
      <style>{`
        ::view-transition-group(active-node-dot),
        ::view-transition-group(active-node-label) {
          animation-duration: 0.5s;
          animation-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Ensure the dot and label don't fade, just move */
        ::view-transition-old(active-node-dot),
        ::view-transition-new(active-node-dot),
        ::view-transition-old(active-node-label),
        ::view-transition-new(active-node-label) {
          animation: none;
          mix-blend-mode: normal;
          height: 100%;
        }
      `}</style>
      <div className="relative flex flex-col items-start w-fit">
        <div className="relative h-[500px]" style={{ width: VIEW_WIDTH }}>

          <svg
            viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
            width={VIEW_WIDTH}
            height={VIEW_HEIGHT}
            className="pointer-events-none overflow-visible"
          >
            {/* Trunk Line */}
            <line 
              x1={50} y1={20} x2={50} y2={480}
              stroke="currentColor" 
              strokeWidth="1" 
              className="text-line opacity-30"
            />

            {SITE_EDGES.map((edge, index) => {
              const from = SITE_NODES.find((n) => n.id === edge.from)!;
              const to = SITE_NODES.find((n) => n.id === edge.to)!;

              const fromX = from.x;
              const fromY = from.y;
              const toX = to.x;
              const toY = to.y;

              // If it's a side branch to an interactive editor
              if (fromX !== toX) {
                return (
                  <path
                    key={`edge-${index}`}
                    d={`M ${fromX} ${fromY} L ${toX - 15} ${fromY} Q ${toX} ${fromY} ${toX} ${fromY + 15} L ${toX} ${toY - 10}`}
                    stroke="currentColor"
                    strokeWidth="1.2"
                    fill="none"
                    className="text-textDim/40"
                  />
                );
              }

              return (
                <line
                  key={`edge-${index}`}
                  x1={fromX} y1={fromY + 10} x2={toX} y2={toY - 10}
                  stroke="currentColor"
                  strokeWidth="1.2"
                  className="text-textDim/40"
                />
              );
            })}

            {SITE_NODES.map((node) => {
              const isActive = currentPath === node.href || (node.id === 0 && currentPath === "/");
              const isApp = node.id >= 8;

              return (
                <g 
                  key={node.id} 
                  className="pointer-events-auto group cursor-pointer"
                  onClick={() => handleNavigation(node.href)}
                >
                  {/* Node Circle */}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={isActive ? ACTIVE_RADIUS : NODE_RADIUS}
                    className={`transition-all duration-300 ${
                      isActive
                        ? "fill-textMain stroke-[3px] stroke-textMain/30 shadow-[0_0_10px_rgba(255,255,255,0.3)]"
                        : "fill-bg1 stroke-1 stroke-textMuted group-hover:fill-textMuted"
                    }`}
                    style={isActive ? { viewTransitionName: 'active-node-dot' } as any : {}}
                  />

                  {/* Label */}
                  <text
                    x={node.x + 15}
                    y={node.y}
                    dy="0.35em"
                    className={`text-[13px] font-medium transition-all duration-300 ${
                      isActive 
                        ? "fill-textMain font-bold translate-x-1" 
                        : "fill-textMuted/60 group-hover:fill-textMain/80"
                    }`}
                    style={isActive ? { viewTransitionName: 'active-node-label' } as any : {}}
                  >
                    {node.title}
                  </text>

                  {/* Visual hint for "App" nodes */}
                  {isApp && !isActive && (
                    <rect 
                      x={node.x - 3} y={node.y - 3} width={6} height={6}
                      className="fill-bg1 stroke-[1px] stroke-textMuted/40"
                      style={{ transform: 'rotate(45deg)', transformOrigin: `${node.x}px ${node.y}px` }}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
};
