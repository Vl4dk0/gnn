import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import * as d3 from "d3";
import { AnimatePresence, motion } from "framer-motion";

// Data Structure
interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  path: string;
  group: "doc" | "app" | "root";
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
}

// Initial Data
const INITIAL_NODES: GraphNode[] = [
  { id: "root", label: "Overview", path: "/", group: "root" },
  { id: "gnns", label: "GNNs", path: "/docs/gnns", group: "doc" },
  { id: "arch", label: "Architectures", path: "/docs/architecture", group: "doc" },
  { id: "degree", label: "Degree", path: "/docs/module-degree", group: "doc" },
  { id: "cycle", label: "Cycle", path: "/docs/module-min-cycle", group: "doc" },
  { id: "assess", label: "Assessment", path: "/docs/module-assessment", group: "doc" },
  { id: "cage", label: "Cage", path: "/docs/module-cage", group: "doc" },
  { id: "voltage", label: "Voltage", path: "/docs/voltage", group: "doc" },
  // Apps (Editors)
  { id: "app-degree", label: "Editor", path: "/degree", group: "app" },
  { id: "app-cycle", label: "Editor", path: "/min_cycle", group: "app" },
  { id: "app-cage", label: "Editor", path: "/cage", group: "app" }
];

// Map path to index number (0 for root, 1-7 for docs)
const getPathNumber = (path: string): number | null => {
  if (path === "/") return 0;
  if (path === "/docs/gnns") return 1;
  if (path === "/docs/architecture") return 2;
  if (path === "/docs/module-degree") return 3;
  if (path === "/docs/module-min-cycle") return 4;
  if (path === "/docs/module-assessment") return 5;
  if (path === "/docs/module-cage") return 6;
  if (path === "/docs/voltage") return 7;
  return null;
};

// Star topology + sequential edges between consecutive docs
const INITIAL_LINKS: GraphLink[] = [
  // Root connects to all docs
  { source: "root", target: "gnns" },
  { source: "root", target: "arch" },
  { source: "root", target: "degree" },
  { source: "root", target: "cycle" },
  { source: "root", target: "assess" },
  { source: "root", target: "cage" },
  { source: "root", target: "voltage" },
  // Sequential edges (docs flow in order)
  { source: "gnns", target: "arch" },
  { source: "arch", target: "degree" },
  { source: "degree", target: "cycle" },
  { source: "cycle", target: "assess" },
  { source: "assess", target: "cage" },
  { source: "cage", target: "voltage" },
  // Branches to apps
  { source: "degree", target: "app-degree" },
  { source: "cycle", target: "app-cycle" },
  { source: "cage", target: "app-cage" },
  { source: "voltage", target: "app-cage" }
];

export const SiteGraphNav = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const locationRef = useRef(location);

  useEffect(() => {
    locationRef.current = location;
  }, [location]);

  // Keep simulation state ref to avoid re-creating it
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);

  useEffect(() => {
    if (!isOpen || !svgRef.current) return;

    const width = 300;
    const height = 630;

    // 1. Setup SVG
    const svg = d3
      .select(svgRef.current)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .style("overflow", "visible");

    // Clear previous if any
    svg.selectAll("*").remove();

    // Definitions for filters/gradients
    const defs = svg.append("defs");

    // Glow filter
    const filter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");

    filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur");

    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // 2. Prepare Data (clone to avoid mutation issues if re-run)
    const nodes: GraphNode[] = JSON.parse(JSON.stringify(INITIAL_NODES));
    const links: GraphLink[] = JSON.parse(JSON.stringify(INITIAL_LINKS));

    // Initialize positions: Root at top, others spread out
    nodes.forEach((n) => {
      if (n.id === "root") {
        n.x = width * 0.3; // Back to left side
        n.y = 50;
      } else {
        n.x = width * 0.3 + (Math.random() - 0.5) * 80; // Moderate spread
        n.y = height / 2 + (Math.random() - 0.5) * 100;
      }
    });

    // 3. Create Simulation
    const simulation = d3
      .forceSimulation<GraphNode, GraphLink>(nodes)
      .alphaDecay(0.008) // 3x slower decay (default is ~0.0228) to prolong movement
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance((d) => {
            const src = (d.source as GraphNode).id;
            const tgt = (d.target as GraphNode).id;
            // App branches: short
            if ((d.target as GraphNode).group === "app") return 55;
            // Sequential edges between consecutive docs: short, tight chain
            const seqPairs = [
              ["gnns", "arch"], ["arch", "degree"], ["degree", "cycle"],
              ["cycle", "assess"], ["assess", "cage"], ["cage", "voltage"]
            ];
            if (seqPairs.some(([a, b]) => (src === a && tgt === b) || (src === b && tgt === a))) {
              return 65;
            }
            // Star edges from root: uniform medium distance
            if (src === "root" || tgt === "root") return 200;
            return 100;
          })
      )
      .force("charge", d3.forceManyBody().strength(-150))
      .force("center", d3.forceCenter(width * 0.3, height / 2).strength(0.003))
      .force(
        "y",
        d3
          .forceY()
          .y((d: any) => {
            // Vertical guidance — evenly spaced chain
            if (d.group === "root") return 35;
            if (d.id === "gnns") return 100;
            if (d.id === "arch") return 165;
            if (d.id === "degree") return 230;
            if (d.id === "cycle") return 295;
            if (d.id === "assess") return 360;
            if (d.id === "cage") return 425;
            if (d.id === "voltage") return 490;

            return height / 2;
          })
          .strength((d: any) => (d.group === "app" ? 0.02 : 0.15))
      )
      .force("x", d3.forceX(width * 0.3).strength(0.01)); // Weak pull to left (0.3)

    simulationRef.current = simulation;

    // 4. Render Elements
    const link = svg
      .append("g")
      .attr("stroke", "var(--graph-nav-link)")
      .attr("stroke-opacity", 0.6)
      .attr("fill", "none")
      .selectAll("path")
      .data(links)
      .join("path")
      .attr("stroke-width", 2);

    const nodeGroup = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(d3.drag<any, any>().on("start", dragstarted).on("drag", dragged).on("end", dragended));

    // Main Node Circle
    nodeGroup
      .append("circle")
      .attr("class", "node-circle transition-all duration-300")
      .attr("r", (d: any) => (d.group === "app" ? 8 : 12))
      .attr("fill", "var(--graph-nav-node-fill)")
      .attr("stroke", "var(--graph-nav-node-stroke)")
      .attr("stroke-width", 2);

    // Number inside circle (for non-apps)
    nodeGroup
      .filter((d: any) => d.group !== "app")
      .append("text")
      .text((d: any) => getPathNumber(d.path))
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em") // Vertically center
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .attr("fill", "var(--graph-nav-number)")
      .attr("class", "node-number pointer-events-none select-none transition-all duration-300");

    // Labels (Beneath Node)
    const labels = nodeGroup
      .append("text")
      .text((d: any) => d.label)
      .attr("class", "node-label select-none transition-all duration-300")
      .attr("text-anchor", "middle") // Center align
      .attr("x", 0)
      .attr("y", (d: any) => (d.group === "app" ? 22 : 28)) // Position below node
      .attr("font-family", "inherit")
      .attr("font-size", (d: any) => (d.group === "app" ? "10px" : "11px"))
      .attr("font-weight", (d: any) => (d.group === "app" ? "400" : "500"))
      .attr("letter-spacing", "0.02em")
      .attr("fill", "var(--graph-nav-label)") // Default dimmed
      .style("text-shadow", "none");

    // Click Handler
    nodeGroup.on("click", (event: any, d: GraphNode) => {
      if (event.defaultPrevented) return; // Dragged
      navigate(d.path);
    });

    // Hover Effects
    nodeGroup
      .on("mouseenter", (event: any) => {
        const el = d3.select(event.currentTarget);
        el.select(".node-circle").attr("stroke", "var(--graph-nav-hover-text)");
        el.select(".node-number").attr("fill", "var(--graph-nav-hover-text)");

        // Highlight label
        el.select(".node-label").attr("fill", "var(--graph-nav-hover-text)");
      })
      .on("mouseleave", (event: any, d: GraphNode) => {
        const normalize = (p: string) => p.replace(/\/$/, "") || "/";
        const current = normalize(locationRef.current.pathname);
        const isActive = normalize(d.path) === current;

        if (!isActive) {
          const el = d3.select(event.currentTarget);
          el.select(".node-circle")
            .attr("fill", "var(--graph-nav-node-fill)")
            .attr("stroke", "var(--graph-nav-node-stroke)");

          el.select(".node-number").attr("fill", "var(--graph-nav-number)");

          // Dim label
          el.select(".node-label").attr("fill", "var(--graph-nav-label)");
        }
      });

    // 5. Simulation Tick
    simulation.on("tick", () => {
      link.attr("d", (d: any) => {
        const x1 = d.source.x;
        const y1 = d.source.y;
        const x2 = d.target.x;
        const y2 = d.target.y;

        // "Cable" style: Vertical S-curve (like node editors)
        // This creates a smooth, weighted feel for hierarchical graphs
        // Control points maintain vertical tangency at start/end
        const distY = Math.abs(y2 - y1);
        const smoothY = distY * 0.5; // Amount of vertical straightness before bending

        // If nodes are very close vertically, add some minimum curve space
        const c1y = y1 + Math.max(smoothY, 20);
        const c2y = y2 - Math.max(smoothY, 20);

        return `M${x1},${y1} C${x1},${c1y} ${x2},${c2y} ${x2},${y2}`;
      });

      nodeGroup.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    // Drag Functions
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [isOpen]); // Re-run when opened

  // 6. Update Styles based on Current Path (Active State)
  useEffect(() => {
    if (!isOpen || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const normalize = (p: string) => p.replace(/\/$/, "") || "/";
    const current = normalize(location.pathname);

    // Update Circle Styles
    svg
      .selectAll<SVGGElement, GraphNode>(".node-circle")
      .transition()
      .duration(300)
      .attr("fill", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "var(--graph-nav-active-fill)" : "var(--graph-nav-node-fill)";
      })
      .attr("stroke", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "var(--graph-nav-active-stroke)" : "var(--graph-nav-node-stroke)";
      })
      .attr("stroke-width", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? 2.5 : 2;
      })
      .attr("r", (d: any) => {
        const isActive = normalize(d.path) === current;
        if (d.group === "app") return isActive ? 10 : 8;
        return isActive ? 14 : 12;
      });

    // Update Number Styles
    svg
      .selectAll<SVGTextElement, GraphNode>(".node-number")
      .transition()
      .duration(300)
      .attr("fill", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "var(--graph-nav-active-text)" : "var(--graph-nav-number)";
      });

    // Update Halo Styles
    // Halo removed per request
    svg.selectAll(".node-halo").remove();

    // Update Label Styles
    svg
      .selectAll<SVGTextElement, GraphNode>(".node-label")
      .transition()
      .duration(300)
      .attr("fill", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "var(--graph-nav-active-text)" : "var(--graph-nav-label)";
      });
  }, [location.pathname, isOpen]);

  return (
    <div className="fixed right-10 top-1/2 z-50 hidden h-[630px] w-[300px] -translate-y-1/2 pointer-events-none min-[1380px]:block">
      <div className="relative h-full w-full">
        {/* Toggle Button - always visible, top-right of container */}
        <div
          className="absolute right-0 top-0 z-50 flex items-center gap-2 pointer-events-auto"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <AnimatePresence>
            {isHovered && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
                className="whitespace-nowrap rounded-md border border-line2 bg-bg1/80 px-2 py-1 text-xs font-medium text-textMain shadow-sm backdrop-blur-sm"
              >
                {isOpen ? "Hide" : "Show"}
              </motion.div>
            )}
          </AnimatePresence>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-line2 bg-bg1/40 text-textMuted shadow-sm backdrop-blur-sm transition-all hover:bg-bg2 hover:text-textMain focus:outline-none"
            aria-label={isOpen ? "Hide Graph Navigation" : "Show Graph Navigation"}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`transition-transform duration-300 ${!isOpen ? "rotate-90" : ""}`}
            >
              {isOpen ? (
                // Minus (Minimize)
                <line x1="5" y1="12" x2="19" y2="12" />
              ) : (
                // Simple Network (Expand)
                <>
                  <circle cx="18" cy="5" r="3" />
                  <circle cx="6" cy="12" r="3" />
                  <circle cx="18" cy="19" r="3" />
                  <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                  <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
                </>
              )}
            </svg>
          </button>
        </div>

        {/* Graph Content */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 pointer-events-auto"
            >
              <svg
                ref={svgRef}
                className="h-full w-full drop-shadow-2xl"
                style={{ overflow: "visible" }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
