import React, { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import * as d3 from "d3";

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
  { id: "train", label: "Try It", path: "/docs/training", group: "doc" },
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
  if (path === "/docs/training") return 7;
  return null;
};

// "Star" topology: Root connects to all docs
const INITIAL_LINKS: GraphLink[] = [
  { source: "root", target: "gnns" },
  { source: "root", target: "arch" },
  { source: "root", target: "degree" },
  { source: "root", target: "cycle" },
  { source: "root", target: "assess" },
  { source: "root", target: "cage" },
  { source: "root", target: "train" },
  // Branches to apps (keep these attached to their respective docs)
  { source: "degree", target: "app-degree" },
  { source: "cycle", target: "app-cycle" },
  { source: "cage", target: "app-cage" }
];

export const SiteGraphNav = () => {
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
    if (!svgRef.current) return;

    const width = 300;
    const height = 600;

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
    const filter = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
      
    filter.append("feGaussianBlur")
      .attr("stdDeviation", "2.5")
      .attr("result", "coloredBlur");
      
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // 2. Prepare Data (clone to avoid mutation issues if re-run)
    const nodes: GraphNode[] = JSON.parse(JSON.stringify(INITIAL_NODES));
    const links: GraphLink[] = JSON.parse(JSON.stringify(INITIAL_LINKS));

    // Initialize positions: Root at top, others spread out
    nodes.forEach((n) => {
      if (n.id === "root") {
        n.x = width * 0.3;
        n.y = 50;
      } else {
        n.x = width * 0.3 + (Math.random() - 0.5) * 100;
        n.y = height / 2 + (Math.random() - 0.5) * 100;
      }
    });

    // 3. Create Simulation
    const simulation = d3
      .forceSimulation<GraphNode, GraphLink>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance((d) => {
             // Shorter links for apps
             if ((d.target as GraphNode).group === "app") return 50;
             // Variable length for star topology to spread them out vertically
             const targetId = (d.target as GraphNode).id;
             if (targetId === "gnns") return 60;
             if (targetId === "arch") return 120;
             if (targetId === "degree") return 180;
             if (targetId === "cycle") return 240;
             if (targetId === "assess") return 300;
             if (targetId === "cage") return 360;
             if (targetId === "train") return 420;
             return 100;
          }) 
      )
      .force("charge", d3.forceManyBody().strength(-150)) // Slightly reduced repel to keep it contained
      .force("center", d3.forceCenter(width * 0.3, height / 2).strength(0.005)) // Moved to left (0.3)
      .force(
        "y",
        d3
          .forceY()
          .y((d: any) => {
            // Vertical guidance
            if (d.group === "root") return 40;
            if (d.id === "gnns") return 100;
            if (d.id === "arch") return 160;
            if (d.id === "degree") return 220;
            if (d.id === "cycle") return 280;
            if (d.id === "assess") return 340;
            if (d.id === "cage") return 400;
            if (d.id === "train") return 460;
            
            return height / 2;
          })
          .strength((d: any) => d.group === "app" ? 0.02 : 0.1) // Very loose vertical constraint
      )
      .force("x", d3.forceX(width * 0.3).strength(0.01)); // Weak pull to left (0.3)

    simulationRef.current = simulation;

    // 4. Render Elements
    const link = svg
      .append("g")
      .attr("stroke", "#555") 
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

    // Halo (subtle glow for active state)
    nodeGroup
      .append("circle")
      .attr("class", "node-halo")
      .attr("r", 16) 
      .attr("fill", "none")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .attr("opacity", 0);

    // Main Node Circle
    nodeGroup
      .append("circle")
      .attr("class", "node-circle transition-all duration-300")
      .attr("r", (d: any) => d.group === "app" ? 6 : 10) // Larger for docs to hold number
      .attr("fill", "#1a1a1a")
      .attr("stroke", "#888") 
      .attr("stroke-width", 2);

    // Number inside circle (for non-apps)
    nodeGroup
      .filter((d: any) => d.group !== "app")
      .append("text")
      .text((d: any) => getPathNumber(d.path))
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em") // Vertically center
      .attr("font-size", "10px")
      .attr("font-weight", "bold")
      .attr("fill", "#888")
      .attr("class", "node-number pointer-events-none select-none transition-all duration-300");

    // Labels (Right of circle)
    nodeGroup
      .append("text")
      .text((d: any) => d.label)
      .attr("x", (d: any) => d.group === "app" ? 12 : 16)
      .attr("y", 4)
      .attr("font-family", "inherit")
      .attr("font-size", (d: any) => d.group === "app" ? "12px" : "14px")
      .attr("fill", "#aaa") 
      .attr("class", "node-label pointer-events-none select-none transition-all duration-300")
      .style("opacity", 0.9); 

    // Click Handler
    nodeGroup.on("click", (event: any, d: GraphNode) => {
      if (event.defaultPrevented) return; // Dragged
      navigate(d.path);
    });

    // Hover Effects
    nodeGroup
      .on("mouseenter", (event: any) => {
        const el = d3.select(event.currentTarget);
        el.select(".node-circle").attr("stroke", "#fff");
        el.select(".node-label").attr("fill", "#fff").style("opacity", 1);
        el.select(".node-number").attr("fill", "#fff");
      })
      .on("mouseleave", (event: any, d: GraphNode) => {
        const normalize = (p: string) => p.replace(/\/$/, "") || "/";
        const current = normalize(locationRef.current.pathname);
        const isActive = normalize(d.path) === current;

        if (!isActive) {
          const el = d3.select(event.currentTarget);
          el.select(".node-circle")
            .attr("fill", "#1a1a1a")
            .attr("stroke", "#888");

          el.select(".node-label").attr("fill", "#aaa").style("opacity", 0.9);
          el.select(".node-number").attr("fill", "#888");
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
  }, []); // Run once on mount

  // 6. Update Styles based on Current Path (Active State)
  useEffect(() => {
    if (!svgRef.current) return;

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
        return isActive ? "#333" : "#1a1a1a"; 
      })
      .attr("stroke", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "#fff" : "#888"; 
      })
      .attr("stroke-width", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? 2.5 : 2; 
      })
      .attr("r", (d: any) => {
        const isActive = normalize(d.path) === current;
        if (d.group === "app") return isActive ? 8 : 6;
        return isActive ? 12 : 10;
      });
      
    // Update Number Styles
    svg
      .selectAll<SVGTextElement, GraphNode>(".node-number")
      .transition()
      .duration(300)
      .attr("fill", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "#fff" : "#888";
      });

    // Update Halo Styles
    svg
      .selectAll<SVGGElement, GraphNode>(".node-halo")
      .transition()
      .duration(300)
      .attr("opacity", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? 0.15 : 0; 
      })
      .attr("r", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? 18 : 6;
      });

    // Update Label Styles
    svg
      .selectAll<SVGTextElement, GraphNode>(".node-label")
      .transition()
      .duration(300)
      .attr("fill", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "#fff" : "#aaa";
      })
      .attr("font-weight", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? "600" : "400";
      })
      .style("opacity", (d: any) => {
        const isActive = normalize(d.path) === current;
        return isActive ? 1 : 0.9;
      });
      
  }, [location.pathname]);

  return (
    <div className="hidden min-[1380px]:block fixed right-10 top-1/2 -translate-y-1/2 z-50">
      <div className="relative w-[300px] h-[600px]">
        <svg
          ref={svgRef}
          className="w-full h-full drop-shadow-2xl"
          style={{ overflow: "visible" }}
        />
      </div>
    </div>
  );
};
