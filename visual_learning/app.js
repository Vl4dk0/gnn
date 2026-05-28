document.addEventListener('DOMContentLoaded', () => {
    const baseCanvas = document.getElementById('base-canvas');
    const liftedCanvas = document.getElementById('lifted-canvas');
    const explanationText = document.getElementById('explanation-text');
    
    let baseGraphInstance = ForceGraph()(baseCanvas);
    let liftedGraphInstance = ForceGraph()(liftedCanvas);

    // Inputs
    const baseTypeInput = document.getElementById('base-graph-type');
    const baseSizeInput = document.getElementById('base-size');
    const groupSizeInput = document.getElementById('group-size');
    const voltagesInput = document.getElementById('voltage-elements');
    const baseSizeVal = document.getElementById('base-size-val');
    const groupSizeVal = document.getElementById('group-size-val');

    let hoveredBaseNode = null;
    let hoveredBaseEdge = null;
    
    let baseData = { nodes: [], links: [] };
    let liftedData = { nodes: [], links: [] };
    let zn = 5;

    // Config Instances
    baseGraphInstance
        .nodeRelSize(8)
        .linkDirectionalArrowLength(6)
        .linkDirectionalArrowRelPos(1)
        .linkColor(link => link.id === hoveredBaseEdge ? '#fbbf24' : 'rgba(255,255,255,0.4)')
        .linkWidth(link => link.id === hoveredBaseEdge ? 3 : 1)
        .nodeColor(node => node.id === hoveredBaseNode ? '#fbbf24' : node.color)
        .linkLabel(link => `Edge from ${link.source.id || link.source} to ${link.target.id || link.target} (Voltage: ${link.voltage})`)
        .onNodeHover(node => {
            hoveredBaseNode = node ? node.id : null;
            updateHighlights();
            updateExplanation(node, null);
        })
        .onLinkHover(link => {
            hoveredBaseEdge = link ? link.id : null;
            updateHighlights();
            updateExplanation(null, link);
        });
        
    // Customize Base Graph drawing to show voltage text on edges
    baseGraphInstance.linkCanvasObjectMode(() => 'after');
    baseGraphInstance.linkCanvasObject((link, ctx) => {
        const MAX_FONT_SIZE = 4;
        const start = link.source;
        const end = link.target;
        
        // ignore unbound links
        if (typeof start !== 'object' || typeof end !== 'object') return;
        
        // calculate label positioning
        const textPos = Object.assign(...['x', 'y'].map(c => ({
            [c]: start[c] + (end[c] - start[c]) / 2 // calc middle point
        })));
        
        ctx.font = `6px Sans-Serif`;
        ctx.fillStyle = link.id === hoveredBaseEdge ? '#fbbf24' : 'rgba(255,255,255,0.8)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`α=${link.voltage}`, textPos.x, textPos.y - 4);
    });

    liftedGraphInstance
        .nodeRelSize(5)
        .linkDirectionalArrowLength(3)
        .linkDirectionalArrowRelPos(1)
        .linkColor(link => link.baseEdgeId === hoveredBaseEdge ? '#fbbf24' : 'rgba(255,255,255,0.15)')
        .linkWidth(link => link.baseEdgeId === hoveredBaseEdge ? 2 : 1)
        .nodeColor(node => node.baseNodeId === hoveredBaseNode ? '#fbbf24' : node.color)
        .d3Force('charge').strength(-60);

    function updateHighlights() {
        // Trigger re-evaluations of colors without regenerating physics
        baseGraphInstance
            .nodeColor(baseGraphInstance.nodeColor())
            .linkColor(baseGraphInstance.linkColor())
            .linkWidth(baseGraphInstance.linkWidth());
        
        liftedGraphInstance
            .nodeColor(liftedGraphInstance.nodeColor())
            .linkColor(liftedGraphInstance.linkColor())
            .linkWidth(liftedGraphInstance.linkWidth());
    }

    function updateExplanation(node, link) {
        if (node) {
            explanationText.innerHTML = `You are hovering over <span class="highlight-text">Base Node ${node.id}</span>. In the Lifted Graph, notice how this single node has been cloned into <span class="highlight-text">${zn} fiber nodes</span> (the yellow dots), representing elements 0 to ${zn-1} of the group Zn.`;
        } else if (link) {
            explanationText.innerHTML = `You are hovering over <span class="highlight-text">Base Edge (Voltage = ${link.voltage})</span>. In the Lifted Graph, notice the yellow edges! Each fiber node 'g' is strictly connected to node '(g + ${link.voltage}) mod ${zn}'. This is the core algebraic shift of Voltage Graph Lifts!`;
        } else {
            explanationText.innerHTML = `Hover over any node or edge in the Base Graph (left) to see how it "lifts" into the generated graph (right).`;
        }
    }

    function getBaseGraphEdges(type, n) {
        let edges = [];
        let numVertices = n;
        
        if (type === 'cycle') {
            for(let i=0; i<n; i++) edges.push({source: i, target: (i+1)%n});
        } else if (type === 'dipole') {
            numVertices = 2;
            for(let i=0; i<n; i++) edges.push({source: 0, target: 1});
        } else if (type === 'bouquet') {
            numVertices = 1;
            for(let i=0; i<n; i++) edges.push({source: 0, target: 0});
        }
        return { edges, numVertices };
    }

    function getColor(v, total) {
        return `hsl(${(v * 360 / Math.max(1, total))}, 70%, 60%)`;
    }

    function generateGraphs() {
        const type = baseTypeInput.value;
        const n = parseInt(baseSizeInput.value, 10);
        zn = parseInt(groupSizeInput.value, 10);
        
        baseSizeVal.textContent = n;
        groupSizeVal.textContent = zn;

        let customVoltages = [];
        if (voltagesInput.value.trim()) {
            customVoltages = voltagesInput.value.split(',').map(s => parseInt(s.trim(), 10)).filter(v => !isNaN(v));
        }

        const { edges, numVertices } = getBaseGraphEdges(type, n);

        baseData = { nodes: [], links: [] };
        liftedData = { nodes: [], links: [] };

        // Generate Base Nodes
        for(let v=0; v<numVertices; v++) {
            baseData.nodes.push({ id: v, name: `Base Node ${v}`, color: getColor(v, numVertices) });
            
            // Generate Lifted Nodes (Fibers)
            for(let g=0; g<zn; g++) {
                liftedData.nodes.push({
                    id: `${v}_${g}`,
                    baseNodeId: v,
                    name: `Node ${v}, Group ${g}`,
                    color: getColor(v, numVertices)
                });
            }
        }

        // Generate Edges
        edges.forEach((edge, idx) => {
            let val = customVoltages.length > 0 ? customVoltages[idx % customVoltages.length] : Math.floor(Math.random() * zn);
            let edgeId = `e_${idx}`;

            // Add Base Edge
            baseData.links.push({
                id: edgeId,
                source: edge.source,
                target: edge.target,
                voltage: val
            });

            // Add Lifted Edges
            for(let g=0; g<zn; g++) {
                let targetG = (g + val) % zn;
                if (targetG < 0) targetG += zn;
                
                liftedData.links.push({
                    source: `${edge.source}_${g}`,
                    target: `${edge.target}_${targetG}`,
                    baseEdgeId: edgeId,
                    voltage: val
                });
            }
        });

        baseGraphInstance.graphData(baseData);
        liftedGraphInstance.graphData(liftedData);
        
        // Slightly different physics for base graph since it's smaller
        baseGraphInstance.d3Force('charge').strength(-400);
        baseGraphInstance.d3Force('link').distance(80);
    }

    [baseTypeInput, baseSizeInput, groupSizeInput, voltagesInput].forEach(el => {
        el.addEventListener('input', generateGraphs);
    });

    window.addEventListener('resize', () => {
        baseGraphInstance.width(baseCanvas.clientWidth).height(baseCanvas.clientHeight);
        liftedGraphInstance.width(liftedCanvas.clientWidth).height(liftedCanvas.clientHeight);
    });

    // Init
    generateGraphs();
    setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
});
