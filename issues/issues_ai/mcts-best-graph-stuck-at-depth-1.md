# mcts-best-graph-stuck-at-depth-1

**Súbor:** `ai/cage/functions/monte_carlo_search_tree.py:169-177`  
**Závažnosť:** HIGH

## Popis

V metóde `MCTSGenerator.step` sa najlepší nájdený graf (`self.graph`) a kontrola úspešnosti vyhľadávania (`self.success`) aktualizujú iba z priamych synov koreňa (`self.root.best_child(c_param=0)`):

```python
        # Update best graph found so far (from root's best child)
        best_child = self.root.best_child(c_param=0)  # Exploitation only
        if best_child:
            self.graph = best_child.graph

            # Check success
            if is_k_regular(self.graph, self.k):
                if compute_girth(self.graph) == self.g:
                    self.is_complete = True
                    self.success = True
```

Keďže priame deti koreňového uzla sú v hĺbke 1 a vznikli pridaním najviac jednej hrany k prázdnemu počiatočnému grafu, majú stupeň najviac 1. Pre ľubovoľné cieľové $k \ge 3$ takýto graf nikdy nebude $k$-regulárny.

## Dôsledok

MCTS generátor nemôže nikdy úspešne dokončiť vyhľadávanie (`self.success = True`), pretože kontroluje iba uzly v hĺbke 1, ktoré nikdy nie sú regulárne. Aj keď vyhľadávanie (v hĺbke stromu alebo počas náhodných simulácií/rolloutov) nájde kompletný valídny $(k,g)$-graf, algoritmus si to nevšimne a bude pokračovať donekonečna alebo kým nevyprší časový limit.

## Oprava

Uchovávať najlepší nájdený graf globálne počas generovania. Zakaždým, keď sa v rámci `expand` (vytvorenie uzla) alebo `simulate` (rollout) narazí na valídny $(k,g)$-graf, aktualizovať `self.graph` a označiť úspech:

```python
    def step(self) -> None:
        if self.start_time == 0:
            self.start_time = time.time()
        self.step_count += 1

        # 1. Selection
        node = self.select_node()

        # 2. Expansion
        if not node.is_terminal and not node.is_fully_expanded():
            node = self.expand(node)
            # Skontrolovať či novovytvorený uzol nie je riešením
            if is_k_regular(node.graph, self.k) and compute_girth(node.graph) == self.g:
                self.graph = node.graph
                self.is_complete = True
                self.success = True
                return

        # 3. Simulation (upraviť simulate tak, aby vrátila aj výsledný graf)
        score, simulated_graph = self.simulate_and_get_graph(node)
        if is_k_regular(simulated_graph, self.k) and compute_girth(simulated_graph) == self.g:
            self.graph = simulated_graph
            self.is_complete = True
            self.success = True
            return

        # 4. Backpropagation
        self.backpropagate(node, score)
```
