# Frontend Remake Task List

## Planning & Prep
- [x] Create implementation plan and obtain user approval <!-- id: 0 -->
- [x] Establish layout of pages and verify code snippets <!-- id: 1 -->

## Route and Navigation Changes
- [x] Remove DIY (`/docs/training`) page from routes in `frontend/src/App.tsx` and delete the file `DocsTrainingPage.tsx` <!-- id: 2 -->
- [x] Add Cayley Graphs (`/docs/cayley`) page to routes in `frontend/src/App.tsx` <!-- id: 3 -->
- [x] Update D3 network navigation in `frontend/src/components/layout/SiteGraphNav.tsx` (remove DIY, add Cayley) <!-- id: 4 -->
- [x] Update `frontend/src/pages/OverviewPage.tsx` cards and navigation flow <!-- id: 5 -->

## Docs Pages Upgrades (Deep Implementation Focus)
- [x] **Chapter 1: GNNs** (`frontend/src/pages/docs/DocsGnnsPage.tsx`): Add PyG `Data` structure explanation, message-passing logic, and a detailed section explaining how hyper-parameters (hidden dimension, layers, LR, dropout, weight decay) affect learning. <!-- id: 6 -->
- [x] **Chapter 2: Architectures** (`frontend/src/pages/docs/DocsArchitecturePage.tsx`): Inject actual Python implementation code snippets for GCN, GraphSAGE, GIN, GPS, and Loopy from the codebase. Explain code mechanics and trade-offs. <!-- id: 7 -->
- [x] **Chapter 3: Degree Prediction** (`frontend/src/pages/docs/DocsModuleDegreePage.tsx`): Expand details on training loop structure, loss functions, random node feature representation, and evaluation metrics. <!-- id: 8 -->
- [x] **Chapter 4: Min-Cycle Prediction** (`frontend/src/pages/docs/DocsModuleMinCyclePage.tsx`): Insert NetworkX `get_min_cycle` algorithm, explain GIN's shortcut/cheating behavior, and details of Loopy's recovery with hidden capacity. <!-- id: 9 -->
- [x] **Chapter 5: Assessment** (`frontend/src/pages/docs/DocsModuleAssessmentPage.tsx`): Elaborate on model capacity, hyperparameter training bounds, and empirical findings. <!-- id: 10 -->
- [x] **Chapter 6: Cage Generation** (`frontend/src/pages/docs/DocsModuleCagePage.tsx`): Show PPO step environment loop, action masking for girth/regularity, and potential-based reward shaping formulas. <!-- id: 11 -->
- [x] **Chapter 7: Voltage Lifts** (`frontend/src/pages/docs/DocsVoltagePage.tsx`): Document `build_lift` construction code, group lookup tables representation, and net voltage cycle check loops. <!-- id: 12 -->

## New Cayley Graphs Page
- [x] **Chapter 8: Cayley Graphs** (`frontend/src/pages/docs/DocsCayleyPage.tsx`): Create a page with definitions, `build_cayley` construction code, BFS non-backtracking girth search code, available groups catalog with matrix families, and GNN group-promise predictor information. <!-- id: 13 -->

## Verification
- [x] Build and run frontend locally to verify visual rendering and navigation flow <!-- id: 14 -->
- [x] Ensure all D3 navigation features work perfectly without console errors <!-- id: 15 -->
- [x] Run typescript checks and prettier if applicable <!-- id: 16 -->
