import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import { SiteGraphNav } from "./components/layout/SiteGraphNav";
import { OverviewPage } from "./pages/OverviewPage";
import { DocsGnnsPage } from "./pages/docs/DocsGnnsPage";
import { DocsArchitecturePage } from "./pages/docs/DocsArchitecturePage";
import { DocsModuleDegreePage } from "./pages/docs/DocsModuleDegreePage";
import { DocsModuleMinCyclePage } from "./pages/docs/DocsModuleMinCyclePage";
import { DocsModuleAssessmentPage } from "./pages/docs/DocsModuleAssessmentPage";
import { DocsModuleCagePage } from "./pages/docs/DocsModuleCagePage";
import { DocsTrainingPage } from "./pages/docs/DocsTrainingPage";
import { PredictionPage } from "./pages/apps/PredictionPage";
import { CagePage } from "./pages/apps/CagePage";
import { useFeatureFlags } from "./hooks/useFeatureFlags";

// Layout for documentation pages (centered column + nav)
const DocsLayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  
  return (
    <div className="h-dvh overflow-y-auto bg-bg1">
      <main className="page mx-auto w-full max-w-[920px] p-10 pb-[60px] pt-12 max-[760px]:w-full max-[760px]:p-3 max-[760px]:pt-3">
        {children}
      </main>
      <SiteGraphNav currentPath={location.pathname} />
    </div>
  );
};

// Layout for full-screen apps (canvas + nav overlay)
const AppLayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();

  return (
    <>
      {children}
      <SiteGraphNav currentPath={location.pathname} />
    </>
  );
};

// Component to handle routes with layout
const AppRoutes = () => {
  const location = useLocation();
  const features = useFeatureFlags();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<DocsLayoutWrapper><OverviewPage /></DocsLayoutWrapper>} />
        
        {/* Docs Routes */}
        <Route path="/docs/gnns" element={<DocsLayoutWrapper><DocsGnnsPage /></DocsLayoutWrapper>} />
        <Route path="/docs/architecture" element={<DocsLayoutWrapper><DocsArchitecturePage /></DocsLayoutWrapper>} />
        <Route path="/docs/module-degree" element={<DocsLayoutWrapper><DocsModuleDegreePage /></DocsLayoutWrapper>} />
        <Route path="/docs/module-min-cycle" element={<DocsLayoutWrapper><DocsModuleMinCyclePage /></DocsLayoutWrapper>} />
        <Route path="/docs/module-assessment" element={<DocsLayoutWrapper><DocsModuleAssessmentPage /></DocsLayoutWrapper>} />
        <Route path="/docs/module-cage" element={<DocsLayoutWrapper><DocsModuleCagePage /></DocsLayoutWrapper>} />
        <Route path="/docs/training" element={<DocsLayoutWrapper><DocsTrainingPage /></DocsLayoutWrapper>} />

        {/* Apps Routes */}
        <Route path="/degree" element={<AppLayoutWrapper><PredictionPage task="degree" /></AppLayoutWrapper>} />
        <Route path="/min_cycle" element={<AppLayoutWrapper><PredictionPage task="min_cycle" /></AppLayoutWrapper>} />
        <Route path="/cage" element={<AppLayoutWrapper><CagePage /></AppLayoutWrapper>} />
        
        {/* Fallback/404 handling could be added here */}
      </Routes>
    </AnimatePresence>
  );
};

export const App = () => {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
};
