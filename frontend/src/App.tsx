import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import { SiteGraphNav } from "./components/layout/SiteGraphNav";
import { DocsLayout } from "./components/docs/DocsLayout";
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

// Layout for full-screen apps (canvas + nav overlay)
const AppLayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  return (
    <>
      {children}
    </>
  );
};

// Component to handle routes with layout
const AppRoutes = () => {
  const location = useLocation();
  const features = useFeatureFlags();

  return (
    <>
      <SiteGraphNav />
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<DocsLayout><OverviewPage /></DocsLayout>} />
          
          {/* Docs Routes */}
          <Route path="/docs/gnns" element={<DocsLayout><DocsGnnsPage /></DocsLayout>} />
          <Route path="/docs/architecture" element={<DocsLayout><DocsArchitecturePage /></DocsLayout>} />
          <Route path="/docs/module-degree" element={<DocsLayout><DocsModuleDegreePage /></DocsLayout>} />
          <Route path="/docs/module-min-cycle" element={<DocsLayout><DocsModuleMinCyclePage /></DocsLayout>} />
          <Route path="/docs/module-assessment" element={<DocsLayout><DocsModuleAssessmentPage /></DocsLayout>} />
          <Route path="/docs/module-cage" element={<DocsLayout><DocsModuleCagePage /></DocsLayout>} />
          <Route path="/docs/training" element={<DocsLayout><DocsTrainingPage /></DocsLayout>} />

          {/* Apps Routes */}
          <Route path="/degree" element={<AppLayoutWrapper><PredictionPage task="degree" /></AppLayoutWrapper>} />
          <Route path="/min_cycle" element={<AppLayoutWrapper><PredictionPage task="min_cycle" /></AppLayoutWrapper>} />
          <Route path="/cage" element={<AppLayoutWrapper><CagePage /></AppLayoutWrapper>} />
          
          {/* Fallback/404 handling could be added here */}
        </Routes>
      </AnimatePresence>
    </>
  );
};

export const App = () => {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
};
