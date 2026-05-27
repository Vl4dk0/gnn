import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import { BackgroundGraphTexture } from "./components/layout/BackgroundGraphTexture";
import { SiteGraphNav } from "./components/layout/SiteGraphNav";
import { ThemeToggle } from "./components/ui/ThemeToggle";
import { OverviewPage } from "./pages/OverviewPage";
import { DocsGnnsPage } from "./pages/docs/DocsGnnsPage";
import { DocsArchitecturePage } from "./pages/docs/DocsArchitecturePage";
import { DocsModuleDegreePage } from "./pages/docs/DocsModuleDegreePage";
import { DocsModuleMinCyclePage } from "./pages/docs/DocsModuleMinCyclePage";
import { DocsModuleAssessmentPage } from "./pages/docs/DocsModuleAssessmentPage";
import { DocsModuleCagePage } from "./pages/docs/DocsModuleCagePage";
import { DocsVoltagePage } from "./pages/docs/DocsVoltagePage";
import { DocsCayleyPage } from "./pages/docs/DocsCayleyPage";
import { PredictionPage } from "./pages/apps/PredictionPage";
import { CagePage } from "./pages/apps/CagePage";

// Layout for full-screen apps (canvas + nav overlay)
const AppLayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  return <>{children}</>;
};

// Component to handle routes with layout
const AppRoutes = () => {
  const location = useLocation();

  return (
    <>
      <BackgroundGraphTexture />
      <SiteGraphNav />
      <ThemeToggle />
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<OverviewPage />} />

          {/* Docs Routes */}
          <Route path="/docs/gnns" element={<DocsGnnsPage />} />
          <Route path="/docs/architecture" element={<DocsArchitecturePage />} />
          <Route path="/docs/module-degree" element={<DocsModuleDegreePage />} />
          <Route path="/docs/module-min-cycle" element={<DocsModuleMinCyclePage />} />
          <Route path="/docs/module-assessment" element={<DocsModuleAssessmentPage />} />
          <Route path="/docs/module-cage" element={<DocsModuleCagePage />} />
          <Route path="/docs/voltage" element={<DocsVoltagePage />} />
          <Route path="/docs/cayley" element={<DocsCayleyPage />} />

          {/* Apps Routes */}
          <Route
            path="/degree"
            element={
              <AppLayoutWrapper>
                <PredictionPage task="degree" />
              </AppLayoutWrapper>
            }
          />
          <Route
            path="/min_cycle"
            element={
              <AppLayoutWrapper>
                <PredictionPage task="min_cycle" />
              </AppLayoutWrapper>
            }
          />
          <Route
            path="/cage"
            element={
              <AppLayoutWrapper>
                <CagePage />
              </AppLayoutWrapper>
            }
          />

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
