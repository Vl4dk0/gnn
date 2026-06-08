import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import { BackgroundGraphTexture } from "./components/layout/BackgroundGraphTexture";
import { SiteGraphNav } from "./components/layout/SiteGraphNav";
import { ThemeToggle } from "./components/ui/ThemeToggle";
import { OverviewPage } from "./pages/OverviewPage";
import { DegreeTaskPage } from "./pages/topics/DegreeTaskPage";
import { MinCycleTaskPage } from "./pages/topics/MinCycleTaskPage";
import { CageAstarPage } from "./pages/topics/CageAstarPage";
import { CageRlPage } from "./pages/topics/CageRlPage";
import { CageVoltagePage } from "./pages/topics/CageVoltagePage";
import { RefinementPage } from "./pages/topics/RefinementPage";
import { ExcisionTopicPage } from "./pages/topics/ExcisionPage";
import { ForgePage } from "./pages/topics/ForgePage";
import { PredictionPage } from "./pages/apps/PredictionPage";
import { CagePage } from "./pages/apps/CagePage";
import { VoltageLiftPage } from "./pages/apps/VoltageLiftPage";
import { ExcisionPage } from "./pages/apps/ExcisionPage";
import { RefinePage } from "./pages/apps/RefinePage";

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

          {/* Topic / explainer pages */}
          <Route path="/degree-task" element={<DegreeTaskPage />} />
          <Route path="/min-cycle-task" element={<MinCycleTaskPage />} />
          <Route path="/cage/astar" element={<CageAstarPage />} />
          <Route path="/cage/rl" element={<CageRlPage />} />
          <Route path="/cage/voltage" element={<CageVoltagePage />} />
          <Route path="/refinement" element={<RefinementPage />} />
          <Route path="/excision" element={<ExcisionTopicPage />} />
          <Route path="/forge" element={<ForgePage />} />

          {/* Editors */}
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
          <Route
            path="/lift"
            element={
              <AppLayoutWrapper>
                <VoltageLiftPage />
              </AppLayoutWrapper>
            }
          />
          <Route
            path="/excise"
            element={
              <AppLayoutWrapper>
                <ExcisionPage />
              </AppLayoutWrapper>
            }
          />
          <Route
            path="/refine"
            element={
              <AppLayoutWrapper>
                <RefinePage />
              </AppLayoutWrapper>
            }
          />
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
