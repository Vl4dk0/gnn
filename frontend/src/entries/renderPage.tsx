import { createRoot } from "react-dom/client";

import "../styles.css";

export const renderPage = (element: React.ReactNode): void => {
  const rootElement = document.getElementById("root");

  if (!rootElement) {
    throw new Error("Missing #root mount element");
  }

  createRoot(rootElement).render(element);
};
