import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./index.css"; // Ensure global styles are imported
import "highlight.js/styles/atom-one-dark.css"; // Syntax highlighting

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
