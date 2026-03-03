import { useEffect } from "react";

import hljs from "highlight.js";

export const useHighlight = () => {
  useEffect(() => {
    hljs.highlightAll();
  }, []);
};
