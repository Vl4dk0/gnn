import { useEffect } from "react";
import { useLocation } from "react-router-dom";

import hljs from "highlight.js";

/**
 * Highlight every `pre code` block on the page with highlight.js.
 *
 * `hljs.highlightAll()` flags each element with `data-highlighted` and refuses
 * to touch it again, and it does not re-run on client-side route changes. In a
 * single-page app that means a freshly mounted page often renders unhighlighted.
 * Instead, on every route change we clear the flag and re-highlight each block
 * directly, so the code always gets coloured.
 */
export const useHighlight = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    const blocks = document.querySelectorAll<HTMLElement>("pre code");
    blocks.forEach((block) => {
      delete block.dataset.highlighted;
      block.classList.remove("hljs");
      hljs.highlightElement(block);
    });
  }, [pathname]);
};
