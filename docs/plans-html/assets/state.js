// state.js — tiny client for docs-server's /state/<project>/<doc> endpoints.
//
// Usage:
//   <script src="assets/state.js" defer></script>
//   <script defer>
//     loadState().then((s) => {
//       // s.data is the most-recent payload you POSTed; s itself includes
//       // {updated, project, doc, data}. Empty object {} if nothing yet.
//       applyState(s.data || {});
//     });
//     // when user changes something:
//     saveState({ decision: 'A', rationale: 'cheaper to ship' });
//   </script>

(function () {
  const path = window.location.pathname.replace(/^\/+/, "");
  const [project = "unknown", ...rest] = path.split("/");
  const file = (rest.pop() || "index.html").replace(/\.html?$/, "");
  const docId = file || "index";

  const baseFromMeta = document
    .querySelector('meta[name="docs-server"]')
    ?.getAttribute("content");
  const base = baseFromMeta || `${window.location.origin}`;

  window.docMeta = { project, docId, base };

  window.loadState = async function loadState() {
    try {
      const r = await fetch(`${base}/state/${project}/${docId}`, {
        cache: "no-store",
      });
      if (!r.ok) return {};
      return await r.json();
    } catch (e) {
      console.warn("loadState failed", e);
      return {};
    }
  };

  window.saveState = async function saveState(data) {
    const r = await fetch(`${base}/state/${project}/${docId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data ?? {}),
    });
    if (!r.ok) throw new Error(`saveState ${r.status}`);
    return await r.json();
  };

  // Convenience: copy the current state object as JSON to the clipboard.
  window.copyState = async function copyState(obj) {
    const text = JSON.stringify(obj ?? {}, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fallback for file:// or no-permission contexts.
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      return true;
    }
  };
})();
