// state.js — read project state as a repo-tracked static JSON file.
//
// The plan inventory and synthesis state live in
//     plans/state/<project>/<doc>.json
// committed to the repo and served as a static asset by both the docs-server
// (local dev) and GitHub Pages (publish). No HTTP /state endpoint is needed.
//
// Project key resolution:
//   1. <meta name="docs-project" content="..."> on the page (preferred)
//   2. First URL path segment (defaults to the docs-server mount name)
// This lets the project key stay stable even when the repo name on Pages
// differs from the project key inside the JSON (e.g. repo=efit, project=imas-efit).
//
// Read API:
//   loadIndexState() → fetches state/<project>/index.json
//   loadState()      → fetches state/<project>/<current-doc>.json
//
// Writes go to localStorage only — decisions are browser-local until promoted
// to the repo by editing the state JSON and committing.
//
// Mode detection: localhost/127.0.0.1 = editable; anything else (Pages) =
// readonly. window.docMode reflects this; saveState refuses writes when
// readonly.

(function () {
  const pathParts = window.location.pathname.replace(/^\/+/, "").split("/");
  const siteSegment = pathParts[0] || "unknown";
  const siteRoot = "/" + siteSegment + "/";

  const lastSeg = pathParts[pathParts.length - 1] || "index.html";
  const fileSegment = lastSeg.replace(/\.html?$/, "");
  const docId = fileSegment || "index";

  // Prefer explicit meta tag; fall back to URL segment (works when repo name
  // matches the project key, as in imas-ambix).
  const projectMeta = document
    .querySelector('meta[name="docs-project"]')
    ?.getAttribute("content");
  const project = projectMeta || siteSegment;

  const baseFromMeta = document
    .querySelector('meta[name="docs-server"]')
    ?.getAttribute("content");
  const origin = baseFromMeta || window.location.origin;

  const localHosts = new Set(["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]);
  const isLocal = localHosts.has(window.location.hostname);
  const mode = isLocal ? "editable" : "readonly";

  function stateUrl(docName) {
    return origin + siteRoot + "state/" + project + "/" + docName + ".json";
  }

  window.docMeta = { project, docId, siteRoot, origin, mode };
  window.docMode = mode;

  // --- Reads ---------------------------------------------------------
  window.loadIndexState = async function loadIndexState() {
    try {
      const r = await fetch(stateUrl("index"), { cache: "no-store" });
      if (!r.ok) return {};
      return await r.json();
    } catch (e) {
      console.warn("loadIndexState failed", e);
      return {};
    }
  };

  window.loadState = async function loadState() {
    try {
      const r = await fetch(stateUrl(docId), { cache: "no-store" });
      if (!r.ok) return {};
      return await r.json();
    } catch (e) {
      console.warn("loadState failed", e);
      return {};
    }
  };

  // --- Writes (localStorage only) -----------------------------------
  const lsKey = (doc) => project + ":" + doc;

  window.saveState = async function saveState(data) {
    if (mode !== "editable") {
      throw new Error(
        "read-only: this site is published; edit the repo state JSON to record a decision"
      );
    }
    try {
      localStorage.setItem(
        lsKey(docId),
        JSON.stringify({
          updated: new Date().toISOString(),
          data: data ?? {},
        })
      );
      return { ok: true, storage: "localStorage" };
    } catch (e) {
      throw new Error("localStorage write failed: " + e.message);
    }
  };

  window.loadLocalOverlay = function loadLocalOverlay() {
    try {
      const raw = localStorage.getItem(lsKey(docId));
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  };

  window.copyState = async function copyState(obj) {
    const text = JSON.stringify(obj ?? {}, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      return true;
    }
  };

  // --- Mode banner --------------------------------------------------
  document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("mode-banner")) return;
    const banner = document.createElement("div");
    banner.id = "mode-banner";
    banner.className = "mode-banner mode-" + mode;
    if (mode === "readonly") {
      banner.innerHTML =
        '<strong>Read-only.</strong> Published via GitHub Pages — viewing the committed plan record. ' +
        "Decision-capture buttons are disabled. To edit a plan, clone the repo, modify the " +
        "relevant <code>plans/state/" +
        project +
        "/&lt;doc&gt;.json</code> (or HTML), and open a PR.";
    } else {
      banner.innerHTML =
        "<strong>Local docs-server.</strong> Decisions you click are saved to <code>localStorage</code> only — " +
        "not written to the repo. To make a decision permanent, edit " +
        "<code>plans/state/" +
        project +
        "/&lt;doc&gt;.json</code> and commit.";
    }
    document.body.insertBefore(banner, document.body.firstChild);

    if (mode === "readonly") {
      document.querySelectorAll("button[data-choice]").forEach((b) => {
        b.setAttribute("disabled", "disabled");
        b.title = "Read-only — open the repo to record a decision";
      });
    }
  });
})();
