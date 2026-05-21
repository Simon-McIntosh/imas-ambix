// state.js — read project state as a repo-tracked static JSON file.
//
// The plan inventory and synthesis state live in
//     docs/state/<project>/<doc>.json
// committed to the repo and served as a static asset by both the docs-server
// (local dev) and GitHub Pages (publish). No HTTP /state endpoint is needed
// for reads.
//
// Project key resolution:
//   1. <meta name="docs-project" content="..."> on the page (preferred)
//   2. First URL path segment (defaults to the docs-server mount name)
// This lets the project key stay stable even when the repo name on Pages
// differs from the project key inside the JSON.
//
// SCHEMA (additive — old files keep working unchanged):
//   {
//     status:    "active" | "pending" | "blocked" | "shipped" | "draft",
//     tier:      "haiku" | "sonnet" | "opus",
//     decisions: { <key>: { choice, rationale, when, by } },
//     notes:     [{ id, who, bot, when, body, quote? }],
//     followups: [{ id, written_by, written_at, title, body,
//                   recommends_skill, touches, blocked_by?, tier?,
//                   est_turn, prompt,
//                   resolved_at?, resolved_by?, outcome? }],
//     research:  [{ id, type, title, source, added_by, when, url }],
//     questions: [{ id, section, body, opened_by, opened_at, resolved_at? }],
//     tests:     [{ name, pass, fail, pulse, fail_now? }]
//   }
//
// Old format (decision keys at top level of `data`) is still read by
// `getDecisions()` — see compat shim there.
//
// Read API:
//   loadIndexState()    → fetches state/<project>/index.json
//   loadState()         → fetches state/<project>/<current-doc>.json
//   loadProjectsState() → fetches /_projects/index.json (cross-project rollup,
//                         only available from the docs-server root)
//
// Writes go to localStorage only — decisions are browser-local until promoted
// to the repo by editing the state JSON and committing. Agents promote
// directly via filesystem writes; this script never POSTs to the server.
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

  // Prefer explicit meta tag; fall back to URL segment.
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

  // Cross-project rollup. Served by the docs-server at /_projects/index.json.
  // Available only when docs-server is running (not on GitHub Pages).
  window.loadProjectsState = async function loadProjectsState() {
    try {
      const r = await fetch(origin + "/_projects/index.json", { cache: "no-store" });
      if (!r.ok) return { projects: [] };
      return await r.json();
    } catch (e) {
      console.warn("loadProjectsState failed", e);
      return { projects: [] };
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

  // --- Schema-aware helpers (additive; safe with old format) -------
  //
  // All helpers operate on the localStorage overlay. The overlay is merged
  // over the canonical repo JSON on read by the page's own hydration code.

  // Return the decision map regardless of file shape.
  //   New shape: data.decisions = { <key>: {choice,...} }
  //   Old shape: data = { <key>: {choice,...}, ... }
  // If both are present, new wins for keys it covers; old fills the rest.
  window.getDecisions = function getDecisions(blob) {
    const data = (blob && blob.data) || blob || {};
    const out = {};
    // Old-format pass: any top-level key with a `choice` property.
    for (const [k, v] of Object.entries(data)) {
      if (v && typeof v === "object" && "choice" in v) out[k] = v;
    }
    // New-format pass overrides.
    if (data.decisions && typeof data.decisions === "object") {
      for (const [k, v] of Object.entries(data.decisions)) out[k] = v;
    }
    return out;
  };

  window.getFollowups = (blob) => ((blob && blob.data) || blob || {}).followups || [];
  window.getNotes     = (blob) => ((blob && blob.data) || blob || {}).notes     || [];
  window.getResearch  = (blob) => ((blob && blob.data) || blob || {}).research  || [];
  window.getQuestions = (blob) => ((blob && blob.data) || blob || {}).questions || [];
  window.getTests     = (blob) => ((blob && blob.data) || blob || {}).tests     || [];
  window.getStatus    = (blob) => ((blob && blob.data) || blob || {}).status    || null;
  window.getTier      = (blob) => ((blob && blob.data) || blob || {}).tier      || null;

  // Merge an updated data object with the current localStorage overlay and
  // persist. Returns the merged data so callers can re-render.
  async function mergeAndSave(patch) {
    const cur = window.loadLocalOverlay();
    const data = { ...(cur.data || {}), ...patch };
    await window.saveState(data);
    return data;
  }

  // Lock a decision into the new-format `decisions` map.
  window.lockDecision = async function lockDecision(key, choice, rationale) {
    if (mode !== "editable") throw new Error("read-only mode");
    const cur = (window.loadLocalOverlay().data) || {};
    const decisions = { ...(cur.decisions || {}) };
    decisions[key] = {
      choice,
      rationale: rationale || "",
      when: new Date().toISOString().slice(0, 16).replace("T", " "),
      by: cur.by || "smc",
    };
    return mergeAndSave({ decisions });
  };

  // Append a new entry to one of the array fields. `entry` is shallow-merged
  // onto a generated `{id, when, …}` envelope unless `id` is supplied.
  function appender(field) {
    return async function (entry) {
      if (mode !== "editable") throw new Error("read-only mode");
      const cur = (window.loadLocalOverlay().data) || {};
      const arr = Array.isArray(cur[field]) ? cur[field].slice() : [];
      const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
      const id = entry.id || `${field.slice(0, 1)}-${Date.now().toString(36)}`;
      arr.push({ id, when: stamp, ...entry });
      return mergeAndSave({ [field]: arr });
    };
  }
  window.appendNote     = appender("notes");
  window.appendFollowup = appender("followups");
  window.appendResearch = appender("research");
  window.appendQuestion = appender("questions");

  // Mark a followup as resolved.
  window.resolveFollowup = async function resolveFollowup(id, outcome, resolvedBy) {
    if (mode !== "editable") throw new Error("read-only mode");
    const cur = (window.loadLocalOverlay().data) || {};
    const arr = Array.isArray(cur.followups) ? cur.followups.slice() : [];
    const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
    const idx = arr.findIndex((f) => f.id === id);
    if (idx === -1) throw new Error("followup not found: " + id);
    arr[idx] = {
      ...arr[idx],
      resolved_at: stamp,
      resolved_by: resolvedBy || "smc",
      outcome: outcome || "",
    };
    return mergeAndSave({ followups: arr });
  };

  // Top-level scalar setters.
  window.setStatus = (s) => mergeAndSave({ status: s });
  window.setTier   = (t) => mergeAndSave({ tier:   t });

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
        "relevant <code>docs/state/" +
        project +
        "/&lt;doc&gt;.json</code> (or HTML), and open a PR.";
    } else {
      banner.innerHTML =
        "<strong>Local docs-server.</strong> Decisions you click are saved to <code>localStorage</code> only — " +
        "not written to the repo. To make a decision permanent, edit " +
        "<code>docs/state/" +
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
