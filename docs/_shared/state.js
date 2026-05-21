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
//     _version:  <integer>,   // set by serve.py; incremented on every write
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
// Write API (local docs-server mode only):
//   saveState(data)     → POST with versioning; falls back to localStorage on
//                         network error or persistent conflict.
//   lockDecision / appendNote / appendFollowup / appendResearch /
//   appendQuestion / resolveFollowup / setStatus / setTier  — all call the
//   internal postState(patch) helper which manages GET-version → POST flow.
//
// Version / 412 contract (mirrors serve.py):
//   - On load, the returned data._version is cached in
//     window._stateVersions[<docId>].
//   - Every POST sends `If-Match: <version>` header.
//   - 200 response carries {ok, path, version} — cache updated.
//   - 412 response carries {current_version, current_data} — re-apply patch
//     over current_data and retry once. Second 412 → alert + localStorage only.
//   - Network error → localStorage fallback (today's behaviour).
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

  // Server POST URL (docs-server endpoint, no .json suffix).
  function statePostUrl(docName) {
    return origin + "/state/" + project + "/" + docName;
  }

  window.docMeta = { project, docId, siteRoot, origin, mode };
  window.docMode = mode;

  // Version cache: window._stateVersions[docId] = integer
  if (!window._stateVersions) window._stateVersions = {};

  // --- Reads ---------------------------------------------------------
  window.loadIndexState = async function loadIndexState() {
    try {
      const r = await fetch(stateUrl("index"), { cache: "no-store" });
      if (!r.ok) return {};
      const j = await r.json();
      // Cache version from loaded data.
      const data = (j && j.data) || j || {};
      if (typeof data._version === "number") {
        window._stateVersions["index"] = data._version;
      }
      return j;
    } catch (e) {
      console.warn("loadIndexState failed", e);
      return {};
    }
  };

  window.loadState = async function loadState() {
    try {
      const r = await fetch(stateUrl(docId), { cache: "no-store" });
      if (!r.ok) return {};
      const j = await r.json();
      // Cache version from loaded data.
      const data = (j && j.data) || j || {};
      if (typeof data._version === "number") {
        window._stateVersions[docId] = data._version;
      }
      return j;
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

  // --- Writes -------------------------------------------------------
  const lsKey = (doc) => project + ":" + doc;

  // Low-level localStorage write (always available as fallback).
  function writeLocalStorage(data) {
    try {
      localStorage.setItem(
        lsKey(docId),
        JSON.stringify({
          updated: new Date().toISOString(),
          data: data ?? {},
        })
      );
    } catch (e) {
      console.warn("localStorage write failed", e);
    }
  }

  // Fetch current version for docId. Uses cached value if available,
  // otherwise fetches from the server.
  async function getVersion() {
    if (typeof window._stateVersions[docId] === "number") {
      return window._stateVersions[docId];
    }
    try {
      const r = await fetch(stateUrl(docId), { cache: "no-store" });
      if (!r.ok) return 0;
      const j = await r.json();
      const data = (j && j.data) || j || {};
      const v = typeof data._version === "number" ? data._version : 0;
      window._stateVersions[docId] = v;
      return v;
    } catch {
      return 0;
    }
  }

  // Internal POST helper — version-aware. Returns {ok, version} on success.
  // On 412 (first): re-applies patch over server's current_data and retries.
  // On second 412 or network error: falls back to localStorage.
  async function postState(data) {
    const version = await getVersion();
    // Strip any _version the caller may have included — server owns it.
    const body = Object.assign({}, data);
    delete body._version;

    try {
      const r = await fetch(statePostUrl(docId), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "If-Match": String(version),
        },
        body: JSON.stringify(body),
      });

      if (r.ok) {
        const j = await r.json().catch(() => ({}));
        if (typeof j.version === "number") {
          window._stateVersions[docId] = j.version;
        }
        writeLocalStorage(body);
        return { ok: true, storage: "docs-server", version: j.version };
      }

      if (r.status === 412) {
        // First 412 — server gave us current state. Re-apply patch and retry.
        const conflict = await r.json().catch(() => ({}));
        const curData = conflict.current_data || {};
        const curVersion = typeof conflict.current_version === "number"
          ? conflict.current_version : 0;

        // Merge: current server data wins on structure; patch wins on keys.
        const retryBody = Object.assign({}, curData, body);
        delete retryBody._version;

        const r2 = await fetch(statePostUrl(docId), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "If-Match": String(curVersion),
          },
          body: JSON.stringify(retryBody),
        });

        if (r2.ok) {
          const j2 = await r2.json().catch(() => ({}));
          if (typeof j2.version === "number") {
            window._stateVersions[docId] = j2.version;
          }
          writeLocalStorage(retryBody);
          return { ok: true, storage: "docs-server (retry)", version: j2.version };
        }

        // Second 412 — genuine conflict. Alert and fall back.
        alert(
          "Conflict detected — another session has updated this page since you loaded it.\n" +
          "Please refresh the page and retry your change."
        );
        writeLocalStorage(body);
        return { ok: false, storage: "localStorage (conflict)", version };
      }

      console.warn("postState: unexpected status", r.status);
    } catch (e) {
      console.warn("postState: network error (docs-server unreachable?)", e);
    }

    // Network error fallback.
    writeLocalStorage(body);
    return { ok: false, storage: "localStorage (network error)", version };
  }

  window.saveState = async function saveState(data) {
    if (mode !== "editable") {
      throw new Error(
        "read-only: this site is published; edit the repo state JSON to record a decision"
      );
    }
    return postState(data);
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
  // In editable mode, helpers POST through to the docs-server via postState.
  // localStorage is written as a cache + fallback path.

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

  // Build a merged data object from the localStorage overlay, then POST it.
  // In editable mode, postState handles server write + local cache.
  // In readonly mode, only the local cache is updated (via the old path).
  async function mergeAndSave(patch) {
    if (mode !== "editable") {
      const cur = window.loadLocalOverlay();
      const data = { ...(cur.data || {}), ...patch };
      writeLocalStorage(data);
      return data;
    }
    // Build merged from localStorage overlay (most recent known state).
    const cur = window.loadLocalOverlay();
    const data = { ...(cur.data || {}), ...patch };
    await postState(data);
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
      by: cur.by || "Simon McIntosh",
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
      resolved_by: resolvedBy || "Simon McIntosh",
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
        "<strong>Local docs-server.</strong> Decisions you click are written through to the " +
        "<code>docs/state/" +
        project +
        "/&lt;doc&gt;.json</code> file via the docs-server (versioned POST). " +
        "Changes are immediately visible to other sessions and committed to git on next push.";
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
