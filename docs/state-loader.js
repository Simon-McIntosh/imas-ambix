// state-loader.js — live state fetcher.
//
// Fetches state/<project>/index.json and per-plan state/<project>/<slug>.json
// to build window.STATE at runtime. The state JSON files are the canonical
// source; this file is just the bridge so the existing templates don't have
// to change their data-shape assumptions.
//
// window.STATE_READY is a Promise that resolves once window.STATE is populated.
// Templates wait on it before rendering (see ReadyGate in each template).

window.STATE_READY = (async function () {
  const PROJECT = (document.querySelector('meta[name="docs-project"]')?.content) ||
                  window.location.pathname.replace(/^\/+/, "").split("/")[0] ||
                  "unknown";

  async function getJson(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return null;
      return await r.json();
    } catch { return null; }
  }

  const stateBase = `state/${PROJECT}`;

  // 1. Central index
  const idxBlob = await getJson(`${stateBase}/index.json`);
  const idx = (idxBlob && idxBlob.data) || {};

  // 2. Per-plan bodies — fetch in parallel
  const inventory = Array.isArray(idx.inventory) ? idx.inventory : [];
  const planEntries = await Promise.all(
    inventory.map(async inv => {
      const blob = await getJson(`${stateBase}/${encodeURIComponent(inv.slug)}.json`);
      const data = (blob && blob.data) || {};

      // Materialise decisions[] in the array form the templates expect
      const defs = Array.isArray(data.decisions_def) ? data.decisions_def
                 : Array.isArray(data.decisions)     ? data.decisions
                 : [];
      const lockedMap = (data.decisions && !Array.isArray(data.decisions)) ? data.decisions : {};
      const decisions = defs.map(d => {
        const l = lockedMap[d.key] || {};
        return {
          ...d,
          chosen:    l.choice    !== undefined ? l.choice    : (d.chosen    || ""),
          rationale: l.rationale !== undefined ? l.rationale : (d.rationale || ""),
          when:      l.when      !== undefined ? l.when      : (d.when      || ""),
          by:        l.by        !== undefined ? l.by        : (d.by        || ""),
        };
      });

      return [inv.slug, { ...inv, ...data, decisions }];
    })
  );
  const plans = Object.fromEntries(planEntries);

  // 3. Resolve active sprint
  const sprints = Array.isArray(idx.sprints) ? idx.sprints : [];
  const activeSprint =
    sprints.find(s => s.id === idx.active_sprint_id) ||
    sprints.find(s => s.status === "active");

  // 4. Populate window.STATE
  // individual plan JSONs are the per-plan source of truth for status, impl,
  // dec_open, etc. Use the merged `plans` data for inventory so UI reads
  // never need to look in two places. index.json inventory values serve only
  // as a fallback when no individual plan JSON exists.
  const mergedInventory = inventory.map(inv => plans[inv.slug] || inv);

  window.STATE = {
    today:            new Date().toISOString().slice(0, 10),
    projects:         Array.isArray(idx.projects) ? idx.projects : [],
    inventory:        mergedInventory,
    active_sprint_id: idx.active_sprint_id || null,
    sprints,
    sprint:           activeSprint,
    blockers:         Array.isArray(idx.blockers) ? idx.blockers : [],
    timeline:         Array.isArray(idx.timeline) ? idx.timeline : [],
    plans,
    planTokenizers:   plans["tokenizers"] || null,
  };
})();
