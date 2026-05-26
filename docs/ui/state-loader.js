// state-loader.js — runtime state fetcher for reckon plan pages.
//
// Builds window.STATE from three sources, in priority order:
//   1. state/<project>/index.json  — per-project central index (if present)
//   2. state/<project>/<slug>.json — per-plan state files (per-doc layout)
//   3. /_discover/<project>        — auto-discovery from HTML meta tags
//
// window.STATE_READY is a Promise. Templates wait on it before rendering.

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

  // ── 1. Central index ───────────────────────────────────────────────────
  const idxBlob = await getJson(`${stateBase}/index.json`);
  const idx = (idxBlob && idxBlob.data) || {};

  let sprints    = Array.isArray(idx.sprints)    ? idx.sprints    : [];
  let milestones = Array.isArray(idx.milestones) ? idx.milestones : [];
  let inventory  = Array.isArray(idx.inventory)  ? idx.inventory  : [];

  // ── 2. Central-index layout: data.plans[] (no data.inventory) ─────────
  // Handles repos that store plan metadata in a central index.json using
  // data.plans[] (with path fields) instead of data.inventory[] with slugs.
  if (inventory.length === 0 && Array.isArray(idx.plans) && idx.plans.length > 0) {
    const pathToSprint = {};
    for (const s of sprints) {
      for (const it of (s.items || [])) {
        const raw = typeof it === "string" ? it : (it.path || it.slug || "");
        const key = raw.replace(/^.*\//, "").replace(/\.[^.]+$/, "");
        if (key) pathToSprint[key] = s.id;
      }
    }
    inventory = idx.plans.map(pl => {
      const rawPath = pl.path || pl.slug || "";
      const slug = rawPath.replace(/^.*\//, "").replace(/\.[^.]+$/, "")
                   || (pl.title || "plan").toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 48);
      return {
        slug,
        title:    pl.title || slug,
        status:   pl.status || "pending",
        ms:       pl.milestone || "—",
        roi:      pl.roi    || "mid",
        effort:   pl.effort || "M",
        impl:     pl.implementation_fraction || 0,
        dec_open: pl.dec_open || 0,
        blockers: Array.isArray(pl.blocked_by) ? pl.blocked_by.length
                  : (typeof pl.blockers === "number" ? pl.blockers : 0),
        sprint:   pathToSprint[slug] || null,
        last:     pl.last_modified || "",
        summary:  pl.summary || "",
        category: pl.category || "",
        _central: true,
      };
    });
  }

  // ── 3. Auto-discovery fallback ─────────────────────────────────────────
  // If still no inventory, ask the server to scan HTML plan pages for
  // <meta name="plan-status"> tags. Zero boilerplate required in the repo.
  if (inventory.length === 0) {
    const disc = await getJson(`/_discover/${PROJECT}`);
    if (Array.isArray(disc?.inventory) && disc.inventory.length > 0) {
      inventory = disc.inventory;
      if (!sprints.length    && Array.isArray(disc.sprints))    sprints    = disc.sprints;
      if (!milestones.length && Array.isArray(disc.milestones)) milestones = disc.milestones;
    }
  }

  // ── 4. Per-plan state JSON (per-doc layout only) ───────────────────────
  // For _central plans the metadata lives in index.json; skip individual fetches.
  const planEntries = await Promise.all(
    inventory
      .filter(inv => !inv._central)
      .map(async inv => {
        const blob = await getJson(`${stateBase}/${encodeURIComponent(inv.slug)}.json`);
        const data = (blob && blob.data) || {};

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

  // ── 5. Resolve active sprint ───────────────────────────────────────────
  const activeSprint =
    sprints.find(s => s.id === idx.active_sprint_id) ||
    sprints.find(s => s.status === "active");

  // ── 6. Assemble window.STATE ───────────────────────────────────────────
  // _central items stay as-is; per-doc items are merged with their plan JSON.
  const mergedInventory = inventory.map(inv =>
    inv._central ? inv : (plans[inv.slug] || inv)
  );

  // Ensure projects[0] is populated. Some central-index repos (e.g. imas-efit)
  // have data.plans[] + data.counts + data.milestones at the top level and no
  // data.projects[]. Synthesise one so the SPA components can read uniformly.
  let projects = Array.isArray(idx.projects) ? idx.projects.slice() : [];
  if (projects.length === 0) {
    const status = (idx.counts && idx.counts.status) || {};
    const total  = (idx.counts && idx.counts.total) || mergedInventory.length;
    const count  = (s) => status[s] !== undefined
                          ? status[s]
                          : mergedInventory.filter(p => p.status === s).length;
    projects = [{
      project:       PROJECT,
      path:          window.location.pathname.replace(/\/$/, "").split("/").pop() || PROJECT,
      published:     "",
      owner:         "",
      plans_count:   total,
      active:        count("active"),
      blocked:       count("blocked"),
      pending:       count("pending"),
      shipped:       count("shipped"),
      last_modified: idx.audit_date || "",
      milestones,
      top:           [],
      activity30:    [],
      tests_30d:     { pass: 0, runs: 0 },
    }];
  } else if (!Array.isArray(projects[0].milestones) || projects[0].milestones.length === 0) {
    // projects[0] exists but lacks milestones — merge from top-level if any
    projects = projects.map((p, i) => i === 0 ? { ...p, milestones: p.milestones || milestones } : p);
  }

  window.STATE = {
    today:            new Date().toISOString().slice(0, 10),
    project:          PROJECT,
    projects,
    milestones,
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
