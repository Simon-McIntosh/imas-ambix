// v7 Shell — top bar + three-column body.
// Brand: reckon (the system). Project name comes from state.
// Changes from design bundle v7:
//   A. Brand shows "reckon" (not project name)
//   B. Fleet prompts use dynamic project name (no imas_ambix hardcoding)
//   C. Fleet prompts allow unilateral decision resolution
//   D/E. "Decisions N" button replaces "Resolve N" — cycles through blocked plans (sprint)
//   F. Sidebar collapse (⌘B handle) collapses BOTH filter + plan list → full-screen
//   G. Sprint nav (‹ ›) also appears in the cockpit/overview view

function v7ParseHash() {
  const h = (window.location.hash || "").replace(/^#/, "");
  if (!h || h === "cockpit") return { view: "cockpit" };
  if (h.startsWith("plan/")) return { view: "plan", slug: decodeURIComponent(h.slice(5)) };
  if (h.startsWith("sprint/")) return { view: "sprint", sprint: decodeURIComponent(h.slice(7)) };
  return { view: "cockpit" };
}

function useV7Hash() {
  const [route, setRoute] = useState(v7ParseHash());
  useEffect(() => {
    const onHash = () => setRoute(v7ParseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const nav = useCallback((to) => {
    if (to.view === "cockpit") window.location.hash = "#cockpit";
    else if (to.view === "plan") window.location.hash = `#plan/${encodeURIComponent(to.slug)}`;
    else if (to.view === "sprint") window.location.hash = `#sprint/${encodeURIComponent(to.sprint)}`;
  }, []);
  return [route, nav];
}

// ─── Top bar ────────────────────────────────────────────────────────────

function V7TopBar({ route, onNav }) {
  const onOverview = route.view === "cockpit";
  const goPlans = () => {
    const M = window.STATE;
    const target = M.inventory.find(p => p.status === "active") || M.inventory[0];
    if (target) onNav({ view: "plan", slug: target.slug });
  };
  const goOverview = () => {
    if (onOverview) goPlans();
    else onNav({ view: "cockpit" });
  };
  const projectName = window.STATE?.projects?.[0]?.project ||
    document.querySelector('meta[name="docs-project"]')?.content || "";
  return (
    <div className="v7-topbar">
      <div className="brand">
        <a href="/" style={{ display: "flex", alignItems: "center", gap: 9, textDecoration: "none", color: "inherit" }}>
          <span className="mark">R</span>
          <span className="name">reckon</span>
        </a>
        {projectName && (
          <>
            <span style={{ color: "var(--faint)", fontSize: 13, fontWeight: 400, margin: "0 1px" }}>·</span>
            <span className="proj">{projectName}</span>
          </>
        )}
      </div>
      <span className="sp"></span>
      <div className="v7-glyph-tabs">
        <button
          className={`v7-glyph ${!onOverview ? "active" : ""}`}
          onClick={goPlans}
          title="Plans — work on individual plans"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M3 4h10M3 8h10M3 12h7"/>
          </svg>
          Plans
        </button>
        <button
          className={`v7-glyph ${onOverview ? "active" : ""}`}
          onClick={goOverview}
          title="Overview — milestones, decisions, recent activity"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2.5" y="2.5" width="4.5" height="4.5" rx="0.7"/>
            <rect x="9" y="2.5" width="4.5" height="4.5" rx="0.7"/>
            <rect x="2.5" y="9" width="4.5" height="4.5" rx="0.7"/>
            <rect x="9" y="9" width="4.5" height="4.5" rx="0.7"/>
          </svg>
          Overview
        </button>
      </div>
    </div>
  );
}

// ─── Filters column ─────────────────────────────────────────────────────

function FiltersCol({ filters, setFilters }) {
  const M = window.STATE;
  const statuses = ["active", "blocked", "pending", "shipped"];
  const milestones = M.projects?.[0]?.milestones || M.milestones || [];
  const sprints = M.sprints || [];

  const toggle = (group, value) => {
    setFilters(f => {
      const cur = (f[group] || []);
      if (cur.includes(value)) return { ...f, [group]: [] };
      return { ...f, [group]: [value] };
    });
  };

  const anyActive = (filters.status?.length || 0) + (filters.ms?.length || 0) + (filters.sprint?.length || 0) > 0;

  return (
    <aside className="v7-filters">
      <div className="v7-filter-group">
        <div className="v7-filter-h">Status</div>
        {statuses.map(s => {
          const n = M.inventory.filter(p => p.status === s).length;
          const on = (filters.status || []).includes(s);
          return (
            <div key={s} className={`v7-chip ${on ? "on" : ""}`} onClick={() => toggle("status", s)}>
              <span className={`dot ${s}`}></span>
              <span style={{ textTransform: "capitalize" }}>{s}</span>
              <span className="n">{n}</span>
            </div>
          );
        })}
      </div>

      <div className="v7-filter-group">
        <div className="v7-filter-h">Milestone</div>
        {milestones.map(m => {
          const n = M.inventory.filter(p => p.ms === m.id).length;
          const on = (filters.ms || []).includes(m.id);
          if (n === 0) return null;
          return (
            <div key={m.id} className={`v7-chip ${on ? "on" : ""}`} onClick={() => toggle("ms", m.id)}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: m.status === "active" ? "var(--accent)" : m.status === "shipped" ? "var(--good)" : "var(--muted)", minWidth: 22 }}>{m.id}</span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, minWidth: 0 }}>{m.name}</span>
              <span className="n">{n}</span>
            </div>
          );
        })}
      </div>

      <div className="v7-filter-group">
        <div className="v7-filter-h">Sprint</div>
        {sprints.map(s => {
          const slugs = (s.items || []).map(it => typeof it === "string" ? it : it.slug);
          const n = slugs.length;
          const on = (filters.sprint || []).includes(s.id);
          return (
            <div key={s.id} className={`v7-chip ${on ? "on" : ""}`} onClick={() => toggle("sprint", s.id)}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: s.status === "active" ? "var(--accent)" : s.status === "shipped" ? "var(--good)" : "var(--muted)", minWidth: 22 }}>{s.id}</span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, minWidth: 0 }}>{s.theme?.slice(0, 22) + (s.theme?.length > 22 ? "…" : "")}</span>
              <span className="n">{n}</span>
            </div>
          );
        })}
      </div>

      <button className="v7-clear" disabled={!anyActive} onClick={() => setFilters({})}>
        {anyActive ? "× clear filters" : "no filters set"}
      </button>
    </aside>
  );
}

// ─── Plans list column ──────────────────────────────────────────────────

function ListCol({ search, setSearch, route, onNav, items }) {
  const M = window.STATE;
  const sprints = M?.sprints || [];

  const groups = React.useMemo(() => {
    const bySprint = {};
    for (const s of sprints) bySprint[s.id] = { sprint: s, plans: [] };
    bySprint._none = { sprint: null, plans: [] };
    for (const p of items) {
      const k = p.sprint || "_none";
      (bySprint[k] || bySprint._none).plans.push(p);
    }
    const order = { active: 0, planned: 1, shipped: 2 };
    return Object.values(bySprint)
      .filter(g => g.plans.length > 0)
      .sort((a, b) => {
        if (!a.sprint) return 1;
        if (!b.sprint) return -1;
        return (order[a.sprint.status] ?? 9) - (order[b.sprint.status] ?? 9);
      });
  }, [items, sprints]);

  const [collapsed, setCollapsed] = useState(() => {
    const c = {};
    for (const s of sprints) if (s.status === "shipped") c[s.id] = true;
    return c;
  });
  const toggle = (id) => setCollapsed(c => ({ ...c, [id]: !c[id] }));

  return (
    <div className="v7-list">
      <div className="v7-search">
        <input
          placeholder="Search plans…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="count">{items.length} plan{items.length === 1 ? "" : "s"}</div>
      </div>
      {items.length === 0 ? (
        <div className="v7-list-empty">No plans match.</div>
      ) : groups.map(g => {
        const id = g.sprint?.id || "_none";
        const isOpen = !collapsed[id];
        const isActiveSprintRoute = route.view === "sprint" && route.sprint === id;
        return (
          <div key={id} className="v7-sgroup">
            <div className={`v7-sgroup-h ${isActiveSprintRoute ? "route-active" : ""}`} onClick={() => toggle(id)}>
              <span className="car">{isOpen ? "▾" : "▸"}</span>
              <span className="id">{g.sprint ? g.sprint.id : "—"}</span>
              <span className="theme">
                {g.sprint ? g.sprint.theme : "Unscheduled"}
                {g.sprint && (
                  <span className={`st ${g.sprint.status}`}>{g.sprint.status}</span>
                )}
              </span>
              <span className="n">{g.plans.length}</span>
              {g.sprint && (
                <a
                  className="board-link"
                  href={`#sprint/${g.sprint.id}`}
                  onClick={(e) => e.stopPropagation()}
                  title="Open sprint board"
                >▦</a>
              )}
            </div>
            {isOpen && g.plans.map(p => {
              const active = route.view === "plan" && route.slug === p.slug;
              return (
                <div
                  key={p.slug}
                  className={`v7-row ${active ? "active" : ""}`}
                  onClick={() => onNav({ view: "plan", slug: p.slug })}
                >
                  <span className={`dot ${p.status}`}></span>
                  <div>
                    <div className="t">{p.title}</div>
                    <div className="meta">
                      <span className="ms">{p.ms}</span>
                      <span className="sp">·</span>
                      <span className="pct">{Math.round((p.impl || 0) * 100)}%</span>
                      <span className="sp">·</span>
                      <span>{p.last}</span>
                    </div>
                    {((p.dec_open || 0) > 0 || (p.blockers || 0) > 0) && (
                      <div className="sigs">
                        {(p.dec_open || 0) > 0 && <span className="sig dec">D {p.dec_open}</span>}
                        {(p.blockers || 0) > 0 && <span className="sig blk">! {p.blockers}</span>}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

// ─── Title bar ──────────────────────────────────────────────────────────

function TitleBar({ route, onNav, onOpenPrompt }) {
  const M = window.STATE;
  if (route.view === "cockpit") {
    return (
      <div className="v7-titlebar">
        <div className="row1">
          <span className="title">Overview</span>
        </div>
      </div>
    );
  }
  if (route.view === "plan") {
    const p = M.inventory.find(x => x.slug === route.slug);
    if (!p) return null;
    const openDecs = (p.decisions || []).filter(d => !d.chosen).length;
    const blockedByDecisions = openDecs > 0;
    return (
      <div className="v7-titlebar">
        <div className="row1">
          <span className="crumbs"><code>/{route.slug}</code></span>
          <span className="title">{p.title}</span>
          <div className="actions">
            {openDecs > 0 && (
              <button
                className="decisions-btn"
                onClick={() => {
                  const el = document.querySelector(".v6-dec:not(.taken)") || document.querySelector(".v6-dec");
                  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
                }}
                title={`${openDecs} pending decision${openDecs === 1 ? "" : "s"}`}
              >
                Decisions <span className="dec-badge">{openDecs}</span>
              </button>
            )}
            <button
              className="gen-prompt"
              onClick={onOpenPrompt}
              disabled={blockedByDecisions}
              title="Generate handoff prompt"
            >
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <path d="M4 3h6l2 2v8H4z"/>
                <path d="M10 3v2h2"/>
                <path d="M6 7h4M6 9h4M6 11h2"/>
              </svg>
              Generate prompt
            </button>
          </div>
        </div>
        <div className="row2">
          <span className={`status-pill ${p.status}`}><span className="dot"></span><span>{p.status}</span></span>
          <span className="dot-sep">·</span>
          <span className="meta-item"><span className="k">ms</span><span className="v">{p.ms}</span></span>
          {p.sprint && <>
            <span className="dot-sep">·</span>
            <span className="meta-item"><span className="k">sprint</span><a className="v" href={`#sprint/${p.sprint}`} style={{ borderBottom: "1px dotted var(--line)" }}>{p.sprint}</a></span>
          </>}
          <span className="dot-sep">·</span>
          <span className="meta-item"><span className="k">progress</span><span className="v">{Math.round((p.impl || 0) * 100)}%</span></span>
          <span className="dot-sep">·</span>
          <span className="meta-item"><span className="k">last</span><span className="v">{p.last}</span></span>
          {p.owner && <>
            <span className="dot-sep">·</span>
            <span className="meta-item"><span className="k">owner</span><span className="v">{p.owner}</span></span>
          </>}
        </div>
      </div>
    );
  }
  if (route.view === "sprint") {
    const sprints = M.sprints || [];
    const idx = sprints.findIndex(s => s.id === route.sprint);
    const s = sprints[idx];
    const slugSet = new Set((s?.items || []).map(it => typeof it === "string" ? it : it.slug));
    const inv = [...slugSet].map(slug => {
      const p = M.inventory.find(x => x.slug === slug);
      if (!p) return null;
      const stored = (window.v6?.planLoad?.(slug)) || {};
      const overlay = stored.decisions || {};
      const liveDecs = (p.decisions || []).map(d => {
        const ov = overlay[d.key];
        if (ov?.choice) return { ...d, chosen: ov.choice, rationale: ov.rationale, when: ov.when, by: ov.by };
        return d;
      });
      return { ...p, decisions: liveDecs };
    }).filter(Boolean);
    const totalOpen = inv.reduce((n, p) => n + (p.decisions || []).filter(d => !d.chosen).length, 0);
    const blocked = totalOpen > 0;
    const blockedPlans = inv.filter(p => (p.decisions || []).some(d => !d.chosen));

    // Cycle through all blocked plans on repeated clicks
    const handleDecisions = () => {
      if (blockedPlans.length === 0) return;
      const currentSlug = route.view === "plan" ? route.slug : null;
      const currentIdx = blockedPlans.findIndex(p => p.slug === currentSlug);
      const nextIdx = (currentIdx >= 0 && currentIdx < blockedPlans.length - 1)
        ? currentIdx + 1
        : 0;
      onNav({ view: "plan", slug: blockedPlans[nextIdx].slug });
    };

    const handleGen = () => {
      window.dispatchEvent(new CustomEvent("v7-open-fleet-prompt"));
    };
    return (
      <div className="v7-titlebar">
        <div className="row1">
          <span className="crumbs">sprint</span>
          <span className="title">{s ? `${s.id} · ${s.theme}` : route.sprint}</span>
          <div className="actions">
            <button className="v7-nav-btn" disabled={idx <= 0} onClick={() => onNav({ view: "sprint", sprint: sprints[idx - 1].id })}>‹</button>
            <button className="v7-nav-btn" disabled={idx >= sprints.length - 1} onClick={() => onNav({ view: "sprint", sprint: sprints[idx + 1].id })}>›</button>
            {blocked && (
              <button className="decisions-btn" onClick={handleDecisions} title="Navigate to next plan with pending decisions">
                Decisions <span className="dec-badge">{totalOpen}</span>
              </button>
            )}
            <button
              className="gen-prompt"
              onClick={handleGen}
              disabled={blocked}
              title="Generate fleet prompt"
            >
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <path d="M4 3h6l2 2v8H4z"/><path d="M10 3v2h2"/><path d="M6 7h4M6 9h4M6 11h2"/>
              </svg>
              Generate prompt
            </button>
          </div>
        </div>
        {s && (
          <div className="row2">
            <span className={`status-pill ${s.status}`}><span className="dot"></span><span>{s.status}</span></span>
            <span className="dot-sep">·</span>
            <span className="meta-item"><span className="k">starts</span><span className="v">{s.starts}</span></span>
            <span className="dot-sep">·</span>
            <span className="meta-item"><span className="k">ends</span><span className="v">{s.ends}</span></span>
            <span className="dot-sep">·</span>
            <span className="meta-item"><span className="k">items</span><span className="v">{s.items.length}</span></span>
          </div>
        )}
      </div>
    );
  }
  return null;
}

// ─── Dependency strip (top of plan body) ────────────────────────────────

function PlanDeps({ slug }) {
  const M = window.STATE;
  const p = M.inventory.find(x => x.slug === slug);
  if (!p) return null;
  const deps = p.depends_on || [];
  const blocks = p.blocks || [];
  if (deps.length === 0 && blocks.length === 0) return null;
  return (
    <div className="v7-deps">
      {deps.length > 0 && <>
        <span className="lbl">depends on</span>
        {deps.map(s => {
          const target = M.inventory.find(i => i.slug === s);
          const blocked = target?.status === "blocked";
          return <a key={s} className={`pill ${blocked ? "blocked" : ""}`} href={`#plan/${s}`}>{s}</a>;
        })}
      </>}
      {blocks.length > 0 && <>
        {deps.length > 0 && <span className="lbl" style={{ marginLeft: 14 }}>blocks</span>}
        {deps.length === 0 && <span className="lbl">blocks</span>}
        {blocks.map(s => <a key={s} className="pill" href={`#plan/${s}`}>{s}</a>)}
      </>}
    </div>
  );
}

// ─── App ────────────────────────────────────────────────────────────────

function V7ReadyGate({ children }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (window.STATE_READY) window.STATE_READY.then(() => setReady(true));
    else setReady(true);
  }, []);
  if (!ready) {
    return <div style={{ padding: 48, textAlign: "center", fontFamily: "var(--mono)", fontSize: 13, color: "var(--muted)" }}>Loading plan state…</div>;
  }
  return children;
}

function V7App() {
  const [route, nav] = useV7Hash();
  const [filters, setFilters] = useState({});
  const [search, setSearch] = useState("");
  const [promptOpen, setPromptOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [cockpitSprintIdx, setCockpitSprintIdx] = useState(null);

  const M = window.STATE;
  const items = useMemo(() => {
    if (!M) return [];
    let list = M.inventory;
    if (filters.status?.length) list = list.filter(p => filters.status.includes(p.status));
    if (filters.ms?.length) list = list.filter(p => filters.ms.includes(p.ms));
    if (filters.sprint?.length) list = list.filter(p => filters.sprint.includes(p.sprint));
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(p =>
        p.title?.toLowerCase().includes(q) ||
        p.slug?.toLowerCase().includes(q) ||
        p.summary?.toLowerCase().includes(q)
      );
    }
    return list;
  }, [M, filters, search]);

  useEffect(() => {
    if (!promptOpen) return;
    setPromptOpen(false);
    window.dispatchEvent(new CustomEvent("v7-open-prompt"));
  }, [promptOpen]);

  // ⌘B / Ctrl+B toggles the sidebar (filters + plan list)
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "b" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSidebarCollapsed(c => !c);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="v7-app">
      <V7TopBar route={route} onNav={nav} />
      <div className={`v7-3col ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${route.view === "cockpit" ? "overview-mode" : ""}`}>
        <button
          className="v7-filter-handle"
          onClick={() => setSidebarCollapsed(c => !c)}
          title={sidebarCollapsed ? "Show sidebar · ⌘B" : "Hide sidebar · ⌘B"}
          aria-label="Toggle sidebar"
        >
          <span></span><span></span>
        </button>
        <FiltersCol filters={filters} setFilters={setFilters} />
        <ListCol search={search} setSearch={setSearch} route={route} onNav={nav} items={items} />
        <div className="v7-content">
          <TitleBar route={route} onNav={nav} onOpenPrompt={() => setPromptOpen(true)} />
          <div className="v7-body">
            {route.view === "sprint" && <V7FleetPrompt sprintId={route.sprint} />}
            {route.view === "plan" && <PlanDeps slug={route.slug} />}
            {route.view === "cockpit" && <V7CockpitBody onNav={nav} cockpitSprintIdx={cockpitSprintIdx} setCockpitSprintIdx={setCockpitSprintIdx} />}
            {route.view === "plan" && <V6Plan slug={route.slug} onNav={nav} />}
            {route.view === "sprint" && <V6Sprint sprintId={route.sprint} onNav={nav} />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Cockpit body ────────────────────────────────────────────────────────
// Change G: accepts cockpitSprintIdx + setCockpitSprintIdx for sprint nav.

function V7CockpitBody({ onNav, cockpitSprintIdx, setCockpitSprintIdx }) {
  const M = window.STATE;
  if (!M) return null;
  const project = M.projects[0] || {};
  const allSprints = M.sprints || [];
  const defaultIdx = allSprints.findIndex(s => s.id === M.active_sprint_id);
  const displayIdx = cockpitSprintIdx !== null ? cockpitSprintIdx : (defaultIdx >= 0 ? defaultIdx : 0);
  const sprint = allSprints[displayIdx] || M.sprint;

  const decisionPlans = M.inventory
    .filter(i => (i.dec_open || 0) > 0)
    .sort((a, b) => (b.dec_open || 0) - (a.dec_open || 0));
  const decisionTotal = decisionPlans.reduce((n, p) => n + (p.dec_open || 0), 0);

  return (
    <>
      {project.plans_count != null && (
        <div className="v7-ck-sub">
          {project.plans_count} plans · owner {project.owner}
        </div>
      )}

      <div className="v7-ck-h">
        <span className="v6-eyebrow">Milestones</span>
      </div>
      <div className="v6-ms" style={{ marginBottom: 4 }}>
        {(project.milestones || []).map(m => (
          <button key={m.id} className={`v6-ms-tile ${m.status}`}
            onClick={() => {
              const target = M.inventory.find(i => i.ms === m.id && i.status === "active")
                || M.inventory.find(i => i.ms === m.id);
              if (target) onNav({ view: "plan", slug: target.slug });
            }}>
            <div className="fill" style={{ "--w": `${m.pct}%` }}></div>
            <div className="lbl">{m.id} · <span className={`stat-${m.status}`}>{m.status}</span></div>
            <div className="nm">{m.name}</div>
            <div className="pct">{m.pct}%</div>
          </button>
        ))}
      </div>

      <div className="v7-ck-h">
        <span className="v6-eyebrow">Sprint {sprint?.id} · {sprint?.theme}</span>
        <div className="v7-sprint-nav">
          <button className="v7-nav-btn" disabled={displayIdx <= 0} onClick={() => setCockpitSprintIdx(displayIdx - 1)}>‹</button>
          <button className="v7-nav-btn" disabled={displayIdx >= allSprints.length - 1} onClick={() => setCockpitSprintIdx(displayIdx + 1)}>›</button>
        </div>
        <a className="v7-board-icon" href={`#sprint/${sprint?.id}`} title="Open sprint board">▦</a>
      </div>
      <div className="v7-ck-list" style={{ marginBottom: 22 }}>
        {(sprint?.items || []).map(it => {
          const slug = typeof it === "string" ? it : it.slug;
          const justification = typeof it === "object" ? it.justification : null;
          const p = M.inventory.find(x => x.slug === slug);
          if (!p) return null;
          const pct = Math.round((p.impl || 0) * 100);
          return (
            <a key={slug} className="v7-ck-row" href={`#plan/${slug}`}>
              <span className={`v7-ck-dot ${p.status}`}></span>
              <div className="v7-ck-body">
                <div className="v7-ck-title">{p.title}</div>
                {justification && <div className="v7-ck-just">{justification}</div>}
              </div>
              <div className="v7-ck-prog">
                <span className="v7-ck-bar"><i style={{ width: `${pct}%` }} className={p.status === "shipped" ? "shipped" : p.status === "blocked" ? "blocked" : ""}></i></span>
                <span className="v7-ck-pct">{pct}%</span>
              </div>
              <span className="v7-ck-arr">›</span>
            </a>
          );
        })}
      </div>

      <div className="v7-ck-h">
        <span className="v6-eyebrow">Decisions · {decisionTotal} open across {decisionPlans.length} plan{decisionPlans.length === 1 ? "" : "s"}</span>
      </div>
      {decisionPlans.length === 0 ? (
        <div className="v7-ck-empty">No open decisions.</div>
      ) : (
        <div className="v7-ck-list">
          {decisionPlans.map(p => (
            <a key={p.slug} className="v7-ck-row" href={`#plan/${p.slug}`}>
              <span className="v7-ck-num">{p.dec_open}</span>
              <div className="v7-ck-body">
                <div className="v7-ck-title">{p.title}</div>
                <div className="v7-ck-slug">/{p.slug}</div>
              </div>
              <span className="v7-ck-arr">›</span>
            </a>
          ))}
        </div>
      )}

      <div className="v7-ck-h"><span className="v6-eyebrow">Recent activity</span></div>
      <div className="card">
        <div className="card-body">
          <div className="ledger">
            {(M.timeline || []).slice(0, 6).map((t, i) => (
              <React.Fragment key={i}>
                <span className="when">{t.when}</span>
                <span className={`who ${t.who.startsWith("agent") ? "bot" : ""}`}>{t.who}</span>
                <span className="what">{t.what}</span>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Fleet prompt modal ─────────────────────────────────────────────────
// Change B: dynamic project name. Change C: allow unilateral decisions.

function V7FleetPrompt({ sprintId }) {
  const M = window.STATE;
  const sprint = M.sprints.find(s => s.id === sprintId);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const h = () => setOpen(true);
    window.addEventListener("v7-open-fleet-prompt", h);
    return () => window.removeEventListener("v7-open-fleet-prompt", h);
  }, []);

  if (!sprint) return null;

  const slugSet = new Set(sprint.items.map(it => typeof it === "string" ? it : it.slug));
  const itemsArr = [...slugSet].map(slug => {
    const p = M.inventory.find(x => x.slug === slug);
    const meta = sprint.items.find(it => (typeof it === "string" ? it : it.slug) === slug);
    const just = typeof meta === "object" ? meta.justification : null;
    return p ? { ...p, justification: just } : null;
  }).filter(Boolean);

  const order = [];
  const visited = new Set();
  const visit = (p) => {
    if (visited.has(p.slug)) return;
    visited.add(p.slug);
    for (const dep of (p.depends_on || [])) {
      const depPlan = itemsArr.find(x => x.slug === dep);
      if (depPlan) visit(depPlan);
    }
    order.push(p);
  };
  for (const p of itemsArr) visit(p);

  const buildPrompt = () => {
    // Change B: dynamic project name from state
    const projectName = M.projects[0]?.project ||
      document.querySelector('meta[name="docs-project"]')?.content ||
      "project";

    let txt = `Orchestration\n  You are coordinating a fleet of workers across ${order.length} plans in a single\n  sprint. Dispatch in the order below; honour the dependency edges. Workers\n  whose dependencies are satisfied may run in parallel. Each worker must read\n  every plan it depends on in full, develop the plan further as it works,\n  inspect code under ${projectName}/ when ambiguous, honour locked decisions,\n  resolve open decisions as part of the work — document rationale in the state JSON.\n\nProject: ${projectName}\nSprint:  ${sprint.id}\nGoal:    ${sprint.theme}\nWindow:  ${sprint.starts} → ${sprint.ends}\n\nExecution sequence (resolved from depends_on within the sprint):\n`;
    order.forEach((p, i) => {
      txt += `  ${i + 1}. ${p.slug}${(p.depends_on || []).length ? "  (← " + p.depends_on.join(", ") + ")" : ""}\n`;
    });
    txt += `\nEach plan's individual prompt follows below as a numbered section.\n\n`;
    order.forEach((p, i) => {
      txt += `\n─── ${i + 1}/${order.length} · ${p.slug} ───\n`;
      const decisions = (p.decisions || []);
      const locked = decisions.filter(d => d.chosen);
      const openD = decisions.filter(d => !d.chosen);
      const lockedBlock = locked.length === 0 ? "  (none)" : locked.map(d => `  ${d.key} → ${d.chosen}`).join("\n");
      const openBlock = openD.length === 0 ? "  (none)" : openD.map(d => `  ${d.key} — ${d.title}`).join("\n");
      const next = (p.followups || [])[0];
      const comments = (p.comments) || (window.v6?.planLoad?.(p.slug)?.comments) || {};
      const commentEntries = Object.entries(comments).filter(([_, arr]) => (arr || []).length > 0);
      const commentsBlock = commentEntries.length === 0 ? "  (none)" :
        commentEntries.map(([sid, arr]) =>
          arr.map(c =>
            `  §${sid} · ${c.who} · ${c.when}\n` +
            (c.quote ? `      quote: "${c.quote.length > 200 ? c.quote.slice(0, 200) + "…" : c.quote}"\n` : "") +
            `      body: ${c.body}`
          ).join("\n")
        ).join("\n");
      txt += `Plan: ${p.slug}\nStatus: ${p.status} · ${p.phase || ""}\nJustification (sprint): ${p.justification || "—"}\n\nState to read\n  state/${projectName}/${p.slug}.json\n\nLocked decisions to honour\n${lockedBlock}\n\nOpen decisions (resolve as part of the work — document rationale)\n${openBlock}\n\nComments\n${commentsBlock}\n\nNext-up\n  ${next?.title || "—"}\n  ${next?.body || ""}\n\nDone-when\n  1. Land the work this prompt describes.\n  2. POST a followup to ${p.slug}.json#followups.\n  3. Mark the current followup resolved.\n`;
    });
    return txt;
  };

  if (!open) return null;
  return (
    <PromptModalAdHoc
      title={`Fleet · ${sprint.id}`}
      subtitle={`Orchestrate ${order.length} workers — sequence honours depends_on`}
      buildText={buildPrompt}
      onClose={() => setOpen(false)}
    />
  );
}

function PromptModalAdHoc({ title, subtitle, buildText, onClose }) {
  const [text, setText] = useState(() => buildText());
  useEffect(() => {
    const k = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", k);
    return () => document.removeEventListener("keydown", k);
  }, [onClose]);
  const copy = () => {
    navigator.clipboard?.writeText(text);
    onClose();
    if (window.flashSaved) window.flashSaved("prompt copied");
  };
  return (
    <div className="v6-modal-scrim" onClick={onClose}>
      <div className="v6-modal" onClick={(e) => e.stopPropagation()}>
        <div className="head">
          <div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, letterSpacing: "0.10em", textTransform: "uppercase", color: "var(--accent)", fontWeight: 600 }}>{title}</div>
            <h3 style={{ margin: "4px 0", fontSize: 17, fontWeight: 600 }}>{subtitle}</h3>
          </div>
          <button className="btn ghost" onClick={onClose}>Close · Esc</button>
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        <div className="foot">
          <span style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 11 }}>{text.length} chars</span>
          <span style={{ flex: 1 }}></span>
          <button className="btn primary" onClick={copy}>Copy to clipboard</button>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <V7ReadyGate><V7App /></V7ReadyGate>
);
