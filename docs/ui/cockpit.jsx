// Cockpit — minimal landing.
// Sidebar handles plan navigation; cockpit is just project-level state
// (milestones, decisions across plans, active sprint, recent activity).

function Cockpit({ onNav }) {
  const M = window.STATE;
  if (!M) return null;
  const project = M.projects?.[0] || { project: M.project || "", plans_count: M.inventory?.length || 0, milestones: M.milestones || [], owner: "" };
  const sprint = M.sprint;

  // Plans with open decisions
  const decisionPlans = M.inventory
    .filter(i => (i.dec_open || 0) > 0)
    .sort((a, b) => (b.dec_open || 0) - (a.dec_open || 0));
  const decisionTotal = decisionPlans.reduce((n, p) => n + (p.dec_open || 0), 0);

  return (
    <div className="r-page wide">
      <div className="r-ck-head">
        <div>
          <div className="r-eyebrow">project · {project.project}</div>
          <h1>{project.project}</h1>
          <div className="sub">
            {project.plans_count} plans · sprint <strong>{sprint.id}</strong> in flight · owner {project.owner}
          </div>
        </div>
        <span style={{ flex: 1 }}></span>
        <button className="btn primary">+ New plan</button>
      </div>

      {/* Milestone arc — each milestone clickable */}
      <div className="r-eyebrow">Milestone arc</div>
      <div className="r-ms">
        {project.milestones.map(m => (
          <button
            key={m.id}
            className={`r-ms-tile ${m.status}`}
            onClick={() => {
              const target = M.inventory.find(i => i.ms === m.id && i.status === "active")
                || M.inventory.find(i => i.ms === m.id);
              if (target) onNav({ view: "plan", slug: target.slug });
            }}
            title={`Jump to a plan on ${m.id}`}
          >
            <div className="fill" style={{ "--w": `${m.pct}%` }}></div>
            <div className="lbl">{m.id} · <span className={`stat-${m.status}`}>{m.status}</span></div>
            <div className="nm">{m.name}</div>
            <div className="pct">{m.pct}%</div>
          </button>
        ))}
      </div>

      {/* Decisions */}
      <div className="r-eyebrow" style={{ marginTop: 8 }}>
        Decisions · {decisionTotal} open across {decisionPlans.length} plan{decisionPlans.length === 1 ? "" : "s"}
      </div>
      {decisionPlans.length === 0 ? (
        <div className="r-dec-list" style={{ padding: 24, textAlign: "center", color: "var(--muted)" }}>
          No open decisions.
        </div>
      ) : (
        <div className="r-dec-list">
          {decisionPlans.map(p => (
            <a
              key={p.slug}
              className="r-dec-row"
              href={`#plan/${p.slug}`}
            >
              <span className="c">{p.dec_open}</span>
              <span className="body">
                <span className="t">
                  {p.title}
                  <span className="slug">/{p.slug}</span>
                </span>
                <div className="s">take in context →</div>
              </span>
              <span className="arr">→</span>
            </a>
          ))}
        </div>
      )}

      {/* Active sprint */}
      <div className="r-eyebrow" style={{ marginTop: 26 }}>Sprint {sprint.id} · {sprint.theme}</div>
      <div className="r-dec-list">
        {sprint.items.map((it, i) => {
          const slug = typeof it === "string" ? it : it.slug;
          const justification = typeof it === "object" ? it.justification : null;
          const p = M.inventory.find(x => x.slug === slug);
          if (!p) return null;
          return (
            <a key={slug} className="r-dec-row" href={`#plan/${slug}`}>
              <span className="c" style={{ background: "var(--bg-3)", color: "var(--ink-2)" }}>
                {p.impl ? Math.round(p.impl * 100) + "%" : "—"}
              </span>
              <span className="body">
                <span className="t">{p.title} <span className="slug">/{slug}</span>
                  <span className={`status ${p.status}`}><span className="dot"></span>{p.status}</span>
                </span>
                {justification && <div className="s" style={{ color: "var(--ink-2)", fontFamily: "var(--sans)" }}>{justification}</div>}
              </span>
              <span className="arr">→</span>
            </a>
          );
        })}
      </div>
      <div style={{ marginTop: 12 }}>
        <a className="btn ghost" href={`#sprint/${sprint.id}`}>Open sprint board →</a>
      </div>

      {/* Recent activity */}
      <div className="r-eyebrow" style={{ marginTop: 26 }}>Recent activity</div>
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
    </div>
  );
}

window.Cockpit = Cockpit;
