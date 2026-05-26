// Two graph components:
//   CriticalPathView — top-level Graph tab. Lead with mini-map; then critical
//     path chain with ‹ › nav across all maximal-length paths; status-grouped
//     list below. Generate-prompt button opens PathPromptModal.
//   RadialFan — plan-view sub-mode. Focal plan centred, deps left, blocks
//     right. Single-hop only. Click satellite to navigate.
//
// Both compute critical path (longest dep chain ending at active/blocked).

function _criticalPath(plans) {
  const bySlug = Object.fromEntries(plans.map(p => [p.slug, p]));
  const pathLen = {}, pathPrev = {};
  function lp(slug, seen = new Set()) {
    if (pathLen[slug] !== undefined) return pathLen[slug];
    if (seen.has(slug)) return 0;
    seen.add(slug);
    const deps = (bySlug[slug]?.depends_on || []).filter(d => bySlug[d]);
    if (deps.length === 0) { pathLen[slug] = 1; return 1; }
    let best = 0, bestPrev = null;
    for (const d of deps) {
      const v = lp(d, new Set(seen));
      if (v > best) { best = v; bestPrev = d; }
    }
    pathLen[slug] = best + 1;
    pathPrev[slug] = bestPrev;
    return pathLen[slug];
  }
  plans.forEach(p => lp(p.slug));
  const live = plans.filter(p => p.status === "active" || p.status === "blocked");
  let critEnd = null, critLen = 0;
  for (const p of live) {
    if (pathLen[p.slug] > critLen) { critLen = pathLen[p.slug]; critEnd = p.slug; }
  }
  const chain = [];
  let cur = critEnd;
  while (cur) { chain.unshift(cur); cur = pathPrev[cur]; }
  return { chain, critLen, critEnd, pathLen, pathPrev, bySlug };
}

// Returns an array of all maximal-length chains (each is a slug array).
// Sorted by chain length descending. Multiple endpoints may share the max length.
function _allCriticalPaths(plans) {
  const bySlug = Object.fromEntries(plans.map(p => [p.slug, p]));
  const pathLen = {}, pathPrev = {};
  function lp(slug, seen = new Set()) {
    if (pathLen[slug] !== undefined) return pathLen[slug];
    if (seen.has(slug)) return 0;
    seen.add(slug);
    const deps = (bySlug[slug]?.depends_on || []).filter(d => bySlug[d]);
    if (deps.length === 0) { pathLen[slug] = 1; return 1; }
    let best = 0, bestPrev = null;
    for (const d of deps) {
      const v = lp(d, new Set(seen));
      if (v > best) { best = v; bestPrev = d; }
    }
    pathLen[slug] = best + 1;
    pathPrev[slug] = bestPrev;
    return pathLen[slug];
  }
  plans.forEach(p => lp(p.slug));

  const live = plans.filter(p => p.status === "active" || p.status === "blocked");
  if (live.length === 0) return [];

  // Group by chain length, sort desc. Return all endpoints (not just the single max).
  const sorted = [...live].sort((a, b) => (pathLen[b.slug] || 0) - (pathLen[a.slug] || 0));
  const maxLen = pathLen[sorted[0]?.slug] || 0;
  if (maxLen === 0) return [];

  // Collect all endpoints with length >= maxLen (i.e. tied for longest)
  const endpoints = sorted.filter(p => (pathLen[p.slug] || 0) >= maxLen);

  return endpoints.map(ep => {
    const chain = [];
    let cur = ep.slug;
    while (cur) { chain.unshift(cur); cur = pathPrev[cur]; }
    return chain;
  });
}

// ─── Path prompt modal ────────────────────────────────────────────────────

function PathPromptModal({ chain, bySlug, onClose }) {
  const M = window.STATE;
  const projectName = M?.projects?.[0]?.project || "project";

  // Check for open decisions in the chain
  const openDecPlans = chain
    .map(slug => bySlug[slug])
    .filter(Boolean)
    .filter(p => (p.decisions || []).some(d => !d.chosen));

  const openDecCount = openDecPlans.reduce((n, p) => n + (p.decisions || []).filter(d => !d.chosen).length, 0);
  const blocked = openDecCount > 0;

  const buildPrompt = () => {
    const endSlug = chain[chain.length - 1] || "—";
    let txt = `Orchestration\n  You are coordinating a fleet of workers across ${chain.length} plans on a\n  dependency route. Dispatch in topological order (dependencies first);\n  honour the dependency edges. Read each referenced plan in full before\n  starting work.\n\nProject: ${projectName}\nRoute:   depends_on chain ending at ${endSlug}\nPlans:   ${chain.length}\n\nExecution sequence (topological — earlier = upstream dependency):\n`;
    chain.forEach((slug, i) => {
      txt += `  ${i + 1}. ${slug}\n`;
    });
    txt += `\nEach plan's detail follows below.\n`;

    chain.forEach((slug, i) => {
      const p = bySlug[slug];
      if (!p) return;
      const decisions = p.decisions || [];
      const locked = decisions.filter(d => d.chosen);
      const openD = decisions.filter(d => !d.chosen);
      const next = (p.followups || [])[0];

      const lockedBlock = locked.length === 0
        ? null
        : locked.map(d => `  ${d.key} → ${d.chosen}${d.rationale ? "\n      reason: " + d.rationale : ""}`).join("\n");
      const openBlock = openD.length === 0
        ? null
        : openD.map(d => `  ${d.key} — ${d.title}`).join("\n");

      const comments = (p.comments) || (window.reckon?.planLoad?.(slug)?.comments) || {};
      const commentEntries = Object.entries(comments).filter(([_, arr]) => (arr || []).length > 0);
      const commentsBlock = commentEntries.length === 0
        ? null
        : commentEntries.map(([sid, arr]) =>
            arr.map(c =>
              `  §${sid} · ${c.who} · ${c.when}\n` +
              (c.quote ? `      quote: "${c.quote.length > 200 ? c.quote.slice(0, 200) + "…" : c.quote}"\n` : "") +
              `      body: ${c.body}`
            ).join("\n")
          ).join("\n");

      txt += `\n─── ${i + 1}/${chain.length} · ${slug} ───\n`;
      txt += `Plan:   ${slug}\nTitle:  ${p.title}\nStatus: ${p.status}${p.phase ? " · " + p.phase : ""}\n`;
      if (p.ms) txt += `MS:     ${p.ms}\n`;
      if (p.sprint) txt += `Sprint: ${p.sprint}\n`;
      if (p.summary) txt += `\nSummary\n  ${p.summary}\n`;
      txt += `\nState to read\n  state/${projectName}/${slug}.json\n`;
      if (lockedBlock) txt += `\nLocked decisions (honour these)\n${lockedBlock}\n`;
      if (openBlock) txt += `\nOpen decisions (surface, do not resolve)\n${openBlock}\n`;
      if (commentsBlock) txt += `\nComments (anchored to sections)\n${commentsBlock}\n`;
      if (next) {
        txt += `\nNext-up\n  ${next.title}\n  ${next.body || ""}`;
        if (next.blocked_by) txt += `\n  Blocked by: ${next.blocked_by.slug || next.blocked_by} — ${next.blocked_by.reason || ""}`;
        txt += `\n`;
      }
      txt += `\nDone-when\n  1. Land the work described above.\n  2. POST a new followup to ${slug}.json#followups with what landed.\n  3. Mark the current followup resolved with outcome.\n`;
    });
    return txt;
  };

  const [text, setText] = React.useState(() => blocked ? "" : buildPrompt());
  React.useEffect(() => {
    if (!blocked) setText(buildPrompt());
  }, [chain.join(","), blocked]);

  React.useEffect(() => {
    const k = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", k);
    return () => document.removeEventListener("keydown", k);
  }, [onClose]);

  const copy = () => {
    navigator.clipboard?.writeText(text);
    onClose();
    if (window.flashSaved) window.flashSaved("path prompt copied");
  };

  const goFirstBlocking = () => {
    if (openDecPlans.length === 0) return;
    window.location.hash = `#plan/${openDecPlans[0].slug}`;
    onClose();
  };

  return (
    <div className="r-modal-scrim" onClick={onClose}>
      <div className="r-modal" onClick={(e) => e.stopPropagation()}>
        <div className="head">
          <div>
            <div className="r-eyebrow">Critical path · fleet prompt</div>
            <h3>{chain[chain.length - 1] || "—"} route · {chain.length} plans</h3>
            <div style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 2 }}>
              Topological order — upstream dependencies first.
            </div>
          </div>
          <button className="btn ghost" onClick={onClose}>Close · Esc</button>
        </div>
        {blocked ? (
          <div style={{ padding: "24px 20px", background: "var(--bad-2)", borderRadius: 6, margin: "12px 20px", display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ color: "var(--bad)", fontWeight: 600 }}>{openDecCount} open decision{openDecCount === 1 ? "" : "s"}</span>
            <span style={{ color: "var(--bad)", fontSize: 13 }}>Resolve all decisions in the chain before generating a prompt.</span>
            <span style={{ flex: 1 }}></span>
            <button className="btn" style={{ background: "var(--bad)", color: "#fff", border: "none" }} onClick={goFirstBlocking}>
              Go to first blocking plan →
            </button>
          </div>
        ) : (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
          />
        )}
        <div className="foot">
          <span style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 11 }}>
            {chain.length} plans · {blocked ? "blocked" : `${text.length} chars`}
          </span>
          <span style={{ flex: 1 }}></span>
          <button className="btn primary" onClick={copy} disabled={blocked}>Copy to clipboard</button>
        </div>
      </div>
    </div>
  );
}

// ─── Critical-path-first view (top-level Graph tab) ──────────────────────

function CriticalPathView({ onNav }) {
  const M = window.STATE;
  if (!M) return null;

  const allPaths = React.useMemo(() => _allCriticalPaths(M.inventory), [M.inventory]);
  const { bySlug, pathLen } = React.useMemo(() => _criticalPath(M.inventory), [M.inventory]);

  const [pathIdx, setPathIdx] = React.useState(0);
  const safeIdx = Math.max(0, Math.min(pathIdx, allPaths.length - 1));
  const chain = allPaths[safeIdx] || [];
  const critLen = chain.length;
  const onPath = new Set(chain);

  const [showPrompt, setShowPrompt] = React.useState(false);

  const otherActive = M.inventory
    .filter(p => !onPath.has(p.slug) && (p.status === "active" || p.status === "blocked"))
    .sort((a, b) => (b.impl || 0) - (a.impl || 0));

  // Mini-map: every plan as a dot, edges between deps; critical path coloured.
  const mini = React.useMemo(() => {
    const plans = M.inventory;
    const depth = {};
    const bs = bySlug;
    function d(s, seen = new Set()) {
      if (depth[s] !== undefined) return depth[s];
      if (seen.has(s)) return 0;
      seen.add(s);
      const dd = (bs[s]?.depends_on || []).filter(x => bs[x]);
      if (dd.length === 0) { depth[s] = 0; return 0; }
      depth[s] = 1 + Math.max(...dd.map(x => d(x, seen)));
      return depth[s];
    }
    plans.forEach(p => d(p.slug));
    const maxD = Math.max(0, ...Object.values(depth));
    const byDepth = {};
    for (const p of plans) (byDepth[depth[p.slug]] = byDepth[depth[p.slug]] || []).push(p);
    const W = 280, H = 130;
    const colW = W / (maxD + 2);
    const pos = {};
    for (let i = 0; i <= maxD; i++) {
      const col = byDepth[i] || [];
      const rowH = (H - 20) / Math.max(1, col.length);
      col.forEach((p, j) => {
        pos[p.slug] = { x: 14 + i * colW, y: 10 + (j + 0.5) * rowH };
      });
    }
    const edges = [];
    for (const p of plans) {
      for (const dep of (p.depends_on || [])) {
        if (!bs[dep]) continue;
        const a = pos[dep], b = pos[p.slug];
        if (!a || !b) continue;
        edges.push({ a, b, crit: onPath.has(dep) && onPath.has(p.slug) });
      }
    }
    return { plans, pos, edges, W, H };
  }, [M.inventory, bySlug, chain]);

  function statusColor(s) {
    return s === "active" ? "var(--accent)" :
      s === "blocked" ? "var(--bad)" :
      s === "pending" ? "var(--warn)" :
      s === "shipped" ? "var(--good)" : "var(--muted)";
  }

  // Check open decisions in the displayed path
  const openDecCount = chain.reduce((n, slug) => {
    const p = bySlug[slug];
    return n + (p?.decisions || []).filter(d => !d.chosen).length;
  }, 0);

  return (
    <div className="r-graph">
      {/* Mini-map — top of the page */}
      <div className="r-graph-section" style={{ marginBottom: 20 }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 8 }}>Map · {M.inventory.length} plans</div>
        <div className="r-mini-map">
          <svg viewBox={`0 0 ${mini.W} ${mini.H}`} width="100%" height={mini.H}>
            {mini.edges.map((e, i) => (
              <line key={i} x1={e.a.x} y1={e.a.y} x2={e.b.x} y2={e.b.y}
                stroke={e.crit ? "var(--accent)" : "var(--hair)"}
                strokeWidth={e.crit ? 1.8 : 1}/>
            ))}
            {mini.plans.map(p => {
              const xy = mini.pos[p.slug];
              if (!xy) return null;
              const c = statusColor(p.status);
              const isCrit = onPath.has(p.slug);
              return (
                <g key={p.slug} transform={`translate(${xy.x},${xy.y})`} style={{ cursor: "pointer" }}
                   onClick={() => onNav({ view: "plan", slug: p.slug })}>
                  <title>{p.title} · {p.status}</title>
                  {isCrit && <circle r="6" fill="none" stroke="var(--accent)" strokeWidth="1.5"/>}
                  <circle r="3" fill={c}/>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Critical path chain with nav and generate-prompt */}
      <div className="r-graph-head">
        <div className="r-crit-chain-nav">
          <button
            className="r-nav-btn"
            disabled={safeIdx <= 0}
            onClick={() => setPathIdx(i => Math.max(0, i - 1))}
            title="Previous critical path"
          >‹</button>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", minWidth: 36, textAlign: "center" }}>
            {allPaths.length > 0 ? `${safeIdx + 1}/${allPaths.length}` : "—"}
          </span>
          <button
            className="r-nav-btn"
            disabled={safeIdx >= allPaths.length - 1}
            onClick={() => setPathIdx(i => Math.min(allPaths.length - 1, i + 1))}
            title="Next critical path"
          >›</button>
        </div>
        <span style={{ flex: 1 }}></span>
        <button
          className="gen-prompt"
          onClick={() => {
            if (openDecCount > 0 && chain.length > 0) {
              // Navigate to first blocking plan
              const firstBlocking = chain.find(slug => {
                const p = bySlug[slug];
                return (p?.decisions || []).some(d => !d.chosen);
              });
              if (firstBlocking) {
                window.location.hash = `#plan/${firstBlocking}`;
                return;
              }
            }
            setShowPrompt(true);
          }}
          disabled={chain.length === 0}
          title={openDecCount > 0 ? `${openDecCount} open decisions — resolve first` : "Generate path fleet prompt"}
          style={{ position: "relative" }}
        >
          <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M4 3h6l2 2v8H4z"/>
            <path d="M10 3v2h2"/>
            <path d="M6 7h4M6 9h4M6 11h2"/>
          </svg>
          Generate prompt
          {openDecCount > 0 && (
            <span className="resolve-badge" style={{ marginLeft: 4 }}>{openDecCount}</span>
          )}
        </button>
      </div>

      <div className="r-crit-chain">
        {chain.length === 0 && (
          <div style={{ color: "var(--muted)", padding: "12px 0", fontSize: 13 }}>No active or blocked plans with dependencies found.</div>
        )}
        {chain.map((slug, i) => {
          const p = bySlug[slug];
          if (!p) return null;
          return (
            <React.Fragment key={slug}>
              <a className="r-crit-card" href={`#plan/${slug}`}>
                <div className="r-crit-card-h">
                  <span className={`r-crit-dot ${p.status}`}></span>
                  <span className="r-crit-card-t">{p.title}</span>
                </div>
                <div className="r-crit-card-meta">
                  /{slug} · {p.ms || "—"}{p.sprint ? " · " + p.sprint : ""}
                </div>
                <div className="r-crit-card-bar">
                  <i style={{ width: `${Math.round((p.impl || 0) * 100)}%`, background: statusColor(p.status) }}></i>
                </div>
                <div className="r-crit-card-foot">
                  <span style={{ color: statusColor(p.status) }}>{p.status}</span>
                  <span style={{ flex: 1 }}></span>
                  <span>{Math.round((p.impl || 0) * 100)}%</span>
                </div>
              </a>
              {i < chain.length - 1 && <span className="r-crit-arr">→</span>}
            </React.Fragment>
          );
        })}
      </div>

      <div className="r-graph-grid">
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)", margin: "0 0 8px" }}>Other active &amp; blocked · {otherActive.length}</div>
          <div className="r-ck-list">
            {otherActive.map(p => (
              <a key={p.slug} className="r-ck-row" href={`#plan/${p.slug}`}>
                <span className={`r-ck-dot ${p.status}`}></span>
                <div className="r-ck-body">
                  <div className="r-ck-title">{p.title}</div>
                  <div className="r-ck-slug">/{p.slug} · {p.ms || "—"}</div>
                </div>
                <div className="r-ck-prog">
                  <span className="r-ck-bar"><i style={{ width: `${Math.round((p.impl || 0) * 100)}%`, background: statusColor(p.status) }}></i></span>
                  <span className="r-ck-pct">{Math.round((p.impl || 0) * 100)}%</span>
                </div>
                <span className="r-ck-arr">›</span>
              </a>
            ))}
            {otherActive.length === 0 && <div className="r-ck-empty">Everything in flight is on the critical path.</div>}
          </div>
        </div>
      </div>

      {showPrompt && (
        <PathPromptModal
          chain={chain}
          bySlug={bySlug}
          onClose={() => setShowPrompt(false)}
        />
      )}
    </div>
  );
}

window.GraphView = CriticalPathView;
window.CriticalPathView = CriticalPathView;
window.PathPromptModal = PathPromptModal;

// ─── Radial fan (plan view sub-mode) ─────────────────────────────────────

function RadialFan({ focalSlug, onNav }) {
  const M = window.STATE;
  if (!M) return null;
  const bySlug = Object.fromEntries(M.inventory.map(p => [p.slug, p]));
  const focal = bySlug[focalSlug];
  if (!focal) return <div style={{ padding: 24, color: "var(--muted)" }}>No plan selected.</div>;

  const MAX_PER_SIDE = 6;
  const deps = (focal.depends_on || []).map(s => bySlug[s]).filter(Boolean);
  const blocks = (focal.blocks || []).map(s => bySlug[s]).filter(Boolean);
  const depsOverflow = Math.max(0, deps.length - MAX_PER_SIDE);
  const blocksOverflow = Math.max(0, blocks.length - MAX_PER_SIDE);
  const depsShown = deps.slice(0, MAX_PER_SIDE);
  const blocksShown = blocks.slice(0, MAX_PER_SIDE);

  const { chain } = _criticalPath(M.inventory);
  const onPath = new Set(chain);

  function statusColor(s) {
    return s === "active" ? "var(--accent)" :
      s === "blocked" ? "var(--bad)" :
      s === "pending" ? "var(--warn)" :
      s === "shipped" ? "var(--good)" : "var(--muted)";
  }

  // Layout
  const W = 880, H = 420;
  const CX = W / 2, CY = H / 2;
  const R = 170;
  const focalR = 70;
  const satR = 44;

  function arcPos(items, side) {
    const n = items.length;
    if (n === 0) return [];
    const base = side === "left" ? Math.PI : 0;
    const span = Math.min(Math.PI * 0.85, Math.max(0.5, n * 0.28));
    return items.map((p, i) => {
      const t = n === 1 ? 0 : (i / (n - 1) - 0.5) * 2;
      const angle = base + t * (span / 2);
      return { plan: p, x: CX + Math.cos(angle) * R, y: CY + Math.sin(angle) * R };
    });
  }
  const left = arcPos(depsShown, "left");
  const right = arcPos(blocksShown, "right");

  return (
    <div className="r-fan">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="fanArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)"/>
          </marker>
          <marker id="fanArrC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)"/>
          </marker>
        </defs>

        {/* zone labels */}
        <text x={50} y={CY - 4} fontSize="10" fontFamily="var(--mono)" fill="var(--muted)" letterSpacing="0.10em">DEPENDS ON</text>
        <text x={50} y={CY + 14} fontSize="9" fontFamily="var(--mono)" fill="var(--faint)">{deps.length} {deps.length === 1 ? "plan" : "plans"}</text>
        <text x={W - 120} y={CY - 4} fontSize="10" fontFamily="var(--mono)" fill="var(--muted)" letterSpacing="0.10em">BLOCKS</text>
        <text x={W - 120} y={CY + 14} fontSize="9" fontFamily="var(--mono)" fill="var(--faint)">{blocks.length} {blocks.length === 1 ? "plan" : "plans"}</text>

        {/* edges */}
        {left.map((n, i) => {
          const isCrit = onPath.has(n.plan.slug) && onPath.has(focalSlug);
          const dx = CX - n.x, dy = CY - n.y;
          const d = Math.hypot(dx, dy);
          const ux = dx / d, uy = dy / d;
          return (
            <line key={"el" + i}
              x1={n.x + ux * satR} y1={n.y + uy * satR}
              x2={CX - ux * focalR} y2={CY - uy * focalR}
              stroke={isCrit ? "var(--accent)" : "var(--hair)"}
              strokeWidth={isCrit ? 2.5 : 1.5}
              markerEnd={isCrit ? "url(#fanArrC)" : "url(#fanArr)"}/>
          );
        })}
        {right.map((n, i) => {
          const isCrit = onPath.has(n.plan.slug) && onPath.has(focalSlug);
          const dx = n.x - CX, dy = n.y - CY;
          const d = Math.hypot(dx, dy);
          const ux = dx / d, uy = dy / d;
          return (
            <line key={"er" + i}
              x1={CX + ux * focalR} y1={CY + uy * focalR}
              x2={n.x - ux * satR} y2={n.y - uy * satR}
              stroke={isCrit ? "var(--accent)" : "var(--hair)"}
              strokeWidth={isCrit ? 2.5 : 1.5}
              markerEnd={isCrit ? "url(#fanArrC)" : "url(#fanArr)"}/>
          );
        })}

        {/* satellites */}
        {[...left, ...right].map((n, i) => {
          const onP = onPath.has(n.plan.slug);
          return (
            <g key={"sat" + n.plan.slug} transform={`translate(${n.x},${n.y})`}
               style={{ cursor: "pointer" }}
               onClick={() => onNav({ view: "plan", slug: n.plan.slug })}>
              {onP && <circle r={satR + 4} fill="var(--accent)" opacity="0.08"/>}
              <circle r={satR} fill="#fff" stroke={onP ? "var(--accent)" : "var(--line)"} strokeWidth={onP ? 2 : 1}/>
              <circle r={satR} fill="none" stroke={statusColor(n.plan.status)} strokeWidth="3"
                      strokeDasharray={`${(n.plan.impl || 0) * 2 * Math.PI * satR} ${2 * Math.PI * satR}`}
                      transform="rotate(-90)"/>
              <text y={-4} textAnchor="middle" fontSize="11" fontWeight="600" fill="var(--ink)">
                {n.plan.title.length > 14 ? n.plan.title.slice(0, 14) + "…" : n.plan.title}
              </text>
              <text y={10} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill="var(--muted)">
                {n.plan.ms || "—"} · {Math.round((n.plan.impl || 0) * 100)}%
              </text>
            </g>
          );
        })}

        {/* overflow pills */}
        {depsOverflow > 0 && (
          <g transform={`translate(${CX - R - 100},${CY + 90})`}>
            <rect x="0" y="0" width="80" height="22" rx="11" fill="var(--bg-2)" stroke="var(--hair)"/>
            <text x="40" y="15" textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--muted)">+{depsOverflow} more</text>
          </g>
        )}
        {blocksOverflow > 0 && (
          <g transform={`translate(${CX + R + 20},${CY + 90})`}>
            <rect x="0" y="0" width="80" height="22" rx="11" fill="var(--bg-2)" stroke="var(--hair)"/>
            <text x="40" y="15" textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--muted)">+{blocksOverflow} more</text>
          </g>
        )}

        {/* focal */}
        <g transform={`translate(${CX},${CY})`}>
          {onPath.has(focalSlug) && <circle r={focalR + 6} fill="var(--accent)" opacity="0.10"/>}
          <circle r={focalR} fill="#fff" stroke="var(--ink)" strokeWidth="1.5"/>
          <circle r={focalR} fill="none" stroke={statusColor(focal.status)} strokeWidth="4"
                  strokeDasharray={`${(focal.impl || 0) * 2 * Math.PI * focalR} ${2 * Math.PI * focalR}`}
                  transform="rotate(-90)"/>
          <text y={-12} textAnchor="middle" fontSize="14" fontWeight="600" fill="var(--ink)">
            {focal.title.length > 24 ? focal.title.slice(0, 24) + "…" : focal.title}
          </text>
          <text y={6} textAnchor="middle" fontSize="10" fontFamily="var(--mono)" fill="var(--muted)">
            /{focal.slug}
          </text>
          <text y={22} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill={statusColor(focal.status)}>
            {focal.status} · {Math.round((focal.impl || 0) * 100)}%
          </text>
        </g>
      </svg>

      {(deps.length === 0 && blocks.length === 0) && (
        <div className="r-fan-empty">No direct dependencies or blocks. This plan stands alone.</div>
      )}
    </div>
  );
}
window.RadialFan = RadialFan;
