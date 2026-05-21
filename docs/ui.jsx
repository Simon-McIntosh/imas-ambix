// Shared UI primitives + chrome for the Plans & Progress prototype.
// Each page imports this via a <script type="text/babel" src="ui.jsx"> tag
// and uses the components attached to window.

const { useState, useEffect, useMemo } = React;

// ─── Header / chrome ────────────────────────────────────────────────────────

function Topbar({ project, active, allProjectsView }) {
  if (allProjectsView) {
    return (
      <div className="topbar">
        <a href="index.html" className="brand">
          <span className="mark">P</span>
          <span>Plans</span>
        </a>
        <div style={{ marginLeft: 6, color: "var(--muted)", fontSize: 13 }}>
          across all projects on this docs-server
        </div>
        <div className="right">
          <a href="implementation.html" className="dim">how this works ↗</a>
          <kbd>⌘K</kbd>
        </div>
      </div>
    );
  }
  return (
    <div className="topbar">
      <a href="project.html" className="brand" title="Project overview">
        <span className="mark">P</span>
        <span className="proj-name">{project || "—"}</span>
        <span className="switcher">▾</span>
      </a>
      <nav>
        <a href="inventory.html" className={active === "plans"     ? "active" : ""}>Plans</a>
        <a href="sprint.html"    className={active === "sprint"    ? "active" : ""}>Sprints</a>
        <a href="decisions.html" className={active === "decisions" ? "active" : ""}>Decisions</a>
      </nav>
      <div className="right">
        <a href="home.html" className="dim">all projects ↗</a>
        <a href="implementation.html" className="dim">handoff doc</a>
        <kbd>⌘K</kbd>
      </div>
    </div>
  );
}

function Crumbs({ items }) {
  return (
    <div className="crumbs">
      {items.map((it, i) => {
        const last = i === items.length - 1;
        return (
          <React.Fragment key={i}>
            {it.href && !last
              ? <a href={it.href}>{it.label}</a>
              : last
                ? <strong>{it.label}</strong>
                : <span>{it.label}</span>}
            {!last && <span className="sep">/</span>}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ─── primitives ─────────────────────────────────────────────────────────────

function Status({ s, label }) {
  return (
    <span className={`status ${s}`}>
      <span className="dot"></span>
      <span>{label ?? s}</span>
    </span>
  );
}

function Roi({ v }) {
  return (
    <span className={`roi ${v}`} title={`ROI ${v}`}>
      <i></i><i></i><i></i>
    </span>
  );
}

function Bar({ v, width = 80, size = "" }) {
  const pct = Math.max(0, Math.min(1, v));
  return (
    <span className={`bar ${size}`} style={{ width }}>
      <i style={{ width: `${pct * 100}%` }}></i>
    </span>
  );
}

function Stack({ s, a, p, b }) {
  const total = s + a + p + b;
  const pc = (x) => `${(100 * x / total).toFixed(1)}%`;
  return (
    <span className="stack">
      <i className="shipped" style={{ width: pc(s) }}></i>
      <i className="active"  style={{ width: pc(a) }}></i>
      <i className="pending" style={{ width: pc(p) }}></i>
      <i className="blocked" style={{ width: pc(b) }}></i>
    </span>
  );
}

function Heat({ data }) {
  return (
    <span className="heat">
      {data.map((v, i) => (
        <i key={i} className={v >= 4 ? "l4" : v >= 3 ? "l3" : v >= 2 ? "l2" : v >= 1 ? "l1" : ""}></i>
      ))}
    </span>
  );
}

function Spark({ data, w = 110, h = 28, fill = true }) {
  const max = Math.max(1, ...data);
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => `${i * step},${h - (v / max) * (h - 4) - 2}`);
  const linePts = pts.join(" ");
  const area = `0,${h} ${linePts} ${w},${h}`;
  return (
    <svg className="spark" width={w} height={h} style={{ display: "block" }}>
      {fill && <polyline className="area" points={area} />}
      <polyline className="line" points={linePts} />
      <circle className="dot" cx={(data.length - 1) * step} cy={h - (data[data.length - 1] / max) * (h - 4) - 2} r="2" />
    </svg>
  );
}

function Tag({ children, kind = "" }) {
  return <span className={`tag ${kind}`}>{children}</span>;
}

function Who({ name, bot }) {
  const initials = name.split(/[\/\- ]/).filter(Boolean)[0]?.slice(0, 2).toUpperCase() || "??";
  return (
    <span className={`who-pill ${bot ? "bot" : ""}`}>
      <span className="av">{bot ? "AI" : initials}</span>
      <span>{name}</span>
    </span>
  );
}

// Icons (inline SVG, currentColor)

function Icon({ name, size = 14 }) {
  const props = { width: size, height: size, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" };
  switch (name) {
    case "arrow":   return <svg {...props}><path d="M3 8h10M9 4l4 4-4 4"/></svg>;
    case "search":  return <svg {...props}><circle cx="7" cy="7" r="4.5"/><path d="M13 13l-2.5-2.5"/></svg>;
    case "open":    return <svg {...props}><path d="M5 11l6-6M11 5h-4M11 5v4"/></svg>;
    case "check":   return <svg {...props}><path d="M3 8.5l3.2 3L13 4.5"/></svg>;
    case "grip":    return <svg {...props}><circle cx="6"  cy="4" r="0.9" fill="currentColor" stroke="none"/><circle cx="6"  cy="8"  r="0.9" fill="currentColor" stroke="none"/><circle cx="6"  cy="12" r="0.9" fill="currentColor" stroke="none"/><circle cx="10" cy="4"  r="0.9" fill="currentColor" stroke="none"/><circle cx="10" cy="8"  r="0.9" fill="currentColor" stroke="none"/><circle cx="10" cy="12" r="0.9" fill="currentColor" stroke="none"/></svg>;
    case "filter":  return <svg {...props}><path d="M2 4h12M4 8h8M6 12h4"/></svg>;
    case "msg":     return <svg {...props}><path d="M3 4h10v7H7l-3 2.5V11H3z"/></svg>;
    case "block":   return <svg {...props}><circle cx="8" cy="8" r="5.5"/><path d="M4.5 4.5l7 7"/></svg>;
    case "dot":     return <svg {...props}><circle cx="8" cy="8" r="3" fill="currentColor" stroke="none"/></svg>;
    case "paper":   return <svg {...props}><path d="M4 2h6l2 2v10H4z"/><path d="M10 2v2h2"/><path d="M6 7h4M6 9h4M6 11h3"/></svg>;
    case "image":   return <svg {...props}><rect x="2" y="3" width="12" height="10" rx="1"/><circle cx="6" cy="7" r="1.3"/><path d="M2 11l3-3 4 4 2-2 3 3"/></svg>;
    case "link":    return <svg {...props}><path d="M6 9.5L9.5 6"/><path d="M5 11l-1 1a2.5 2.5 0 01-3.5-3.5l2-2A2.5 2.5 0 016 7"/><path d="M11 5l1-1a2.5 2.5 0 013.5 3.5l-2 2A2.5 2.5 0 0110 9"/></svg>;
    case "plan":    return <svg {...props}><rect x="3" y="3" width="10" height="10" rx="1"/><path d="M5 6h6M5 8h6M5 10h4"/></svg>;
    case "thread":  return <svg {...props}><path d="M3 4h10v6H7l-3 2.5V10H3z"/></svg>;
    case "dataset": return <svg {...props}><ellipse cx="8" cy="4" rx="5" ry="1.8"/><path d="M3 4v4c0 1 2.2 1.8 5 1.8s5-.8 5-1.8V4"/><path d="M3 8v4c0 1 2.2 1.8 5 1.8s5-.8 5-1.8V8"/></svg>;
    case "web":     return <svg {...props}><circle cx="8" cy="8" r="6"/><path d="M2 8h12M8 2c2 2 2 10 0 12M8 2c-2 2-2 10 0 12"/></svg>;
    case "copy":    return <svg {...props}><rect x="5" y="5" width="8" height="9" rx="1"/><path d="M3 11V3a1 1 0 011-1h7"/></svg>;
    case "send":    return <svg {...props}><path d="M2 8l12-5-5 12-2-5z"/></svg>;
    case "plus":    return <svg {...props}><path d="M8 3v10M3 8h10"/></svg>;
    case "kebab":   return <svg {...props}><circle cx="8" cy="3" r="1.2" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r="1.2" fill="currentColor" stroke="none"/><circle cx="8" cy="13" r="1.2" fill="currentColor" stroke="none"/></svg>;
    default: return null;
  }
}

Object.assign(window, {
  Topbar, Crumbs,
  Status, Roi, Bar, Stack, Heat, Spark, Tag, Who, Icon,
});

// ─── Persistence ──────────────────────────────────────────────────────────
//
// On localhost we write to localStorage immediately. This mirrors the
// dotfiles state.js semantics (writes are browser-local until promoted to the
// repo by editing the JSON and committing). The Persist module exposes a
// small API both plan.html and decisions.html call from chip clicks +
// rationale blurs.
//
// In production this thin shim is replaced by state.js's saveState().

(function () {
  // The prototype runs inside an editor iframe whose hostname isn't
  // necessarily "localhost". We always allow writes here; in production the
  // dotfiles state.js handles the readonly-on-Pages distinction by hostname.
  // Override per-session by setting `localStorage.__plans_readonly = "1"`.
  const isLocal = localStorage.getItem("__plans_readonly") !== "1";
  window.Persist = {
    isLocal,
    save(plan, patch) {
      if (!isLocal) return { ok: false, reason: "read-only on published pages" };
      const key = `imas-ambix:${plan}`;
      let prev = {};
      try { prev = JSON.parse(localStorage.getItem(key) || "{}"); } catch {}
      // Patch keys can be dotted (e.g. "decisions.plasma-decoder-finetune");
      // merge at the leaf.
      const next = { ...prev };
      for (const [k, v] of Object.entries(patch)) {
        const parts = k.split(".");
        let cur = next;
        for (let i = 0; i < parts.length - 1; i++) {
          cur[parts[i]] = cur[parts[i]] || {};
          cur = cur[parts[i]];
        }
        cur[parts[parts.length - 1]] = v;
      }
      next._updated = new Date().toISOString();
      localStorage.setItem(key, JSON.stringify(next));
      return { ok: true, key, _updated: next._updated };
    },
    load(plan) {
      const key = `imas-ambix:${plan}`;
      try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch { return {}; }
    },
  };

  // Tiny toast for "saved" confirmations.
  window.flashSaved = function (msg) {
    let t = document.getElementById("__saved-toast");
    if (!t) {
      t = document.createElement("div");
      t.id = "__saved-toast";
      t.style.cssText = "position:fixed;bottom:24px;right:24px;background:var(--ink);color:var(--bg);padding:8px 14px;border-radius:6px;font:500 12.5px/1.4 var(--mono);z-index:1000;opacity:0;transition:opacity 180ms;box-shadow:0 4px 12px rgba(0,0,0,0.15);pointer-events:none;";
      document.body.appendChild(t);
    }
    t.textContent = "✓ " + (msg || "saved to localStorage");
    t.style.opacity = "1";
    clearTimeout(window.__savedTimer);
    window.__savedTimer = setTimeout(() => { t.style.opacity = "0"; }, 1600);
  };
})();
