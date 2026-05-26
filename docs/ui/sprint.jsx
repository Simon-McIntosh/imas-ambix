// Sprint view — goal block, kanban with drag-drop hover feedback.

function SprintView({ sprintId, onNav }) {
  const M = window.STATE;
  if (!M) return null;
  const allSprints = M.sprints || [];

  const idx = useMemo(() => {
    const i = allSprints.findIndex(s => s.id === sprintId);
    return i >= 0 ? i : allSprints.findIndex(s => s.id === M.active_sprint_id);
  }, [sprintId, allSprints]);

  const sprint = allSprints[idx];
  if (!sprint) return <div className="plan-page">No sprint.</div>;

  const items = sprint.items.map(it => {
    const slug = typeof it === "string" ? it : it.slug;
    const justification = typeof it === "object" ? it.justification : null;
    const plan = M.inventory.find(p => p.slug === slug);
    return plan ? { ...plan, justification } : null;
  }).filter(Boolean);

  const [localStatus, setLocalStatus] = useState({});
  const [dragOver, setDragOver] = useState(null);
  const STATUS_TO_COL = { pending: "todo", draft: "todo", active: "doing", blocked: "doing", in_progress: "doing", shipped: "done", done: "done" };
  const COL_TO_STATUS = { todo: "pending", doing: "active", done: "shipped" };

  const cols = useMemo(() => {
    const g = { todo: [], doing: [], done: [] };
    for (const p of items) {
      const effectiveStatus = localStatus[p.slug] || p.status;
      const col = STATUS_TO_COL[effectiveStatus] || "doing";
      g[col].push({ ...p, _eff: effectiveStatus });
    }
    return g;
  }, [items, localStatus]);

  const onDragStart = (e, slug) => {
    e.dataTransfer.setData("text/plain", slug);
    e.dataTransfer.effectAllowed = "move";
    e.currentTarget.classList.add("dragging");
  };
  const onDragEnd = (e) => {
    e.currentTarget.classList.remove("dragging");
    setDragOver(null);
  };
  const onDrop = (e, colId) => {
    e.preventDefault();
    const slug = e.dataTransfer.getData("text/plain");
    if (slug) setLocalStatus(s => ({ ...s, [slug]: COL_TO_STATUS[colId] }));
    setDragOver(null);
  };

  return (
    <div className="plan-page wide">
      <div className="sprint-header">
        <div className="eyebrow">Sprint</div>
        <div className="sprint-switcher">
          <button className="nav-btn" disabled={idx <= 0} onClick={() => onNav({ view: "sprint", sprint: allSprints[idx - 1].id })}>←</button>
          <div className="current">
            <span className="id">{sprint.id}</span>
            <span className={`st ${sprint.status}`}>{sprint.status}</span>
          </div>
          <button className="nav-btn" disabled={idx >= allSprints.length - 1} onClick={() => onNav({ view: "sprint", sprint: allSprints[idx + 1].id })}>→</button>
          <span className="range">{sprint.starts} → {sprint.ends}</span>
        </div>
      </div>

      <div className="sprint-goal">
        <div className="lbl">Goal</div>
        <div className="theme">{sprint.theme}</div>
        {sprint.summary && <div className="summary">{sprint.summary}</div>}
      </div>

      <div className="kanban">
        {[
          { id: "todo",  title: "To do",  cards: cols.todo  },
          { id: "doing", title: "Doing",  cards: cols.doing },
          { id: "done",  title: "Done",   cards: cols.done  },
        ].map(col => (
          <div
            key={col.id}
            className={`kanban-col ${dragOver === col.id ? "drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(col.id); }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(null);
            }}
            onDrop={(e) => onDrop(e, col.id)}
          >
            <div className="col-h">
              <span>{col.title}</span>
              <span className="n">{col.cards.length}</span>
            </div>
            {col.cards.map(p => (
              <a
                key={p.slug}
                className={`kanban-card ${p._eff === "blocked" ? "blocked" : ""}`}
                href={`#plan/${p.slug}`}
                draggable
                onDragStart={(e) => onDragStart(e, p.slug)}
                onDragEnd={onDragEnd}
              >
                <div className="t">{p.title}</div>
                <div className="slug">/{p.slug}</div>
                {p.justification && <div className="just">{p.justification}</div>}
                <div className="progress">
                  <span className="bar"><i className={p._eff === "shipped" ? "shipped" : p._eff === "blocked" ? "blocked" : ""} style={{ width: `${Math.round((p.impl || 0) * 100)}%` }}></i></span>
                  <span style={{ minWidth: 28, textAlign: "right" }}>{Math.round((p.impl || 0) * 100)}%</span>
                </div>
                <div className="row">
                  <span className={`status ${p._eff}`}><span className="dot"></span>{p._eff}</span>
                  <span>·</span>
                  <span>{p.ms}</span>
                  {(p.dec_open || 0) > 0 && <><span>·</span><span style={{ color: "var(--warn)" }}>D {p.dec_open}</span></>}
                  {(p.blockers || 0) > 0 && <><span>·</span><span style={{ color: "var(--bad)" }}>! {p.blockers}</span></>}
                </div>
              </a>
            ))}
            {col.cards.length === 0 && (
              <div style={{ textAlign: "center", padding: "20px 0", color: "var(--muted)", fontSize: 12 }}>
                drop here
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

window.SprintView = SprintView;
