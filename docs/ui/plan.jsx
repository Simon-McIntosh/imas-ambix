// v6 Plan view — reading-first document with inline decisions, select-to-comment,
// and a "Generate handoff prompt" button in the header. No internal topbar/breadcrumb
// because the sidebar handles nav.

function V6Plan({ slug, onNav }) {
  const M = window.STATE;
  if (!M) return null;
  const PG = M.plans[slug];
  if (!PG) return <div className="v6-page">Plan "{slug}" not found.</div>;

  // For tokenizers we use the rich hand-authored prose. For other plans the data-driven path.
  const isTokenizers = slug === "tokenizers";
  const P = isTokenizers ? (M.planTokenizers || PG) : PG;

  // Hydrate decisions from local persist overlay
  const stored = planLoad(slug) || {};
  const storedDec = stored.decisions || {};
  const initialDecs = (P.decisions || []).map(d => {
    const s = storedDec[d.key];
    if (!s) return d;
    return { ...d, chosen: s.choice || d.chosen || "", rationale: s.rationale ?? (d.rationale || ""), when: s.when || "", by: s.by || "" };
  });

  const [decs, setDecs] = useState(initialDecs);
  const [showPrompt, setShowPrompt] = useState(false);
  const [composingAt, setComposingAt] = useState(null);

  // v7 integration: listen for "open prompt" event from the title bar
  useEffect(() => {
    const open = () => {
      const openDecs = decs.filter(d => !d.chosen);
      if (openDecs.length > 0) {
        // Refuse — caller will see no modal. Toast explains why.
        if (window.flashSaved) window.flashSaved(`✗ ${openDecs.length} open decision${openDecs.length === 1 ? "" : "s"} — take them first`);
        return;
      }
      setShowPrompt(true);
    };
    window.addEventListener("v7-open-prompt", open);
    return () => window.removeEventListener("v7-open-prompt", open);
  }, [decs]);

  // Comments — section-anchored. Source: state JSON + local overlay.
  const initialComments = (stored.comments) || P.comments || {};
  const [comments, setComments] = useState(initialComments);

  const articleRef = useRef(null);
  const [sel, clearSel] = window.v6.useSelectionToComment(articleRef, slug);

  const onUpdateDec = (key, choice, rationale) => {
    const now = new Date().toISOString().slice(0, 16).replace("T", " ");
    setDecs(arr => arr.map(x => x.key === key ? { ...x, chosen: choice || "", rationale, when: now, by: "Simon McIntosh" } : x));
    planSave(slug, {
      [`decisions.${key}`]: { choice: choice || null, rationale, when: now, by: "Simon McIntosh" },
    });
    if (window.flashSaved) window.flashSaved(`${slug}.${key} → ${choice || "rationale saved"}`);
  };

  const addComment = (sectionId, body, quote) => {
    if (!body || !body.trim()) return;
    const now = new Date().toISOString().slice(0, 16).replace("T", " ");
    const c = {
      id: `c-${Date.now()}`,
      who: "Simon McIntosh",
      when: now,
      body: body.trim(),
      ...(quote ? { quote } : {}),
    };
    const next = { ...comments, [sectionId]: [...(comments[sectionId] || []), c] };
    setComments(next);
    planSave(slug, { [`comments.${sectionId}`]: next[sectionId] });
    if (window.flashSaved) window.flashSaved(`${slug}.comments.${sectionId} +1`);
  };

  const decByKey = (k) => decs.find(d => d.key === k);

  return (
    <div className="v6-page">
      <div className="v6-plan-head">
        <div>
          <h1>{P.title || PG.title}</h1>
          <div className="sub">
            <code>/{slug}</code>
            <span>·</span>
            <span className={`status ${PG.status}`}><span className="dot"></span><span>{PG.status}</span></span>
            <span>·</span>
            <span className="badge ms" style={{ background: "var(--bg-3)", padding: "1px 7px", borderRadius: 4, fontFamily: "var(--mono)", fontSize: 11 }}>{PG.ms}</span>
            {PG.sprint && <>
              <span>·</span>
              <a href={`#sprint/${PG.sprint}`} style={{ color: "var(--ink-2)", borderBottom: "1px dotted var(--line)" }}>{PG.sprint}</a>
            </>}
            <span>·</span>
            <span>{fmtPct(PG.impl)}</span>
          </div>
        </div>
        <div className="v6-plan-actions">
          <button className="btn" onClick={() => setShowPrompt(true)}>⌘ Generate handoff prompt</button>
        </div>
      </div>

      <div className="v6-plan-layout">
        <article className="v6-reading" ref={articleRef}>
          {isTokenizers ? (
            <V6TokenizersBody P={P} decs={decs} onUpdateDec={onUpdateDec} comments={comments} />
          ) : (
            <V6GenericBody PG={PG} decs={decs} onUpdateDec={onUpdateDec} comments={comments} />
          )}
        </article>

        <aside className="v6-rail">
          <V6PlanRail plan={PG} comments={comments} onNav={onNav} />
        </aside>
      </div>

      {/* Floating Comment button when selection is active */}
      {sel && !composingAt && (
        <button
          className="v6-float-btn"
          style={{ top: sel.top + 6, left: Math.min(sel.left - 60, window.innerWidth - 140) }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            setComposingAt(sel);
            clearSel();
            window.getSelection()?.removeAllRanges();
          }}
        >
          ¶ Comment
        </button>
      )}

      {composingAt && (
        <CommentPopover
          anchor={composingAt}
          onClose={() => setComposingAt(null)}
          onPost={(body) => {
            addComment(composingAt.sectionId, body, composingAt.quote);
            setComposingAt(null);
          }}
        />
      )}

      {showPrompt && (
        <PromptModal
          planSlug={slug}
          initialPrompt={buildHandoffPrompt(P, decs, P.followups || PG.followups || [])}
          onClose={() => setShowPrompt(false)}
        />
      )}
    </div>
  );
}

// ─── Generic data-driven body (sections[] + decisions at bottom) ─────────

function V6GenericBody({ PG, decs, onUpdateDec, comments }) {
  return (
    <>
      <p style={{ color: "var(--muted)", fontSize: 14 }}>{PG.summary}</p>
      {(PG.sections || []).map(s => (
        <React.Fragment key={s.id}>
          <h2 id={s.id}><span className="sec">{s.sec}</span>{s.h}</h2>
          <p>{s.body}</p>
          <SectionComments comments={comments[s.id]} />
        </React.Fragment>
      ))}

      {decs.length > 0 && (
        <>
          <h2 id="decisions"><span className="sec">§</span>Decisions</h2>
          {decs.map(d => (
            <V6Decision key={d.key} d={d}
              onUpdate={(choice, rat) => onUpdateDec(d.key, choice, rat)} />
          ))}
          <SectionComments comments={comments["decisions"]} />
        </>
      )}

      {(PG.followups_done || []).length > 0 && (
        <>
          <h2 id="log"><span className="sec">§</span>Log</h2>
          <div className="followup-log">
            {PG.followups_done.map(f => (
              <div className="item" key={f.id}>
                <div className="mark">✓</div>
                <div>
                  <div className="title">{f.title}</div>
                  <div className="meta">{f.written_by} → {f.resolved_by} · {f.written_at} → {f.resolved_at}</div>
                  <div className="outcome">{f.outcome}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

// ─── Right rail ───────────────────────────────────────────────────────────

function V6PlanRail({ plan, comments, onNav }) {
  const totalComments = Object.values(comments || {}).reduce((n, arr) => n + arr.length, 0);
  return (
    <>
      <div className="block">
        <h4>About</h4>
        <div className="kv">
          <span className="k">owner</span>
          <span className="v">{plan.owner || "—"}</span>
          <span className="k">milestone</span>
          <span className="v">{plan.ms}</span>
          {plan.sprint && <>
            <span className="k">sprint</span>
            <span className="v">
              <a onClick={() => onNav({ view: "sprint", sprint: plan.sprint })}
                 style={{ cursor: "pointer", borderBottom: "1px dotted var(--line)" }}>{plan.sprint}</a>
            </span>
          </>}
          <span className="k">last write</span>
          <span className="v" style={{ fontFamily: "var(--mono)", fontSize: 11.5 }}>{plan.last}</span>
          <span className="k">phase</span>
          <span className="v" style={{ fontSize: 12 }}>{plan.phase}</span>
        </div>
      </div>

      {((plan.depends_on?.length || 0) > 0 || (plan.blocks?.length || 0) > 0) && (
        <div className="block">
          <h4>Graph</h4>
          {plan.depends_on?.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--muted)", marginBottom: 4 }}>depends on</div>
              <div className="pills">
                {plan.depends_on.map(s => {
                  const target = window.STATE.inventory.find(i => i.slug === s);
                  const blocked = target?.status === "blocked";
                  return (
                    <a key={s} href={`#plan/${s}`} className={`pill ${blocked ? "blocked" : ""}`}>{s}</a>
                  );
                })}
              </div>
            </div>
          )}
          {plan.blocks?.length > 0 && (
            <div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--muted)", marginBottom: 4 }}>blocks</div>
              <div className="pills">
                {plan.blocks.map(s => <a key={s} href={`#plan/${s}`} className="pill">{s}</a>)}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="block">
        <h4>Comments · {totalComments}</h4>
        {totalComments === 0
          ? <div style={{ color: "var(--muted)", fontSize: 12.5, lineHeight: 1.5 }}>
              Select any text in the plan to add a comment. Persists to <code style={{ fontSize: 11 }}>{plan.slug}.json#comments</code>.
            </div>
          : (
            <div>
              {Object.entries(comments).map(([sid, arr]) => (
                <div key={sid} style={{ marginBottom: 10 }}>
                  <a href={`#${sid}`} style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--muted)", letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}
                     onClick={(e) => { e.preventDefault(); document.getElementById(sid)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>
                    § {sid} · {arr.length}
                  </a>
                  {arr.slice(-2).map(c => (
                    <div key={c.id} style={{ padding: "6px 10px", background: "var(--bg-2)", borderLeft: "2px solid var(--accent)", borderRadius: "0 6px 6px 0", marginTop: 4, fontSize: 12 }}>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>{c.who} · {c.when}</div>
                      <div style={{ marginTop: 3 }}>{c.body}</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )
        }
      </div>
    </>
  );
}

window.V6Plan = V6Plan;
window.V6GenericBody = V6GenericBody;
window.V6PlanRail = V6PlanRail;
