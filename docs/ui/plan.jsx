// Plan view — reading-first document with inline decisions and select-to-comment.
// No internal topbar/breadcrumb because the sidebar handles nav.
// The right rail has been removed; About/Graph/Comments live in the title bar
// row 2 and inline in the section thread.
//
// Row 2 of the plan title bar (rendered in shell.jsx) carries the plan meta.
// Inside the plan body there is a Reading | Graph toggle near the top.
// Graph mode replaces the article body with RadialFan; followup + decision
// sections are hidden in graph mode for cleaner focus.

function Plan({ slug, onNav }) {
  const M = window.STATE;
  if (!M) return null;
  const PG = M.plans[slug];
  if (!PG) return <div className="r-page">Plan "{slug}" not found.</div>;

  const isTokenizers = slug === "tokenizers";
  const P = isTokenizers ? (M.planTokenizers || PG) : PG;

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

  // Reading / Graph toggle — resets to Reading on each navigation (slug change)
  const [viewMode, setViewMode] = useState("reading");
  useEffect(() => {
    setViewMode("reading");
  }, [slug]);

  // Listen for "open prompt" event from the title bar
  useEffect(() => {
    const open = () => {
      const openDecs = decs.filter(d => !d.chosen);
      if (openDecs.length > 0) {
        if (window.flashSaved) window.flashSaved(`✗ ${openDecs.length} open decision${openDecs.length === 1 ? "" : "s"} — resolve them first`);
        return;
      }
      setShowPrompt(true);
    };
    window.addEventListener("reckon:open-prompt", open);
    return () => window.removeEventListener("reckon:open-prompt", open);
  }, [decs]);

  const initialComments = (stored.comments) || P.comments || {};
  const [comments, setComments] = useState(initialComments);

  const articleRef = useRef(null);
  const [sel, clearSel] = window.reckon.useSelectionToComment(articleRef, slug);

  const author = window.STATE?.projects?.[0]?.owner || "user";

  const onUpdateDec = (key, choice, rationale) => {
    const now = new Date().toISOString().slice(0, 16).replace("T", " ");
    setDecs(arr => arr.map(x => x.key === key ? { ...x, chosen: choice || "", rationale, when: now, by: author } : x));
    planSave(slug, {
      [`decisions.${key}`]: { choice: choice || null, rationale, when: now, by: author },
    });
    if (window.flashSaved) window.flashSaved(`${slug}.${key} → ${choice || "rationale saved"}`);
  };

  const addComment = (sectionId, body, quote) => {
    if (!body || !body.trim()) return;
    const now = new Date().toISOString().slice(0, 16).replace("T", " ");
    const c = {
      id: `c-${Date.now()}`,
      who: author,
      when: now,
      body: body.trim(),
      ...(quote ? { quote } : {}),
    };
    const next = { ...comments, [sectionId]: [...(comments[sectionId] || []), c] };
    setComments(next);
    planSave(slug, { [`comments.${sectionId}`]: next[sectionId] });
    if (window.flashSaved) window.flashSaved(`${slug}.comments.${sectionId} +1`);
  };

  return (
    <div className="r-page">
      {/* Reading / Graph toggle — sits above the article */}
      <div className="r-plan-mode-toggle">
        <button
          className={`r-mode-pill${viewMode === "reading" ? " active" : ""}`}
          onClick={() => setViewMode("reading")}
        >Reading</button>
        <button
          className={`r-mode-pill${viewMode === "graph" ? " active" : ""}`}
          onClick={() => setViewMode("graph")}
        >Graph</button>
      </div>

      {viewMode === "graph" ? (
        /* Graph mode — radial fan only; no article or followups */
        window.RadialFan
          ? <window.RadialFan focalSlug={slug} onNav={onNav} />
          : <div style={{ padding: 24, color: "var(--muted)" }}>Graph view loading…</div>
      ) : (
        /* Reading mode (default) */
        <article className="r-reading" ref={articleRef}>
          {isTokenizers ? (
            <TokenizersBody P={P} decs={decs} onUpdateDec={onUpdateDec} comments={comments} />
          ) : (
            <GenericBody PG={PG} decs={decs} onUpdateDec={onUpdateDec} comments={comments} />
          )}
        </article>
      )}

      {/* Floating comment button — only in reading mode */}
      {viewMode === "reading" && sel && !composingAt && (
        <button
          className="r-float-btn"
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

// ─── Generic data-driven body (sections[] + decisions inline at end) ─────

function GenericBody({ PG, decs, onUpdateDec, comments }) {
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
            <Decision key={d.key} d={d}
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

window.Plan = Plan;
window.GenericBody = GenericBody;
