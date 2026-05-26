// v6 shared bits — comment popover, prompt modal, helpers.
// Made global on window so other Babel scripts can use them.

const { useState, useEffect, useMemo, useRef, useCallback } = React;

// ─── Helpers ────────────────────────────────────────────────────────────

function fmtPct(v) { return Math.round((v || 0) * 100) + "%"; }

function whenShort(s) {
  if (!s) return "";
  return s.length > 10 ? s.slice(5, 16) : s;
}

// ─── Persistence: comments + decisions on top of Persist.save ─────────

function planSave(slug, patch) {
  if (window.Persist && window.Persist.save) {
    return window.Persist.save(slug, patch);
  }
  // Fallback to localStorage
  try {
    const proj = window.Persist?.project || document.querySelector('meta[name="docs-project"]')?.content || "unknown";
    const key = `${proj}:${slug}`;
    const prev = JSON.parse(localStorage.getItem(key) || "{}");
    const next = { ...prev, ...patch, _updated: new Date().toISOString() };
    localStorage.setItem(key, JSON.stringify(next));
  } catch {}
}

function planLoad(slug) {
  if (window.Persist && window.Persist.load) {
    return window.Persist.load(slug) || {};
  }
  try {
    const proj = window.Persist?.project || document.querySelector('meta[name="docs-project"]')?.content || "unknown";
    return JSON.parse(localStorage.getItem(`${proj}:${slug}`) || "{}");
  } catch { return {}; }
}

// ─── Build handoff prompt ───────────────────────────────────────────────

function buildHandoffPrompt(plan, decisions, followups) {
  const locked = (decisions || []).filter(d => d.chosen);
  const open = (decisions || []).filter(d => !d.chosen);
  const lockedBlock = locked.length === 0
    ? "  (none yet)"
    : locked.map(d => `  ${d.key} → ${d.chosen}${d.rationale ? "\n      reason: " + d.rationale : ""}`).join("\n");
  const openBlock = open.length === 0
    ? "  (none)"
    : open.map(d => `  ${d.key} — ${d.title}`).join("\n");
  const next = (followups || [])[0];
  const deps = (plan.depends_on || []).length
    ? plan.depends_on.map(s => `  ${s}.json`).join("\n")
    : "  (none)";

  const comments = (plan.comments) || (planLoad(plan.slug)?.comments) || {};
  const sectionsWithComments = Object.entries(comments).filter(([_, arr]) => (arr || []).length > 0);
  const commentsBlock = sectionsWithComments.length === 0
    ? "  (none)"
    : sectionsWithComments.map(([sid, arr]) =>
        arr.map(c =>
          `  §${sid} · ${c.who} · ${c.when}\n` +
          (c.quote ? `      quote: "${c.quote.length > 200 ? c.quote.slice(0, 200) + "…" : c.quote}"\n` : "") +
          `      body: ${c.body}`
        ).join("\n")
      ).join("\n");

  return `Instructions
  Read every referenced plan in full before starting work. Develop the plan
  further as you go — refine sections, capture rationale inline, and inspect
  the code under the project tree when the spec is ambiguous. Locked
  decisions must be honoured. Open decisions should be surfaced first; you
  may resolve one yourself if and only if you record a clear rationale.
  Quote-anchored comments locate the user's intent inside the section's
  prose. This planning system is repo-agnostic; the canonical state lives in
  the `reckon` repository (formerly part of dotfiles).

Project: ${plan.project || "project"}
Plan:    ${plan.slug}
Status:  ${plan.status} · ${plan.phase || ""}

Intent
  ${plan.summary || plan.title}

State to read
  state/${plan.project || "<project>"}/${plan.slug}.json
${deps}

Locked decisions (honour these)
${lockedBlock}

Open decisions (surface, do not resolve unilaterally)
${openBlock}

Comments (anchored to sections — use the quote as locator)
${commentsBlock}

${next ? `Next-up

  ${next.title}

  ${next.body}
${next.blocked_by ? `  Blocked by: ${next.blocked_by.slug} — ${next.blocked_by.reason}` : ""}

Recommended skill
  ${next.recommends_skill || "/plan-implement " + plan.slug}
` : "Next-up\n  (no pending followup — propose one)\n"}
Done-when
  1. Land the work this prompt describes.
  2. Update the plan body / section the work touches.
  3. POST a new followup to ${plan.slug}.json#followups with what landed.
  4. Mark the current followup resolved with outcome.
`;
}

// ─── Prompt modal: EDITABLE textarea, auto-close on Copy ────────────────

function PromptModal({ initialPrompt, planSlug, onClose }) {
  const [text, setText] = useState(initialPrompt);
  const textareaRef = useRef(null);

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
            <div className="v6-eyebrow">Handoff prompt</div>
            <h3>{planSlug}</h3>
            <div style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 2 }}>
              Paste into a new Claude conversation. Edit before copying if you want to refine the framing.
            </div>
          </div>
          <button className="btn ghost" onClick={onClose}>Close · Esc</button>
        </div>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
        />
        <div className="foot">
          <span style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 11 }}>
            built from <code>{planSlug}.json</code> · {text.length} chars
          </span>
          <span style={{ flex: 1 }}></span>
          <button className="btn primary" onClick={copy}>Copy to clipboard</button>
        </div>
      </div>
    </div>
  );
}

// ─── Comment composer popover ───────────────────────────────────────────

function CommentPopover({ anchor, onClose, onPost }) {
  const [body, setBody] = useState("");
  const ref = useRef(null);

  // Click-outside closes
  useEffect(() => {
    function onDown(e) {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    }
    setTimeout(() => document.addEventListener("mousedown", onDown), 30);
    return () => document.removeEventListener("mousedown", onDown);
  }, [onClose]);

  const post = () => {
    if (!body.trim()) return;
    onPost(body.trim());
  };

  // Enter posts; Shift+Enter for newline
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      post();
    }
  };

  return (
    <div
      ref={ref}
      className="v6-pop"
      style={{
        top: anchor.top + 8,
        left: Math.max(20, Math.min(anchor.left - 200, window.innerWidth - 440)),
      }}
    >
      <div className="quote-strip">
        <span className="qm">¶</span>
        <span className="qt">"{anchor.quote.length > 130 ? anchor.quote.slice(0, 130) + "…" : anchor.quote}"</span>
      </div>
      <div className="row">
        <textarea
          autoFocus
          placeholder="Comment — Enter to post, Shift+Enter for newline"
          value={body}
          onChange={(e) => setBody(e.target.value)}
          onKeyDown={onKey}
          rows={1}
        />
        <button className="post-btn" onClick={post} disabled={!body.trim()} title="Post (Enter)">↵</button>
      </div>
      <div className="footer">
        <span>writes to <code>{anchor.planSlug}.json#comments.{anchor.sectionId}</code></span>
        <span style={{ flex: 1 }}></span>
        <span className="cancel" onClick={onClose}>Cancel · Esc</span>
      </div>
    </div>
  );
}

// ─── Selection-to-comment hook ──────────────────────────────────────────
//
// Returns [selection, clear] where selection is { quote, sectionId, top, left }
// or null. Listens for mouseup within rootRef.current; finds the nearest
// preceding h2[id] to anchor the comment to.

function useSelectionToComment(rootRef, planSlug) {
  const [sel, setSel] = useState(null);
  useEffect(() => {
    function onMouseUp() {
      setTimeout(() => {
        const s = window.getSelection();
        if (!s || s.isCollapsed || !s.toString().trim()) { setSel(null); return; }
        const range = s.getRangeAt(0);
        if (!range) { setSel(null); return; }
        let node = range.commonAncestorContainer;
        if (node.nodeType === 3) node = node.parentElement;
        if (!rootRef.current || !rootRef.current.contains(node)) { setSel(null); return; }
        // Walk back through the article to find the nearest preceding <h2 id="...">
        const headings = rootRef.current.querySelectorAll("h2[id]");
        const startRect = range.getBoundingClientRect();
        let lastAbove = null;
        headings.forEach(h => {
          const r = h.getBoundingClientRect();
          if (r.top <= startRect.top) lastAbove = h;
        });
        const sectionId = lastAbove?.id || "_top";
        setSel({
          quote: s.toString().trim(),
          sectionId,
          top: startRect.bottom + window.scrollY,
          left: startRect.right,
          planSlug,
        });
      }, 1);
    }
    document.addEventListener("mouseup", onMouseUp);
    return () => document.removeEventListener("mouseup", onMouseUp);
  }, [planSlug, rootRef]);
  return [sel, () => setSel(null)];
}

// ─── Inline section comment cluster ─────────────────────────────────────

function SectionComments({ comments }) {
  if (!comments || comments.length === 0) return null;
  return (
    <div className="v6-section-comments">
      {comments.map(c => (
        <div key={c.id} className="v6-inline-comment">
          <div className="meta">{c.who} · {c.when}</div>
          {c.quote && <div className="quote">"{c.quote}"</div>}
          <div>{c.body}</div>
        </div>
      ))}
    </div>
  );
}

Object.assign(window, {
  v6: {
    fmtPct, whenShort, planSave, planLoad,
    buildHandoffPrompt,
    PromptModal, CommentPopover, useSelectionToComment, SectionComments,
  },
});
