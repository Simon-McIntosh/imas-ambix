// Decision row — select a choice + rationale, commit with Update.
// Renamed from plan-decision.jsx; canonical namespace is window.reckon.

function Decision({ d, onUpdate }) {
  const [selected, setSelected] = useState(d.chosen || null);
  const [rationale, setRationale] = useState(d.rationale || "");
  const [editing, setEditing] = useState(false);
  const isTaken = !!(d.chosen || d.rationale);

  const commit = () => {
    if (!selected && !rationale.trim()) return;
    onUpdate(selected || null, rationale.trim());
    setEditing(false);
  };

  const canCommit = !!selected || rationale.trim() !== (d.rationale || "");

  return (
    <div className={`r-dec ${isTaken ? "taken" : ""} ${editing ? "editing" : ""}`}>
      <div className="h">
        <span className="key">{d.key}</span>
        <span className="title">{d.title}</span>
        {isTaken && !editing && <span className="when">✓ {d.when} · {d.by}</span>}
      </div>
      {isTaken && !editing ? (
        <div className="taken-summary">
          {d.chosen && <span className="chosen-tag">{d.chosen}</span>}
          {d.rationale && <span className="rat">{d.rationale}</span>}
          <span className="edit-link" onClick={() => setEditing(true)}>edit</span>
        </div>
      ) : (
        <div className="r-dec-form">
          <div className="ctx">{d.context}</div>
          <div className="choices">
            {d.choices.map(c => (
              <button
                key={c}
                className={`choice ${selected === c ? "selected" : ""}`}
                onClick={() => setSelected(selected === c ? null : c)}
                title="Click to select; click Update to commit"
              >
                {c}
              </button>
            ))}
          </div>
          <div className="rat-row">
            <input
              placeholder={isTaken ? "edit rationale and Update to overwrite" : "rationale (or free-form response if no option fits)"}
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
            />
            <button className="upd" onClick={commit} disabled={!canCommit && !isTaken}>
              {isTaken ? "Update" : "Take decision"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Export under both names — plan.jsx uses <Decision>, plan-tokenizers.jsx uses <DecisionRow>.
window.Decision = Decision;
window.DecisionRow = Decision;
