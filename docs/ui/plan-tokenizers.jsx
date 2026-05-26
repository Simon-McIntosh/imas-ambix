// v6 Tokenizers body — the rich hand-authored prose for the hero plan.
// Decisions inline under § 12. Same content as the current site, but
// rendered inside the v6 SPA reading layout.

function V6TokenizersBody({ P, decs, onUpdateDec, comments }) {
  const dByKey = (k) => decs.find(d => d.key === k);
  const sectionComments = (sid) => <SectionComments comments={comments[sid]} />;

  return (
    <>
      <p>
        <strong>Status:</strong> {P.phase} · depends on{" "}
        <a href="#plan/data-acquisition">data-acquisition</a> mirror (shipped) and{" "}
        <a href="#plan/compute">compute</a> (blocked on SDCC reservation).
      </p>
      <p>
        The Fusion World Model treats every diagnostic stream as a sequence of integer tokens drawn from
        a shared vocabulary. This plan fixes the per-modality tokenizer choice, the codebook layout, the
        token-id namespacing across modalities, the training data, and the persistence format so that
        downstream model code (<code>imas_ambix/model/wham.py</code>) and training code
        (<code>imas_ambix/train/loop.py</code>) consume a stable interface.
      </p>
      <p>
        The choice is biased toward off-the-shelf, Apache-2.0-licensed components. We do not invent a
        new tokenizer. If a chosen tokenizer underperforms, the fallback is documented inline; the
        public interface does not change when we swap implementations.
      </p>

      <h2 id="s1"><span className="sec">§ 1</span>Modality breakdown and tokenizer choices</h2>
      <ul>
        <li><strong>Visible camera</strong> (<code>camera_visible.*.image_raw</code>) — Open-MAGVIT2 (TencentARC). 2<sup>18</sup> LFQ codebook, 8× spatial / 4× temporal compression, Apache-2.0, rFID 0.39 @ 8× ImageNet.</li>
        <li><strong>IR camera</strong> (<code>camera_ir.*.image_raw</code>) — Open-MAGVIT2, same checkpoint, possibly fine-tuned on IR. v0 default is to share with visible; decision below at §12.</li>
        <li><strong>Magnetics (high rate)</strong> — PatchTST, patch length 64, stride 32. Channel-independent patches; fastest to retrofit; linear projection into token id space.</li>
        <li><strong>Low-frequency signals</strong> (4 kHz interpolated) — Chronos T5-small (Amazon). Scale-and-quantize to discrete token IDs; Apache-2.0; HF-native.</li>
        <li><strong>Scalar state / control actions</strong> — Learned embedding table per categorical field.</li>
      </ul>
      <div className="callout">
        <span className="lbl">Why off-the-shelf</span>
        Every chosen tokenizer is a stable, externally-maintained codebase with a permissive licence.
        We pay no maintenance cost on the tokenizer itself; we pay it on the protocol that lets us swap one out.
      </div>
      {sectionComments("s1")}

      <h2 id="s4"><span className="sec">§ 4</span>Token id namespacing</h2>
      <p>
        A single global vocabulary serves all modalities. Each tokenizer is assigned a contiguous id
        range; the model never sees the modality string, only the integer. Block boundaries are
        recorded in <code>BLOCKWEIGHTS</code> so the loader can apply block-weighted cross-entropy.
      </p>
      {sectionComments("s4")}

      <h2 id="s8"><span className="sec">§ 8</span>Open questions to revisit before Phase 2</h2>
      <ul>
        <li><strong>IR codebook sharing</strong> — gate is MAE 2× rbb; decision below at §12.</li>
        <li><strong>PatchTST passthrough vs real embedding</strong> — decision below at §12.</li>
        <li><strong>Equilibrium 2-D</strong> — currently continuous cross-attention; decision below at §12.</li>
      </ul>
      {sectionComments("s8")}

      <h2 id="s12"><span className="sec">§ 12</span>Decisions in flight</h2>
      <p style={{ color: "var(--muted)", fontSize: 13.5 }}>
        Click a choice to select it, then click <strong>Take decision</strong> (or just type a rationale and Update) to commit.
        Decisions POST to <code>state/{window.Persist?.project || "project"}/tokenizers.json#decisions</code>; follow-on agents read them on pickup.
      </p>
      {(decs || []).map(d => (
        <V6Decision key={d.key} d={d}
          onUpdate={(choice, rat) => onUpdateDec(d.key, choice, rat)} />
      ))}
      {sectionComments("s12")}
      {sectionComments("decisions")}

      <h2 id="s13"><span className="sec">§ 13</span>What landed — 2026-05-19 to 2026-05-21</h2>
      <ul className="ship">
        <li>§ 9 v0 tokenizer scaffold (<code>imas_ambix.tokenizer</code> protocol surfaces) <span className="sha">5a137fc</span></li>
        <li>§ 9.2 Chronos T5-small + PatchTST identity passthrough <span className="sha">7e1c4ab</span></li>
        <li>§ 10 BlockKind side data + block-weighted CE loss <span className="sha">3f04922</span></li>
        <li>§ 12.1 plasma-decoder fine-tune scaffold <span className="sha">5e7c5fc</span></li>
        <li>§ 13.3 IR camera codebook benchmark wiring <span className="sha">dec0082</span></li>
      </ul>
      {sectionComments("s13")}

      {(P.followups_done || []).length > 0 && (
        <>
          <h2 id="log"><span className="sec">§</span>Followup log</h2>
          <div className="followup-log">
            {P.followups_done.map(f => (
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

window.V6TokenizersBody = V6TokenizersBody;
