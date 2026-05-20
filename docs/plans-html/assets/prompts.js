/* ===================================================
   Ambix Plan Site — prompts.js
   Decision-capture → follow-on prompt builder
   =================================================== */

(function () {
  "use strict";

  /**
   * Each entry: function(choice, notes) -> string
   * choice: the selected radio value
   * notes:  free-text from the textarea (may be empty string)
   */
  const DECISIONS = {

    /* -------- 1: Drop charge_exchange from v0 training ------------------- */
    "drop-charge-exchange": function (choice, notes) {
      const header = "[decision: drop-charge-exchange-from-training]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("yes")
        ? "Drop charge_exchange entirely from the v0 training manifest.\n" +
          "Rationale: audit found ~50 % bit-pattern corruption in t_i and v_i\n" +
          "columns (values 10^26 – 10^38 K / m/s; physical ranges ≤ 30 keV / 10^7 m/s).\n" +
          "These values are 12–28 orders of magnitude beyond physical range and\n" +
          "represent genuine float-encoding defects in the FAIR-MAST CX ingestion."
        : choice.startsWith("keep")
        ? "Keep charge_exchange but apply a hard-clip mask for abs > 1e25.\n" +
          "The masked shots remain in the manifest but with CX channels zeroed out."
        : "Defer the CX decision until a full 11,573-shot audit is complete.";
      const notesBlock = notes
        ? "\nAdditional notes:\n" + notes + "\n"
        : "";
      const next =
        "\nPlease update plans/data-quality.md §5 (training-grade gate) and\n" +
        "plans/world-model-v0.md §3 (training data shape) accordingly, then\n" +
        "re-derive the training-grade-shots.json manifest " +
        (choice.startsWith("yes") ? "excluding CX." : "with the chosen CX handling.");
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 2: Open-MAGVIT2 plasma decoder fine-tune ------------------- */
    "plasma-decoder-finetune": function (choice, notes) {
      const header = "[decision: open-magvit2-plasma-decoder-finetune]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("yes-now")
        ? "Trigger decoder fine-tune immediately (before the 125M smoke run).\n" +
          "Requires the bulk-encode pass on the GPU node to have produced\n" +
          "~5 k rbb frames for the fine-tune set. Takes ~4–6 GPU-hours exclusive."
        : choice.startsWith("yes-if")
        ? "Trigger decoder fine-tune only if the measured baseline rFID > 5\n" +
          "on the 100-shot rbb benchmark (see plans/tokenizer-benchmarks.md §5.1).\n" +
          "If rFID ≤ 5, fine-tune is deferred to v1 as a quality improvement."
        : "Defer plasma decoder fine-tune to v1 entirely.\n" +
          "Proceed with the ImageNet pretrained decoder for v0 regardless of rFID.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/tokenizers.md §12.1 with this decision and\n" +
        "update plans/v0-runway.md §2 (ROI ranking row 5) to reflect the trigger.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 3: PatchTST real embedding — defer vs land in v0 ---------- */
    "patchtst-real-embedding": function (choice, notes) {
      const header = "[decision: patchtst-real-embedding]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("defer")
        ? "Defer real PatchTST embedding to v1.\n" +
          "Keep the identity passthrough (token ID 0, raw floats in metadata).\n" +
          "The patch-projection layer trains end-to-end in the WHAM trunk.\n" +
          "Trigger for revisit: Pearson r < 0.90 on magnetics channels after\n" +
          "the 125M smoke run."
        : "Land real PatchTST embedding (transformers.PatchTSTModel) in v0.\n" +
          "Requires updating signals.py, ShotTokenizer, and WHAM input pipeline.\n" +
          "Estimated: 1 Sonnet session. Must rerun the 51 tokenizer tests.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/tokenizers.md §12.2 with the chosen path and\n" +
        "plans/v0-runway.md §2 row 5 to reflect the scope decision.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 4: Equilibrium 2-D tokenizer architecture ----------------- */
    "equilibrium-2d-tokenizer": function (choice, notes) {
      const header = "[decision: equilibrium-2d-tokenizer]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("option-a")
        ? "Use Option A: reuse Open-MAGVIT2 at upsampled 256×256.\n" +
          "Upsample the 65×65 equilibrium grid → 256×256 bilinear, treat as\n" +
          "single-channel 'frame'. Reuses existing codebook; 0.5 Sonnet sessions."
        : choice.startsWith("option-b")
        ? "Use Option B: Cosmos-Tokenizer-DV at native 65×65.\n" +
          "Needs NVIDIA OML license check before publishing weights.\n" +
          "Separate registry block; new codebook; ~2–4 GPU-hours fine-tune."
        : "Keep equilibrium in v0 as continuous cross-attention tensor.\n" +
          "Tokenization deferred to v1 per current plans/world-model-v0.md §8.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/tokenizers.md §12.3 with the chosen architecture\n" +
        "and plans/world-model-v0.md §8 (anti-goals) if the decision is to\n" +
        "bring tokenization forward into v0 scope.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 5: IR camera codebook — share or separate ----------------- */
    "ir-camera-codebook": function (choice, notes) {
      const header = "[decision: ir-camera-codebook]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("share")
        ? "Share the Open-MAGVIT2 visible codebook for IR frames (v0 default).\n" +
          "Proceed without a separate registry block for IR."
        : "Allocate a separate registry block for IR and fine-tune a second\n" +
          "decoder on the 25 available rir shots.\n" +
          "Note: only 25 rir shots available — use MAE (not rFID) as primary\n" +
          "metric due to insufficient sample for reliable FID estimate.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/tokenizers.md §12.5 with the codebook decision.\n" +
        "If separate codebook chosen, also update plans/tokenizers.md §4\n" +
        "(token id namespacing) to extend the IR registry range.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 6: SLURM dedicated training reservation ------------------- */
    "slurm-training-reservation": function (choice, notes) {
      const header = "[decision: slurm-dedicated-training-reservation]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("file-now")
        ? "File the dedicated training reservation request with SDCC now.\n" +
          "Request body is in plans/compute.md §3.2.\n" +
          "Owner: <add your ITER account>. Stakeholder: Science Division (S. Pinches)."
        : choice.startsWith("exclusive-pause")
        ? "Continue with the current exclusive-pause strategy (stop DeepSeek V4-Flash\n" +
          "before training). No separate reservation needed for v0.\n" +
          "Protocol: ambix agent stop deepseek-v4-flash → train → restart serve."
        : "Schedule training around serving in available windows (weekends,\n" +
          "overnight 00:00–06:00 local). Announce in SDCC GPU / Ambix Teams channel.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/compute.md §3 and plans/v0-runway.md §2 row 12\n" +
        "to reflect the chosen compute strategy.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 7: 125M smoke training run timing ----------------------- */
    "smoke-training-timing": function (choice, notes) {
      const header = "[decision: 125m-smoke-training-run-timing]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("run-now")
        ? "Run the 125M smoke training now, concurrent with the L1-all download.\n" +
          "Use the already-encoded tokens from the 15 CPU-encoded shots (bulk-encode\n" +
          "CLI produced 9,380 tokens for shot 15085). The smoke run only needs ~10 shots.\n" +
          "This validates the FSDP training loop before the full corpus is ready."
        : "Wait for the L1-all download and full bulk-encode pass to complete.\n" +
          "Run the 125M smoke with the GPU-encoded rbb corpus (~3,000 shots).\n" +
          "Better signal-to-noise on the loss curve but 1–2 days later.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/v0-runway.md §4 (sequenced follow-up) to\n" +
        "reflect the chosen timing and add the smoke-run submission command.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 8: Demo shots final selection --------------------------- */
    "demo-shots-selection": function (choice, notes) {
      const header = "[decision: demo-shots-final-selection]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("30420-30421-m6")
        ? "Use shots 30420 + 30421 + one M6-era TBD (original plan).\n" +
          "M6-era shot to be picked from the probe manifest by pulse_schedule diversity.\n" +
          "Note: 30420/30421 appear in FAIR-MAST quickstart docs — good for continuity."
        : choice.startsWith("30420-30421-no-cx")
        ? "Use shots 30420 + 30421 + one M8/M9-era shot with confirmed clean CX\n" +
          "(no charge_exchange corruption in the audit). Selects cleaner test data\n" +
          "if CX is included in the signal stream for the demo."
        : "Defer demo shot selection until after the full audit surfaces\n" +
          "corpus-level campaign statistics. Pick then based on quality distribution.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/demo.md §3 with the pinned demo shot IDs.\n" +
        "Freeze the IDs once chosen — changing them invalidates v0 comparisons.";
      return header + "\n" + body + notesBlock + next;
    },

    /* -------- 9: Camera selection for v0 training -------------------- */
    "camera-selection-v0": function (choice, notes) {
      const header = "[decision: camera-selection-v0-training]\n" +
        "Chosen: " + choice + "\n";
      const body = choice.startsWith("rbb-only")
        ? "Use rbb (camera_center) only for v0 training.\n" +
          "Coverage: 9,527 level-1 shots (55.7 % of 17,111).\n" +
          "Rationale: widest coverage, centre-stack midplane view, most informative\n" +
          "for plasma shape and disruption precursors."
        : choice.startsWith("rbb-rba")
        ? "Use rbb + rba (camera_center + camera_lower) for v0 training.\n" +
          "rbb: 9,527 shots; rba: 6,155 shots. Overlap is the smaller set.\n" +
          "Adds divertor / lower-SOL view; each camera treated as independent sample."
        : "Use rbb + rba + rir (camera_center + camera_lower + camera_ir) for v0.\n" +
          "rir is rare (only 25 shots) — include for completeness but weight low.\n" +
          "Most diverse view set but adds IR codebook decision complexity.";
      const notesBlock = notes ? "\nAdditional notes:\n" + notes + "\n" : "";
      const next =
        "\nPlease update plans/demo.md §1 (camera choice) and\n" +
        "plans/world-model-v0.md §3 (training data shape) with the chosen\n" +
        "camera set, and update the level1-cameras.json manifest filter if needed.";
      return header + "\n" + body + notesBlock + next;
    },

  };

  // Expose globally
  window.DECISIONS = DECISIONS;
})();
