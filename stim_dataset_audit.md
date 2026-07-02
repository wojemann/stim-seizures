# Consistency Audit — Scientific Data Resubmission

**Purpose:** Cross-check the revised manuscript, the response-to-reviewers, and the updated open dataset for internal consistency before resubmission. This document only *records* discrepancies — no source files were modified. Adjudicate each item in your own copies.

**Sources compared**
- **Manuscript (new):** `Seizure annotation open dataset - Google Docs.pdf` (repo root)
- **Response to reviewers:** `stim_dataset_revisions.pdf` (repo root)
- **Preprint (old) + supplement:** `~/local_data/stim_dataset_data/seizure annotation open dataset preprint.docx`, `…preprint supplement.docx`
- **Dataset (updated):** `~/local_data/stim_dataset_data/RAW_DATA/DATA/*` (participants.tsv, annotations.tsv, *.json, per-subject `ieeg/` and `derivatives/`)
- **Analysis code:** `code/analyzing_annotator_reliability.ipynb`, `code/annotation_analysis_and_consensus.ipynb`

**Method:** text extracted from all PDFs/DOCX; dataset tables parsed directly; reliability statistics (κ), timing (MAD), and Table 1 numbers independently recomputed from the released `annotations.tsv` + `channels.tsv` sidecars and `participants.tsv`. See the [Recomputation appendix](#recomputation-appendix).

---

## Summary

| # | Severity | Discrepancy | Sources in conflict |
|---|----------|-------------|---------------------|
| D1 | Critical | Seizure count 82 (data) vs 83 (manuscript/participants); spontaneous 45 vs 46 | data ↔ manuscript ↔ participants.tsv |
| D2 | Critical | License is `CC-BY-NC-SA 4.0` in dataset, but response promises **CC-BY** (journal requires CC0/CC-BY) | dataset ↔ response ↔ editor requirement |
| D3 | Critical | Headline reliability (abstract κ, Fig 5) is **HUP-only (N=63)** but the disclosing sentence was deleted | manuscript ↔ code ↔ old preprint |
| D4 | Major | Orphan subject `sub-CHOP032` on disk, not in participants.tsv/annotations.tsv/cohort | dataset ↔ manuscript |
| D5 | Major | Figure numbering broken (two "Figure 5"; onset-time fig is Fig 6 and Fig 7; no "Figure 7" caption) | manuscript internal |
| D6 | Major | Neurostimulation-device patients: manuscript says **3**, dataset has **4**; "10 no surgery" vs 9 unavailable | manuscript ↔ dataset |
| D7 | Major | Live placeholders in manuscript (CHOP IRB `***`, dataset data-citation, Pennsieve links) | manuscript ↔ editor requirements |
| D8 | Major | Response letter still has placeholders / internal to-do notes ("Still need…", "To be done", `***`, `****`) | response internal |
| D9 | Moderate | Annotator count: "three" (Background, Fig 1) vs "three or more" vs "three to five" | manuscript internal |
| D10 | Moderate | Figure N's (63 HUP / 21 CHOP) don't match released annotations.tsv (62 HUP / 20 CHOP); CHOP onset 21 vs spread 20 | manuscript ↔ dataset |
| D11 | Moderate | "gold-standard" retained (justified via FDA) rather than softened per R1 Major 4 | manuscript ↔ response ↔ reviewer |
| D12 | Moderate | Only a binary single-rater `lvfa` field added; rhythmic-activity/discharges labels not included | dataset ↔ reviewer ask |
| D13 | Minor | Duplicate reference entries (6≡19, 20≡34) | manuscript internal |
| D14 | Minor | Orphan/uncited references (32 WaveNet, 33 Taxonomy) after model citation change | manuscript internal |
| D15 | Minor | Acknowledgements list grants for non-authors (RTS, KAD) | manuscript ↔ author list ↔ dataset funding |
| D16 | Minor | `lvfa` (dataset/manuscript) vs "LVFA" (response); casing | manuscript ↔ response |
| D17 | Minor | De-identification: exact ages/age-at-onset now released (previously binned) | dataset ↔ old supplement |
| D18 | Minor | Benchmarking AUC/AUPRC not independently recomputed; model citation changed (28/29 → 30) | manuscript (note) |

---

## Critical

### D1 — Seizure count does not reconcile (82 vs 83; 45 vs 46 spontaneous)
- **Manuscript** (Abstract, Background, Methods–Participants, Table 1): "83 seizures (46 spontaneous, 37 stimulation-induced)."
- **participants.tsv:** `n_seizures` sums to **83**, `n_stim_induced` sums to 37 → 46 spontaneous.
- **annotations.tsv (ground truth):** **82** rows = 45 Spontaneous + 37 Stim. Induced.

Root cause — two per-patient errors in **participants.tsv** that offset in the stim total (both 37) but not the spontaneous/overall total:

| Patient | participants.tsv | annotations.tsv + EDFs on disk | Issue |
|---|---|---|---|
| **HUP261** | n_seizures=2, n_stim_induced=1 | 1 seizure only (spontaneous, `task-ictal271899`); no stim EDF | Counts a stim seizure that isn't in the released data |
| **HUP249** | n_seizures=4, n_stim_induced=1 | 4 EDFs = 2 `stim` + 2 `ictal`; annotations = 2 stim + 2 spont (`annotations.json` even documents the 2nd stim at onset 439029.32 as high-frequency) | `n_stim_induced` should be **2**, not 1 |

- **Suggested resolution:** treat `annotations.tsv` + EDFs as ground truth (82 seizures, 45 spontaneous, 37 stim). Either (a) fix participants.tsv (HUP261 → `n_seizures=1, n_stim_induced=0`; HUP249 → `n_stim_induced=2`) and change every "83"→"82" and "46 spontaneous"→"45 spontaneous" in the manuscript/Table 1 (the 55%/45% split is unchanged), or (b) if HUP261's stim seizure genuinely exists, restore its EDF + annotation row. The κ/timing figures were generated on the pre-correction set (see D10) and would need regeneration.

### D2 — Dataset license conflicts with what was promised to the editor
- **Editor requirement:** "Data need to be shared under a CC0 or CC-BY licence."
- **Response to reviewers** (editor comment + R1 Major 1): "We now share data under a CC-BY license." / "Data from both centers is shared under a CC-BY license."
- **Dataset:** `dataset_description.json` → `"License": "CC-BY-NC-SA 4.0"`.

`CC-BY-NC-SA` is **not** CC-BY — it adds NonCommercial and ShareAlike restrictions the journal does not permit. This can trigger a desk hold/reject because it was an explicit editorial gate.
- **Suggested resolution:** decide the intended license. If CC-BY, change `dataset_description.json` and the Pennsieve dataset page to `CC-BY 4.0`. If CC-BY-NC-SA is intended, the response to the editor is inaccurate and the journal must be contacted.

### D3 — Primary reliability analysis is HUP-only (N=63), but the disclosure was deleted
- **Manuscript (Abstract):** "Inter-rater agreement was κ = 0.64 (onset) / 0.62 (spread). Individual rater agreement with consensus was κ = 0.81 / 0.80." Presented as the dataset's reliability. Fig 5B reports "N = 63 seizures"; Fig 5C splits stim N=21 / spontaneous N=42 (= 63).
- **Code** (`analyzing_annotator_reliability.ipynb`, CELL 17): the cell that produces Fig 5B/C/D and the MAD analysis filters `consensus_annots[... 'HUP' in x]` — **HUP only** (comment: *"Comment out to generate figure comparing centers"*).
- **Old preprint (Technical Validation)** disclosed this: *"Because our annotators were experts in adult epilepsy, we report validation of expert annotations on patients from the adult epilepsy center in these primary analyses…"*; supplement Fig S2: *"we primarily focused our technical validation findings on solely the HUP patients."* **The new manuscript removed this sentence** but kept the HUP-only N and κ.
- **Recomputation (independent):** the reported κ reproduce only under the HUP-only filter. Whole-dataset values are meaningfully lower:

  | | Manuscript (N=63) | Recomputed HUP-only (N=62) | Recomputed ALL 82 |
  |---|---|---|---|
  | Interrater onset / spread | 0.64 / 0.62 | 0.655 / 0.625 | **0.612 / 0.593** |
  | Consensus onset / spread | 0.81 / 0.80 | 0.821 / 0.803 | **0.796 / 0.781** |

- **Why it matters:** a reader cannot tell why N=63 (< 82/83), and the abstract attributes HUP-only agreement to the full cohort. R1 Major 3 asked to *summarize* the HUP vs CHOP difference in the main text (done via the center-comparison figure), but that does not license dropping the scope statement for the primary analysis.
- **Suggested resolution:** restore a sentence stating the primary reliability analysis (Fig 5, and the MAD/timing analysis) uses adult-center (HUP) seizures only (N=63), or recompute and report the all-cohort metrics (0.61/0.59/0.80/0.78) alongside the center breakdown, and reconcile the abstract.

---

## Major

### D4 — Orphan subject `sub-CHOP032`
- **Dataset:** `sub-CHOP032/` exists with only `derivatives/electrodes.tsv` + `electrodes.json` (CHOP format); **no** `ieeg/` data. It is **not** in `participants.tsv` or `annotations.tsv`, and is not among the 32 patients.
- **Manuscript:** cohort is 32 patients (19 HUP / 13 CHOP); CHOP032 is not referenced.
- **Suggested resolution:** remove `sub-CHOP032/` from the released dataset (or, if it should be included, add its ieeg data + participants/annotations rows and update all counts).

### D5 — Figure numbering is broken (fallout from the SI merge/reorder)
Captions present, in order: Figure 1, 2, 3, 4, **5**, **5 (again)**, **6**. In-text references include **Fig. 6**, **Fig. 7A/7C/7D**, and (inconsistently) **Fig. 6A** for the same onset-time figure.

| Content | Caption says | Text refers to it as | Should be |
|---|---|---|---|
| Pipeline | Figure 1 | Fig. 1 | 1 ✓ |
| Onset annotation | Figure 2 | Fig. 2 | 2 ✓ |
| File manifest | Figure 3 | Fig. 3 | 3 ✓ |
| Benchmarking | Figure 4 | Fig. 4A–C | 4 ✓ |
| Channel reliability | Figure 5 | Fig. 5A–D | 5 ✓ |
| Center comparison (HUP vs CHOP) | **Figure 5** (dup) | **Fig. 6** | **6** |
| Onset-time reliability (MAD) | **Figure 6** | **Fig. 7A/7C/7D** and once **Fig. 6A** | **7** |

There is **no "Figure 7" caption**, yet "Fig. 7" is referenced 4×.
- **Suggested resolution:** renumber the last two captions to **Figure 6** (center comparison) and **Figure 7** (onset-time reliability); make all in-text references to the onset-time figure "Fig. 7"; keep "Fig. 6" for the center comparison.

### D6 — Neurostimulation-device patient count (3 vs 4)
- **Manuscript (Participants):** "An additional **3** patients were treated with an implanted neurostimulation device… The remaining **10** patients had not received any surgery." (19 available + 3 + 10 = 32)
- **Dataset:** `outcome == "Device"` for **4** patients — CHOP010, CHOP024, CHOP041, CHOP046; **9** patients have `n/a` outcome. (19 + 4 + 9 = 32)
- **Suggested resolution:** change manuscript to "4 … device" and "9 … no surgery," or reclassify one dataset patient if 3/10 is correct.

### D7 — Live placeholders in the manuscript
- Methods–Participants: "Children's Hospital of Philadelphia (CHOP, **protocol # \*\*\***)" — real CHOP IRB number missing.
- Data Records, first sentence: "publically available at **(\<link to open dataset upon publication\>)**." The editor required a **data citation (DOI URL) in the reference list**, cited in the first Data Record sentence. No dataset data-citation reference exists (ref 18 is the Pennsieve *platform* paper, not the dataset). Note the old preprint had a concrete DOI (`10.26275/n1sw-dymc`) that is now a placeholder.
- Usage Notes: "downloaded from Pennsieve **(provide link to pennsieve website here)**."
- **Suggested resolution:** insert CHOP protocol #; add a dataset data-citation with DOI URL to the references and cite it in the first Data Record sentence; fill Pennsieve links.

### D8 — Response-to-reviewers still contains placeholders and internal notes
These will be visible to the editor/reviewers in the response file:
- "At CHOP, research was approved by the CHOP IRB (protocol # **\*\*\***)." (appears twice)
- "We have added this citation, DOI: **\*\*\*\***"
- End of R1 Major 2 response: internal note "**Still need:** final published dataset version, reviewer access instructions (?), and code used to generate figures"
- R1 Minor 12 response: "**To be done**" (package versions / environment file). Note the repo already has `requirements.txt`; the manuscript Code Availability could point to it to close this, but the response currently leaves it open.
- **Suggested resolution:** resolve each before submitting the response document.

---

## Moderate

### D9 — Annotator-count wording inconsistent (R1 Minor 2 only partially applied)
- "Three or more" — Abstract; Methods–Participants.
- "three to five" — Data Records (annotations.tsv description, "clinician" field).
- **"three board-certified epileptologists"** — Background & Summary (implies exactly three).
- **"Three independent expert epileptologists"** — Figure 1 caption (implies exactly three).
- **Data:** 72 seizures have 3 annotators, 10 have 5 (so "3 or 5").
- **Suggested resolution:** standardize (e.g., "three or five") in the Background and Figure 1 caption too.

### D10 — Figure N's don't match the released annotations.tsv
- Manuscript: Fig 5B "N = 63 seizures"; Fig 5C stim N=21 / spontaneous N=42; center figure "N = 63 HUP, **21** CHOP" (onset) and "63 HUP, **20** CHOP" (spread).
- Released annotations.tsv: HUP = **62** (21 stim, 41 spontaneous); CHOP = **20** (both onset and spread).
- So the figure Ns reflect the pre-correction analysis set (an extra HUP seizure ← the HUP261 issue in D1; 42 HUP spontaneous ← the HUP249 labeling in D1), and the CHOP onset N=21 vs spread N=20 is internally inconsistent (should both be 20).
- **Suggested resolution:** regenerate the figures on the finalized dataset and update all Ns; resolve D1 first.

### D11 — "gold-standard" retained rather than softened (R1 Major 4)
- Reviewer: soften "gold-standard" given meaningful inter-rater variability.
- Response: clarified (multi-expert consensus = FDA reference standard, now cited, ref 26) rather than softening.
- Manuscript: still "gold-standard consensus" (Background; Technical Validation) and "Multi-expert consensus is the gold standard for seizure annotation established by the FDA²⁶."
- **Suggested resolution:** either soften the wording or ensure the FDA citation (Ceribell 510(k), ref 26) genuinely supports "multi-expert consensus is the gold standard for seizure annotation"; the reviewer may re-raise.

### D12 — Only a binary, single-rater LVFA label added (R1 Minor 4)
- Reviewer: state whether LVFA, rhythmic activity, and discharges onset-pattern labels are included.
- Response/dataset: added a binary `lvfa` field only, "clinician-determined (EC)" — i.e., a single reviewer, post-hoc, not multi-expert.
- The removed SOP (old supplement §4.2) described annotators characterizing onset patterns as LVFA or [frequency] + [rhythmic activity / discharges]; those richer labels are not released.
- **Suggested resolution:** state explicitly that only LVFA *presence* (single-rater) is provided, and note that rhythmic-activity/discharges labels are not included (and why).

---

## Minor

### D13 — Duplicate reference entries
- Ref **6** ≡ Ref **19**: Ojemann et al., "Unsupervised seizure annotation and detection with neural dynamic divergence," 2026.02.15.26346325.
- Ref **20** ≡ Ref **34**: Ojemann et al., "Can electrical stimulation replace spontaneous seizures in epilepsy surgery?" 2025.08.29.25334082.
- **Suggested resolution:** deduplicate and renumber (fix in-text callouts, e.g., Usage Notes "19,34").

### D14 — Orphan (uncited) references after model-citation change
- Ref **32** (WaveNet, van den Oord) and Ref **33** (Revell, "A Taxonomy of Seizure Spread Patterns") appear only in the reference list. The Data Overview model description was shortened and now cites Ref **30** (Revell, "AI-Driven Mapping of Seizure Spread Patterns") instead of the old refs 28/29 (WaveNet/Taxonomy).
- **Suggested resolution:** remove refs 32/33 or cite them; confirm ref 30 is the correct citation for the single-channel WaveNet-style detector.

### D15 — Acknowledgements list grants for non-authors
- New manuscript funding adds "**RTS**: R01MH112847, R01NS112274" and "**KAD**: R01NS116504." Neither RTS nor KAD is in the author list (WKSO, DJZ, CVKS, JK, JJL, CA, BCK, SKK, EDM, NS, BL, EC). These were **not** in the old preprint's acknowledgements.
- `dataset_description.json` Funding omits R01MH112847, R01NS112274, P50HD105354.
- **Suggested resolution:** verify/attribute these grants (likely copied from another paper) and reconcile the two funding lists.

### D16 — `lvfa` vs "LVFA" casing
- Dataset/manuscript use lowercase `lvfa` (values `yes`/`no`); response letter wrote 'the "LVFA" field'. Cosmetic — align casing.

### D17 — De-identification: exact ages now released
- Old supplement patient table binned age (e.g., 40–49) and age_at_onset (e.g., 30–39). Current `participants.tsv` releases exact values (e.g., 42.8; onset 35).
- No age exceeds 89 (HIPAA Safe Harbor), so likely acceptable, but this is *less* aggregation than before, and the editor/R1 raised de-identification for the human iEEG data. The date-shift to 2000-01-01 is claimed in Methods but was not independently verified here (EDF headers not inspected).
- **Suggested resolution:** confirm exact ages are intended and compliant; spot-check EDF start dates.

### D18 — Benchmarking metrics not independently recomputed
- Fig 4 AUC (onset 0.83, spread 0.91) and AUPRC (onset 0.45, spread 0.52) require the model-prediction pipeline (`WAVENET_validation_analysis.py` + model outputs) and were **not** recomputed here. They are unchanged from the old preprint. (The old preprint had a typo "AUPRC (onset: 45 …)" now corrected to 0.45.)

---

## Recomputation appendix

Independent recomputation from the released dataset (Python: pandas / scikit-learn `cohen_kappa_score`; channel universe = per-seizure `channels.tsv` SEEG channels; majority-vote consensus = `sum ≥ n/2`, matching `annotation_analysis_and_consensus.ipynb`).

**Reliability κ (per-seizure mean, then mean over seizures):**

| Subset | Interrater onset / spread | Consensus onset / spread | Manuscript |
|---|---|---|---|
| HUP only (N=62) | 0.655 / 0.625 | 0.821 / 0.803 | **0.64 / 0.62 · 0.81 / 0.80** (N=63) ✓ |
| All seizures (N=82) | 0.612 / 0.593 | 0.796 / 0.781 | not reported |
| CHOP only (N=20) | 0.478 / 0.486 | 0.719 / 0.709 | — |
| HUP Stim (N=21) | — | 0.812 / 0.775 | 0.81 / 0.78 ✓ |
| HUP Spont (N=41) | — | 0.825 / 0.817 | 0.82 / 0.81 ✓ |
| Center consensus | — | HUP 0.821/0.803 · CHOP 0.719/0.709 | HUP 0.82/0.80 · CHOP 0.68/0.71 (onset CHOP slightly off) |

**Timing (onset-time MAD/range):** HUP-only median MAD = **0.224 s**, range < 1 s = 56%, range < 5 s = 90% (manuscript: 0.22 s, 53%, 87%) — reproduces best under HUP-only, confirming the timing analysis is also HUP-only.

**Table 1 (from participants.tsv):** all demographic/clinical values reproduce — Female 18 (56%), Male 14 (44%), age-at-onset median 14 (IQR 8–22), duration median 9 (IQR 2.8–13.3), unifocal 20 (63%), MTLE 19 (59%), lesional 15 (47%), outcomes available 19 (59%), Engel I 12 (63% of available). Divergences: Device = **4** (manuscript 3, see D6); seizures = **83/37/46** in participants.tsv but **82/37/45** in annotations.tsv (see D1).

**Conclusion:** the reported statistics are faithfully reproducible from the released data — the issues are (a) the undisclosed HUP-only scope of the primary metrics (D3) and (b) the participants.tsv / manuscript count errors (D1, D6), not fabricated numbers.

---

## Items verified as consistent / correctly addressed (for reference)
- Subdural grid/strip electrodes removed; text now sEEG-only (R1 Minor 1). ✓
- Outcome terminology corrected to Engel with decimal subclasses, `Device` level, `follow_up` descriptor, `n/a` coding — manuscript, Table 1, and `participants.json` agree (R1 Minor 5, R2 #5). ✓
- Engel I text/table reconciled to 12 (63%) of 19 — matches participants.tsv (R1 Minor 6, R2 #4). ✓
- Table 1 percentage denominator note added (R1 Minor 7). ✓
- Consensus/majority-voting subsection added; denominator = # annotators; odd rater counts → no ties (R1 Major 4). ✓ (matches code and data)
- Data Overview moved before Technical Validation and shortened/de-hypothesized (Editor; R1 Major 5). ✓
- Inclusion/exclusion (38 met criteria, 6 excluded → 32) added (R1 Major 6, R2 #2). ✓
- BIDS-validator statement + `.bidsignore` (contains annotations.tsv/json) + known deviations added (R1 Major 2). ✓
- Fig 3D model description clarified (fixed effect = annotator identity, OLS fallback) (R2 #1). ✓
- EDF↔annotations.tsv linking example added to Usage Notes (R1 Minor 11). ✓
- Electrode dictionary matches sidecars — CHOP: `label,x,y,z,matter,brain_area`; HUP: `label,mm_*,surfmm_*,vox_*,roi,roiNum`. ✓
- "59–61 Hz bandpass (notch)" typo → "bandstop" corrected. ✓
- Ethics/consent detail (IRB #821778 for HUP; pediatric parent/guardian consent + assent) added. ✓ (CHOP IRB # still a placeholder — D7)
