# Contribution table

<!--
HOW TO FILL THIS TABLE (this comment is not printed)

1. Exactly ONE page in the exported PDF. Check with
   .venv\Scripts\python.exe scripts\build_report.py --parts contrib   (or a trial export).
   Every core-report section and every OWNED extension must map to a member. The dash counts in the
   separator rows set the relative column widths (build_report.py and pandoc honour them); keep them.
2. Names, student IDs and Kaggle display names: only the members' own data, spelled as in the
   core-report title block and the extension headers. Never guess them. The Kaggle display name
   must match the one given in the Canvas checkpoint (Kaggle rule 1). Kaggle team name: as on Kaggle.
3. ID = the member's short id: the same as project.yaml team.members[].id, which the archive uses in
   its author fields (SSC_AUTHOR). Lead and Contributors give these IDs, not full names, so that the
   table stays on one page. Lead = the member who answers for the part and can explain every detail
   of it (Kaggle rule 6). Contributors = other members who really worked on it; a two-word note in
   brackets is fine in a few rows, e.g. "XY (re-ran E3)", but notes in every row push the table past
   one page. Write "-" rather than list someone who did not work on it.
4. AI use = row numbers of the table in registry/ai_usage.md (legend under the table).
   Keep it true. When a member re-ran, rewrote or checked AI output, say so here (e.g. "8, 10;
   rewritten by XY") AND in that row's "Verified by" cell in ai_usage.md. The AI work of 2026-11-01
   has rows 14-21 (legend). Later AI work needs a new row APPENDED at the end of ai_usage.md (never
   inserted: the numbers in this table are row positions); then cite its number here.
5. Team size (Kaggle rule 2: 1, 2 or 3 members). Each member owns ONE different extension, marked
   individually; a member without a reflection PDF gets zero for their extension.
   - 1 member:  keep 1 member row. Keep only the ONE extension row you own; delete the other three.
                Lead = you in every row; Contributors "-" (never "sole author": every extension draft
                was written by the AI, ai_usage rows 3, 8, 10).
   - 2 members: keep 2 member rows. Keep the TWO owned extension rows (two different extensions,
                one owner each); delete the other two.
   - 3 members: keep 3 member rows. Keep the THREE owned extension rows; delete the fourth.
   Extensions nobody owns are not graded as extensions. Their runs may only be cited inside the
   core report, as supporting evidence.
6. The owner's name must also stand under the title of the owned extension
   (reports/extensions/e#_*.md, the "Author:" line).
7. No placeholder may remain before export. build_report.py warns about any it finds, and the
   final check in docs/ASSIGNMENT_CHECKLIST.md searches this file for them. This comment contains
   none, so that search only finds what is still to be filled.
Pointers below were checked against the archive on 2026-11-01; background for each one is in
reports/reflections/FACTS.md. Rnnn = research/rnnn_*.py and its run folder(s) in experiments/.
Full run ids of the runs named in the table (a run name points to experiments/<date-time>_<name>):
  S003 20261022-050708_r003_gmm2_hv (config configs/final_s003_gmm2_hv.yaml; re-run
       20261022-212507_final_s003_gmm2_hv; R910 = 20261022-091211_r910_s003_gmm2_hv);
       S005 20261022-071234_r201_qda_block
  E1   20261022-064054_e1_validation_protocol, 20261022-064148_e1_validation_protocol_qda_pair
  E2   20261022-070248_e2_learning_curve, 20261022-070339_e2_learning_curve_qda_block,
       20261022-070445_e2_learning_curve_gmm
  E3   20261022-063908_e3_leakage        E4   20261022-071414_e4_feature_stability
Files: r440_journey.md = reports/tables/r440_journey.md; fix_round1.md = research/reports/fix_round1.md.
Decisions whose decider or team confirmation is still open: D001-D003, D007, D009-D015.
-->

**Team:** TODO_TEAM_NAME · **Kaggle team:** TODO

| Member | ID | Student ID | Kaggle display name | Owned extension |
|------------------------------|--------|--------------|----------------------|--------------------------|
| TODO Full Name 1 | TODO | TODO | TODO | TODO (E1 / E2 / E3 / E4) |
| TODO Full Name 2 | TODO | TODO | TODO | TODO |
| TODO Full Name 3 | TODO | TODO | TODO | TODO |

| Part | Lead | Contributors | Evidence (archive) | AI use |
|----------------------|--------|------------|----------------------------------------------|-----------|
| Core 1: Problem formulation | TODO | TODO | R001, R002 (structure); R150, R920 (generator); R901 (ids, order, duplicates) | 2, 3, 7, 14 |
| Core 2: Validation protocol | TODO | TODO | D002, D015, D010; `docs/VALIDATION_PROTOCOL.md`; E1; R900 (permutation nulls) | 1, 3, 7, 13, 14 |
| Core 3: Submission history | TODO | TODO | S001–S006; B001–B008 (pre-registered); `r440_journey.md` | 2, 4, 11, 13, 14 |
| Core 4: Final solution | TODO | TODO | S003 = run `r003_gmm2_hv`, config `final_s003_gmm2_hv.yaml`, re-run R910; finals S003 + S005 (D010, D013); rule basis D007, D014 | 2, 7, 12, 13, 14 |
| Core 5: Analysis | TODO | TODO | R921 (Bayes gap); R922, R425 (label noise); R936, R404 (replicas); R441 (label budget); R249; I001–I014 | 3, 7, 11, 14 |
| Use of generative AI; References | TODO | TODO | `registry/ai_usage.md`; `registry/references.md` | 1, 7, 9, 14 |
| E1: Validation protocol design | TODO | – | run `e1_validation_protocol` (+ `_qda_pair`); R300, R306, R940 | 3, 7, 8, 10, 21 |
| E2: Learning curve | TODO | – | run `e2_learning_curve` (+ `_qda_block`, `_gmm`); R304, R307, R941, R942 | 3, 7, 8, 10, 16, 21 |
| E3: Information leakage | TODO | – | run `e3_leakage`; S004, B005; R321, R944 | 3, 4, 7, 8, 10, 21 |
| E4: Stability of the feature ranking | TODO | – | run `e4_feature_stability`; R320, R943 | 3, 7, 8, 10, 16, 21 |
| Code and reproducibility | TODO | TODO | `src/ssc/`, `scripts/`, `colab/`, `tests/`; R910, R911; git since 2026-11-15 (D016) | 1, 2, 7, 12, 21 |
| Decisions, beliefs, ideas, journal | TODO | TODO | D001–D017 (11 not yet team-confirmed); B001–B008; I001–I014 | 1, 5, 9, 13, 18, 21 |
| Kaggle submissions | TODO | TODO | S001–S006, at most 3 per day (D008); final selection by D010 | 2, 4, 13 |
| Video (every member presents) | TODO | all members | `video/PLAN.md`, `SCRIPT.md`, `slides.pdf` | 1, 19 |
| Report assembly, page checks | TODO | TODO | `scripts/build_report.py` (prints page counts); `reports/README.md` | 10, 15, 20 |

*AI use = row of `registry/ai_usage.md`: 1 setup · 2 research day 1 · 3 research round 1, E1–E4 runs · 4 uploads · 5 registry entries · 6 archive tools · 7 verification R900–R945 · 8 extension drafts · 9 registry fixes · 10 extension fixes · 11 research round 2 · 12 code fixes · 13 D010, D013, S005–S006 · 2026-11-01: 14 core report · 15 PDF build · 16 E2/E4 figures · 17 this table, reflection facts · 18 rules, D014, docs · 19 video kit · 20 final check · 21 review fixes, D015 · 11-15: 22–23 git, GitHub, D016–D017 (code; decisions) · 11-23/24: 24 figures, fonts, 25 formal rewrite (all report parts). In these rows Claude Code did the operating work (code, all Colab runs, registry entries, extension drafts, uploads with authorisation). Lead and Contributors give member IDs. Rnnn = `research/rnnn_*.py` and its runs.*
