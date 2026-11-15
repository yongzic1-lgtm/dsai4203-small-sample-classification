# Video script: Small-Sample Classification Challenge

**Status:** draft, written 2026-11-01 by Claude Code (Anthropic, Claude Opus 5.5; workflow agent `video_kit`) from `reports/report_core.md`, the extension drafts in `reports/extensions/` and the archive. Before recording, every speaker rewrites their lines in their own words and must be able to explain each sentence (Kaggle rule 6). Say "I" only for work you did yourself; this draft uses "we" for the team together with its AI tool, which the opening discloses.

**Files:** slides `video/slides.pdf` (source `video/slides.tex`); recording, editing and compression `video/RECORDING.md`; hard limits and storyboard `video/PLAN.md`.

**Limits:** at most 5:00 whatever the team size (target 4:40); every member presents, with their face on camera while speaking; file at most 300 MB; English.

## How to use this script

- **Speakers.** A, B and C are the members; A opens. Each member presents the extension they own. An extension nobody owns is not presented on its own: switch its slide off in `slides.tex` (`\Eonefalse` etc., see `RECORDING.md` §2) and skip its block below.
- **Where the extension blocks go.** The E1 block follows the protocol slide (slide 5), as in `PLAN.md`. The E2, E3 and E4 blocks come after the submission journey (slides 8–10).
- **Lines marked "only if".** One protocol sentence is said only when nobody owns E1. Each extension block has one optional sentence, for teams that present two extensions or fewer.
- **Reading numbers.** "S001" = "S-one" … "S006" = "S-six"; "0.9992" = "point nine nine nine two"; "5-fold" = "five-fold"; "QDA", "BIC", "AUC" letter by letter.
- **Timing.** About 130 words per minute. Section 5 gives word counts and times measured on this text. Rehearse each segment with a timer; the slides carry more numbers than you need to say.
- **If a rehearsal runs long, cut in this order:** (1) the optional extension sentences; (2) the "only if nobody owns E1" sentence (slide 4 still cites the E1 result); (3) the second sentence on slide 12; (4) round the spoken decimals ("about 0.999" instead of "0.9992"); (5) shorter pauses at slide changes.

## 1. Who speaks when

The tables show the longest case for each team size. Times are a deliberately slow estimate: every number read out in full ("point nine nine nine two") and 1 s for each slide change. At an even 130 words per minute the same text takes about 45 s less (§5). Aim to finish by 4:40 and never pass 5:00.

### Three members (A, B, C)

Longest case: nobody owns E1, so the protocol slot includes its extra line, and E2, E3 and E4 are presented. If a member owns E1, the E1 block (slide 5) is spoken right after slide 4 instead; the whole video then ends about 10 s earlier.

| Time | Slides | Section | Speaker | Words |
|---|---|---|---|---|
| 0:00–0:20 | 1 | Opening: team, names, one-line result, AI disclosure | all (A leads; B and C say their names) | 44 |
| 0:20–0:53 | 2–3 | The problem | A | 59 |
| 0:53–1:24 | 4 | Validation protocol + the line for an unowned E1 | A, or the E1 owner | 63 |
| 1:24–2:09 | 6–7 | Submission journey, S001–S006 | B | 77 |
| 2:09–2:41 | 8 | E2 block | E2 owner | 54 |
| 2:41–3:13 | 9 | E3 block | E3 owner | 54 |
| 3:13–3:43 | 10 | E4 block | E4 owner | 58 |
| 3:43–4:34 | 11–13 | Final solution and the selection rule | C | 92 |
| 4:34–4:52 | 14 | One takeaway each | all (A, B, C, then A) | 32 |

With one extension block each, A speaks about 2:00, B about 1:20 and C about 1:30. For a more even split, give slide 4 to B. Every speaker is on camera while speaking.

### Two members (A, B)

Longest case: E3 and E4 owned (any pair without E1 is within 1 s of it). Each block includes its optional sentence.

| Time | Slides | Section | Speaker | Words |
|---|---|---|---|---|
| 0:00–0:19 | 1 | Opening (B says only their name) | A, B | 42 |
| 0:19–0:52 | 2–3 | The problem | A | 59 |
| 0:52–1:23 | 4 | Validation protocol + the line for an unowned E1 | A, or the E1 owner | 63 |
| 1:23–2:08 | 6–7 | Submission journey, S001–S006 | B | 77 |
| 2:08–2:50 | 9 | E3 block + optional sentence | E3 owner | 76 |
| 2:50–3:31 | 10 | E4 block + optional sentence | E4 owner | 76 |
| 3:31–4:22 | 11–13 | Final solution and the selection rule | B | 92 |
| 4:22–4:34 | 14 | One takeaway each | A, B, then A | 21 |

### One member

Longest case: E3 owned (E2 and E4 are within 1 s). One speaker throughout, with the solo lines of §3; there is no need to fill five minutes.

| Time | Slides | Section | Words |
|---|---|---|---|
| 0:00–0:17 | 1 | Opening (solo) | 37 |
| 0:17–0:50 | 2–3 | The problem | 59 |
| 0:50–1:21 | 4 | Validation protocol + the line for an unowned E1 | 63 |
| 1:21–2:06 | 6–7 | Submission journey, S001–S006 | 77 |
| 2:06–2:48 | 9 | E3 block + optional sentence | 76 |
| 2:48–3:39 | 11–13 | Final solution and the selection rule | 92 |
| 3:39–3:50 | 14 | Takeaways (solo) | 21 |

## 2. The script (three-member version)

### Slide 1: Opening (all), 0:00–0:20

<!-- seg:open -->
> **A:** Hi, we're team TODO_TEAM_NAME, Kaggle team TODO_KAGGLE_TEAM_NAME.
> **B:** I'm TODO_NAME_B.
> **C:** I'm TODO_NAME_C.
> **A:** And I'm TODO_NAME_A. Our model: two clusters, fitted without labels, on 13 high-variance features. One disclosure: the AI assistant Claude Code wrote most of our code, ran our experiments and drafted our texts.

Recording: gallery view or each face in turn; the slide may be hidden here.

### Slides 2–3: The problem (A), 0:20–0:53

<!-- seg:problem -->
> **A** *(slide 2)***:** The task: 250 labelled rows, 150 features, almost 20,000 unlabelled test rows; metric ROC AUC. With so few rows, every score is noisy, and the public leaderboard has only about 395 rows, so we aimed at the expected private score.
> **A** *(slide 3)***:** Without labels, 13 features stand out by their variance; most others are pure noise. That block holds the signal.

### Slide 4: Validation protocol (A, or the E1 owner), 0:53–1:24

<!-- seg:protocol -->
> **A:** Our official check is stratified 5-fold cross-validation, repeated 20 times. Anything that uses labels, feature selection included, is refitted inside each fold. Models are compared fold by fold; a win needs a gap above two standard errors and a corrected t-test. The public leaderboard is only a bug check.

Only if nobody owns E1 (otherwise the E1 block follows):

<!-- seg:protocol_e1 -->
> **A:** With 50 split seeds, it proved the cheapest scheme that resolves a 0.01 gap.

### Slides 6–7: Submission journey (B), 1:24–2:09

<!-- seg:journey -->
> **B** *(slide 6)***:** Each upload tested one prediction, written down first; all six landed within 1.4 standard errors.
> **B** *(slide 7)***:** S001, plain L1 logistic, matched its CV. S002, QDA on the high-variance block, jumped to 0.993: the signal is mainly in the covariances. S003, the label-free mixture, reached 0.9992; the rest is likely label noise. S004, the leakage test, matched honest CV; our proxy guess was too high. S005 and S006 came out higher than S003 by about 12 swapped pairs: noise.

### Extension blocks (each owner, about 25–40 s), 2:09–3:43

Only the extensions your team owns. The owner's name is on the slide ("Extension E# · name").

#### E1, slide 5 (spoken right after slide 4)

<!-- seg:e1 -->
> **E1 owner:** E1 asked which validation scheme can separate two models 0.01 AUC apart, using 50 split seeds per scheme. A 20% hold-out ranks such a pair correctly only 64% of the time, and more folds don't help. Repeats do: only 5-fold repeated 20 times keeps the detectable difference clearly below 0.01.

Optional:

<!-- seg:e1_opt -->
> **E1 owner:** Near the ceiling, with QDA, the noise is about five times smaller, so size the protocol for the hardest decision.

#### E2, slide 8

<!-- seg:e2 -->
> **E2 owner:** E2 asked how AUC grows with training size. The L1 baseline reaches 0.85 at 250 rows and is still rising; a power law predicts about 0.91 at 500. But its 90% interval held the truth in only 21% of replicas, so we quote plus or minus 0.04. The mixture scores 1.0 at every size.

Optional:

<!-- seg:e2_opt -->
> **E2 owner:** Each size had ten random subsets. QDA is nearly flat too: more labels would only help the weak model.

#### E3, slide 9

<!-- seg:e3 -->
> **E3 owner:** E3 asked how much a leaky feature filter inflates CV. Picking the 20 features most correlated with the label on all rows gives a CV of 0.867; picking them inside each fold gives 0.793. Our S004 upload scored 0.791, so the leak added about 0.07. Any step that uses labels belongs inside the fold.

Optional:

<!-- seg:e3_opt -->
> **E3 owner:** The gap is almost zero at 7 features, where every fold picks the same strong features, and largest at 30 to 50.

#### E4, slide 10

<!-- seg:e4 -->
> **E4 owner:** E4 asked whether stably selected features are worth keeping. Over bootstrap resamples, seven features almost always make the top 10, then the frequency drops sharply. They help a linear model, but QDA on them scores 0.961, against 0.996 on the block, because six block features carry their signal in the variance. Stable is not the same as useful.

Optional:

<!-- seg:e4_opt -->
> **E4 owner:** There were 2,000 resamples; with a linear model, the stable set adds 0.063 AUC over all 150 features.

### Slides 11–13: Final solution and selection rule (C), 3:43–4:34

<!-- seg:final -->
> **C** *(slide 11)***:** Our final model, S003, fits a two-component Gaussian mixture to the 13 high-variance features of all 20,000 rows, without labels; training labels only name the components. It scored 1.0 in all 100 CV folds.
> **C** *(slide 12)***:** BIC, using no labels, picks two components. Two labels already rank perfectly, and on 222 simulated data sets none beat it on average.
> **C** *(slide 13)***:** Our final picks follow a rule written before the last two scores: S003, plus S005, which never sees test features, in case those are ruled out. Expected private AUC: about 0.997, limited by label noise.

### Slide 14: Takeaways (all), 4:34–4:52

<!-- seg:close -->
> **A:** My takeaway: a 0.01 difference needs repeated, paired cross-validation.
> **B:** Mine: 395 public rows catch bugs; they can't rank good models.
> **C:** Mine: here, unlabelled rows were worth more than labels.
> **A:** Thanks for watching.

Recording: gallery view again, all faces. Each member may pick a different sentence from §4; the slide shows the three sentences set in `slides.tex` (`\TakeA`, `\TakeB`, `\TakeC`), so keep the two in step.

## 3. Changes for two members and for one member

**Two members (A, B).** Opening and closing as below; everything else as in §2, with B also presenting slides 11–13. Each extension block includes its optional sentence. In `slides.tex` leave `\NameC` and `\TakeC` empty.

<!-- seg:open_2 -->
> **A:** Hi, we're team TODO_TEAM_NAME, Kaggle team TODO_KAGGLE_TEAM_NAME.
> **B:** I'm TODO_NAME_B.
> **A:** And I'm TODO_NAME_A. Our model: two clusters, fitted without labels, on 13 high-variance features. One disclosure: the AI assistant Claude Code wrote most of our code, ran our experiments and drafted our texts.

<!-- seg:close_2 -->
> **A:** My takeaway: a 0.01 difference needs repeated, paired cross-validation.
> **B:** Mine: here, unlabelled rows were worth more than labels.
> **A:** Thanks for watching.

**One member.** Use these two blocks, present one extension block with its optional sentence, and change "we / our" to "I / my" in §2: "so I aimed" (slide 2), "My official check" (slide 4), "my proxy-based guess" (slide 7), "so I quote" (E2), "My S004 upload" (E3), "My final model" (slide 11), "My final picks" (slide 13). In `slides.tex` leave `\NameB`, `\NameC`, `\TakeB` and `\TakeC` empty, and change "What we take away" if you like.

<!-- seg:open_1 -->
> **A:** Hi, I'm TODO_NAME_A, team TODO_TEAM_NAME on Kaggle. My model: two clusters, fitted without labels, on 13 high-variance features. One disclosure: the AI assistant Claude Code wrote most of my code, ran my experiments and drafted my texts.

<!-- seg:close_1 -->
> **A:** My takeaways: a 0.01 difference needs repeated, paired cross-validation, and here, unlabelled rows were worth more than labels. Thanks for watching.

## 4. Takeaway sentences to choose from

One sentence per member, in your own words, about something you really learned. Each one below is backed by the archive.

| Sentence | Evidence |
|---|---|
| A 0.01 difference needs repeated, paired cross-validation. | E1: only 5-fold × 20 has a detectable difference whose 95% CI lies below 0.01 |
| 395 public rows catch bugs; they can't rank good models. | Public SE 0.0015–0.02 (Hanley–McNeil); S005/S006 vs S003 |
| Here, unlabelled rows were worth more than labels. | R441: 2 labels for the mixture, 120 for strict QDA |
| Write the prediction down before the upload; it keeps you honest. | B001–B008; S004's guess was too high |
| Any step that uses labels belongs inside the fold. | E3: leak of 0.075 against the public score |
| A feature can be stable without being useful. | E4 and R320 |
| More labels help a weak model, not one at the limit. | E2: L1 still rising; mixture flat at 1.0 |
| When CV says 1.0 for every model, it can no longer rank them. | D011, R936: 222/222 replicas not significant |

## 5. Word counts and timing

Counted on the quoted lines above (2026-11-01). Words: every spoken token, speaker labels excluded; each TODO name counts as one word, so real names add a few. Strict estimate: numbers read out in full ("0.9992" = five words, "250" = three) at 130 words per minute, plus 1 s per slide change. Ranges run over which extensions the team owns.

| Team | What is spoken | Words | At 130 words/min | Strict estimate |
|---|---|---|---|---|
| 3 members | opening, problem, protocol (+ the E1 line if nobody owns E1), journey, 3 blocks, final, takeaways | 511–533 | 3:56–4:06 | 4:40–4:52 |
| 2 members | the same with 2 blocks, each with its optional sentence | 483–506 | 3:43–3:54 | 4:21–4:34 |
| 1 member | the same with 1 block and its optional sentence; solo opening and closing | 405–425 | 3:07–3:16 | 3:38–3:50 |

Every version stays under the 600-word ceiling of `PLAN.md`. The longest one (three members, nobody owning E1, 4:52) is the only one whose strict estimate passes 4:41; cut (2) of the list above brings it to about 4:45, and a normal pace is faster than the strict estimate. Counts updated 2026-11-01 after the review fixes on slides 7 and 12 (main session; +2 words).

| Segment | Words | Strict time |
|---|---|---|
| Opening (3 / 2 / 1 members) | 44 / 42 / 37 | 0:20 / 0:19 / 0:17 |
| Problem, slides 2–3 | 59 | 0:30 |
| Protocol, slide 4 | 49 | 0:23 |
| Protocol line when nobody owns E1 | 14 | 0:07 |
| Journey, slides 6–7 | 77 | 0:43 |
| E1 block / optional sentence | 50 / 20 | 0:26 / 0:09 |
| E2 block / optional sentence | 54 / 19 | 0:31 / 0:09 |
| E3 block / optional sentence | 54 / 22 | 0:30 / 0:10 |
| E4 block / optional sentence | 58 / 18 | 0:30 / 0:11 |
| Final solution, slides 11–13 | 92 | 0:48 |
| Takeaways (3 / 2 / 1 members) | 32 / 21 / 21 | 0:17 / 0:11 / 0:11 |

Segment times exclude slide changes, so they add up to slightly less than the tables in §1.

## 6. Sources of every number

Spoken numbers and slide numbers, with where they come from. Run ids are folders under `experiments/`; R### = `research/r###_*.py` and its runs. Audited values: `reports/tables/r440_core_numbers.csv` (R440, ids in brackets).

| Number | Meaning | Source |
|---|---|---|
| 250 (113 positive), 150, 19,750; 20,000 | training rows, features, test rows; train + test rows | `scripts/verify_data.py`; R440 D1–D4 |
| ≈ 395 (2 %) | public leaderboard rows | competition rule 3 (`docs/COMPETITION_RULES.md`); R440 D5 |
| 0.065; 0.02 | Hanley–McNeil SE of a 50-row fold at AUC 0.80; of a 395-row public score at 0.85 | R440 D10; `report_core.md` §1 (Hanley & McNeil, 1982) |
| 13; 7 + 130 | high-variance block (sd > 1.5 × median, all 20,000 rows); linear combinations and pure-noise columns | R001 `20261022-051627_r001_data_structure`; R150 `20261022-064606_r150_taxonomy` (R440 G6, G20) |
| 5-fold × 20, seed 2026 + r, 100 folds | official CV | D002; `project.yaml` `official_cv` |
| 2 split SE, corrected p < 0.05 | rule for a win | `docs/ARCHIVE_RULES.md` §6; `scripts/compare_runs.py` (Nadeau & Bengio, 2003) |
| 50 seeds; 64 %; MDD 0.057 / 0.030 / 0.0061 [0.0051, 0.0076] | E1 split seeds; hold-out p_correct; minimum detectable difference by scheme | `20261022-064054_e1_validation_protocol`, `reports/tables/e1_paired_difference.csv` |
| about five times smaller | QDA pair: split noise 4.5–6.1 times smaller | `20261022-064148_e1_validation_protocol_qda_pair`; `reports/extensions/e1_validation_protocol.md` |
| six uploads; within 1.4 SE | S001–S006; z(exp) from −1.35 to +0.55 | `registry/submissions.csv`; `reports/tables/core_journey.md` |
| 0.83589, 0.99277, 0.99923, 0.79136, 0.99953, 0.99956 | public scores S001–S006 | `registry/submissions.csv` |
| 0.9958 vs 0.878 | CV of QDA vs LDA on the block | R003 runs `20261022-0506xx_r003_*`; `report_core.md` §3 |
| 0.999 predicted; 0.82 guess; honest CV 0.7934 | B003 and B005 pre-registrations; S004's honest CV | `registry/beliefs.md`; `registry/submissions.csv` |
| 0.0003; ≈ 12 of ≈ 39,000 pairs; sd 3–6 × 10⁻⁴ | S005/S006 minus S003; 198 × 197 public pairs; sd of public differences | `registry/submissions.csv`; `report_core.md` §3; D010 item 5 and R936 `20261022-172146_r936_final_selection` |
| 0.659 → 0.847; 0.906 (≈ 0.91); 21 %; ± 0.04 | E2 L1 curve 50 → 250 rows; power law at 500; coverage of the nominal 90 % interval; quoted uncertainty | `20261022-070248_e2_learning_curve`; R942 `20261022-094142_r942_e2_coverage`; `reports/extensions/e2_learning_curve.md` |
| 0.973 → 0.996; 1.000 | E2 QDA and mixture curves | `20261022-070339_e2_learning_curve_qda_block`, `20261022-070445_e2_learning_curve_gmm` |
| 0.867, 0.793, 0.791; 0.075 (≈ 0.07); ≈ 0 at 7; ≈ 0.09 at 30–50 | E3 leaky CV, honest CV, S004 public at 20 features; leak vs public; gap by k | `20261022-063908_e3_leakage` (`reports/tables/e3_grid.csv`, `e3_leaderboard.csv`) |
| 2,000; 7 features ≥ 98.5 %; +0.063 (p = 0.038) | E4 resamples; stable features at k = 10; linear gain over all 150 | `20261022-071414_e4_feature_stability` (`e4_frequency.csv`, `e4_paired.csv`) |
| 0.961 vs 0.996 (p = 0.0006); six never stable | QDA on the stable set vs the block; block features outside the stable set | R320 `20261022-074624_r320_e4_stable_vs_block`; `reports/extensions/e4_feature_stability.md` |
| 1.0000 in 100/100 folds; 0.99923 | S003's CV and public score | `20261022-050708_r003_gmm2_hv`; `registry/submissions.csv` |
| K = 2 | BIC without labels | R101 `20261022-064201_r101_bic_model_choice` |
| 2 labels; 120 | labels for mean AUC ≥ 0.99: mixture; strict QDA (S005 procedure) | R441 `20261022-182022_r441_label_budget_rerun` (`reports/tables/core_label_budget.md`) |
| 0.804 vs 0.806; 222/222 | boundary-row AUC of S003 vs Bayes; replicas where S003 beats each other procedure | R404 `20261022-194238_r404_inductive_pooled` (`reports/tables/core_boundary.md`) |
| 222 replicas; no candidate beats S003 on average (each still wins 6–21% of single replicas) | private-split comparison | R936 `20261022-172146_r936_final_selection` (`reports/tables/r936_private.csv`) |
| S003 + S005; S003 + S006; r930 + S005 | final picks by case | D010, D013 (`registry/decisions.md`) |
| 0.997 (90 %: 0.992–0.9996) | expected private AUC of S003 | R922 `20261022-092146_r922_label_noise_private` (R440 G35, G36) |
