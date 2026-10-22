# Verified possession methods for this challenge

Read against `CHALLENGE.md`, `DATA.md`, and `EVALUATION.md` on 2026-10-21. Only public documentation and primary papers were used; no private labels or corrected tracking were inspected.

## Three relevant papers, and what transfers

### Link & Hoernig (2017), Individual ball possession in soccer

[Full primary article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953), particularly sections 2 and 3.1–3.4.

The paper distinguishes team possession, individual action, and stricter control. Team possession includes pass flight; individual possession persists until another player can act on the ball. Its preprocessing rejects ball jumps, discards very short usable sequences, and smooths player/ball trajectories with a Rauch–Tung–Striebel smoother. Detection uses distance, ball motion and acceleration; a Bayesian network estimates control using opponent context. Evaluation reports IBA F1 0.88 and IBP F1 0.83 on Bundesliga material, not challenge spellF1.

**Critical limitation:** the input includes a running flag telling whether play is live. Its definition of possession starting at action-space entry is also weaker than this challenge's controlled opponent recovery. Transfer its trajectory/contact features and distinction between action and control, not its label boundaries or reported score.

### Vidal-Codina et al. (2022), Automatic event detection in football using tracking data

[Full primary article](https://link.springer.com/article/10.1007/s12283-022-00381-6), sections 2.1 and 2.3.

A deterministic algorithm at 25 Hz uses possession zones around players and a larger duel zone. Opponent proximity makes a duel take precedence. It compares ball direction and speed before/after candidate contacts; despite an introductory sentence saying “both,” the implemented rule described in section 2.3.3 accepts **either** direction or speed change. Gains are assigned to the start of the corresponding proximity interval. A second-order Savitzky–Golay filter over seven frames stabilizes derivatives. Looking ahead to the next controller avoids declaring loss when the same player regains the ball.

**Critical limitation:** live/dead status is supplied explicitly or encoded by missing ball coordinates for specific providers. Set-piece configurations classify already identified stoppages; they are not demonstrated as an independent whistle detector. Our missing ball observations cannot inherit that meaning. The four control states are useful internally, but are not this challenge's output classes.

### Kim et al. (2026), PathCRF

[Primary preprint](https://arxiv.org/html/2602.12080v1), sections 2.3 and 3; [paper-linked code](https://github.com/hyunsungkim-ds/pathcrf).

PathCRF models possession as a path of player-to-player edges and self-loops, with impossible consecutive edge transitions masked. Viterbi provides global decoding. In its same-backbone comparison, independent prediction gives event F1 65.34%, constrained Viterbi 73.14%, and the full dynamic masked CRF 75.69%. These are event scores using player/type agreement and a one-second tolerance, not spellF1.

**Critical limitation:** training/inference uses preidentified in-play episodes, stable player entities, five training matches, one validation match and one test match. Ground-truth edges are constructed from synchronized and augmented events. Our fragmented anonymous slots and five minutes of labels make training its neural architecture directly a poor first investment. Transfer the structured-decoding idea. The inspected version is a preprint with placeholder publication metadata; cite it as such.

## Recommended algorithm, in order of effort

The following is a design proposal, not an experimentally established winner.

1. **Build trustworthy observations.** Convert normalized coordinates to metres; detect jumps and duplicate tracks; smooth within valid short segments without bridging long ball gaps. Compute nearest-team distances, the margin between them, short-term ball/player relative velocity, trajectory bends, and speed changes. Aggregate by team so slot identity changes do not create turnovers.
2. **Detect candidate control, then confirm it.** Proximity alone proposes a contact. Direction/speed change or sustained co-motion strengthens it. Opponent pressure and an isolated transient touch weaken it. Preserve the last owning team through loose balls and duels, as explicitly required by `CHALLENGE.md`. Offline confirmation should retrospectively place a transition at the earliest supported recovery, avoiding a systematic delay equal to the confirmation window. Permit supported one-touch controlled passes.
3. **Decode H/A/D jointly over time.** Start with a small Viterbi model using observation scores, persistence penalties and separately scored transitions into/out of D. Permit H↔A directly; do not mandate a fake intervening state. Avoid a hard minimum possession duration: valid short spells affect spellF1. Treat Smart Tagging team phases and set-piece intervals as soft evidence, not copied labels. Retain boundary candidates at 25 Hz even if contextual features run more slowly.
4. **Treat D as the main unresolved measurement problem.** Combine credible boundary exits, cessation of ball motion, collective player slowing, restart configurations, and automatic set-piece cues. Missing ball alone is not D; stationary ball alone is not D. Referee stoppages without visible geometric evidence remain hard. A dedicated automatic tactical-video live/dead model is a reasonable next investment if tracking cues remain weak; none of these three papers establishes its performance here.
5. **Use the five-minute sample to check semantics and alignment.** Avoid large parameter searches against 17 spells. Prefer a few predetermined ablations and inspect algorithmic diagnostics without adding manual labels. External provider-labelled data can support broader training, but source label conventions and synthetic ball/identity corruption must be documented. Select improvements using independent validation when available.

For spellF1, the primary aim is to avoid spurious H/A/D boundaries while preserving real short possessions. X merely forward-fills within each half before this score: it cannot rescue an uncertain turnover or restart, and explicit retention has the same spellF1 effect. D errors split or merge possessions even when ownership is correct.

## Corrections to the earlier memo

- Its claims that schema, scoring code, annotation policy, and target examples are unavailable are stale. The repository now supplies these; loose/contested live frames retain the previous owning team.
- Its joint live/dead architecture is a proposed extension, not a result established by the cited possession papers. Both older methods depend on supplied game status, as does PathCRF's episode construction.
- Its statement about Link & Hoernig spanning flight with “no individual controller” should distinguish **action/control** from **possession**: the paper explicitly keeps individual possession with the passer during flight.
- “Vidal-Codina requires more than proximity” is supported for its contact/gain logic, but contact evidence alone does not prove the challenge's stricter controlled recovery.
- The earlier memo's other dataset/tool/IFAB references were not re-audited in this focused three-paper reading. They should not be described as newly verified here.
