# Frontier Systems Playground — Weekend Project Plan

**Drafted:** May 2026
**Status:** Draft v1
**Horizon:** 18 weeks, weekend cadence (~3 productive days/week)

---

## Purpose

Playground where all the toys are neural networks.

Weekend project to mess with things I don't get to explore properly: training dynamics when you swap architectural components, why agent benchmarks fail in specific patterns, scaffold design choices. Build stuff, break stuff, write up what actually happened.

---

## Roadmap

| Phase | Weeks | Focus | Deliverable | Compute budget |
|-------|-------|-------|-------------|----------------|
| 1 | 1–3 | ML fundamentals — llmkit replication & extension | Public repo + writeup (1,500 words) | AUD $150 |
| 1a | 4–6 | **SFT fundamentals — instruction tuning** | **Public repo + SFT writeup (1,500 words)** | **AUD $100** |
| 1b | 7–10 | **RL fundamentals — PPO vs DPO comparison** | **Public repo + RL writeup (2,000 words)** | **AUD $200** |
| 2 | 11–14 | Agent benchmark replication (τ-bench or SWE-Bench) | Public repo + replication notes (2,000 words) | AUD $200 |
| 3 | 15–21 | Empirical study: flow-first vs reasoning-first scaffolds | Workshop-shaped paper + repo + blog post | AUD $400 |
| 4 | 22–25 | Capability eval design *(optional)* | Eval suite + benchmark proposal writeup | — |
| 5 | 26–29 | SAE-based task boundary recognition *(optional)* | Workshop paper + repo + SAE analysis | AUD $140 |

**Gate rule:** no phase begins until the previous phase's artefact is published.

Phases 4 and 5 are optional. Skip if Phase 3 produces something genuinely interesting, if calendar is at risk, or if the curiosity has been scratched. Three solid explorations beat five rushed ones.

---

## Phase Detail

### Phase 1 — ML Fundamentals (weeks 1–3)

Reproduce GPT-2 124M training on a small corpus (TinyStories or OpenWebText subset), then introduce **one** substantive architectural modification in isolation: RoPE instead of learned positional embeddings, GQA instead of MHA, or an alternative optimiser (Lion, Sophia). Compare loss curves quantitatively.

Want to see what actually changes when you swap out a component, not just read about it in papers.

**Constraints**
- Single GPU (rented H100 by the hour — Lambda, Modal, or Runpod). No multi-node.
- One variable, isolated. Do not stack changes.
- Compute cap: AUD $150.

**Acceptance criteria**
- Public repo with reproducible training script and README.
- Writeup (1,500 words): modification, hypothesis, result, what comes next.
- Loss curves charted and explained.
- Honest discussion of where the modification helped, hurt, or did neither.

---

### Phase 1a — SFT Fundamentals (weeks 4–6) NEW

Take a pre-trained checkpoint from Phase 1 and fine-tune it on instruction-following data. Compare base model vs SFT'd model on instruction-following tasks. Understand what changes during SFT and why it matters for agent systems.

This bridges pre-training and agents: agents need instruction-following capability, and SFT is how you get it.

**Constraints**
- Single checkpoint from Phase 1 baseline (GPT-2 124M)
- One instruction dataset (Alpaca-52k or Dolly-15k)
- Single GPU (rented H100 by the hour)
- Compute cap: AUD $100

**Acceptance criteria**
- Public repo: `llmkit-sft` package with reusable SFT trainer
- Comparison: base vs SFT on simple instruction-following tasks
- Writeup (1,500 words): what is SFT, data format choices, hyperparameters, quantitative comparison, qualitative sample analysis
- Honest discussion: where SFT helps, where it doesn't, what breaks
- Loss curves and instruction-following accuracy charted

**Key questions to answer**
- How much does loss drop during SFT vs pre-training?
- What hyperparameters matter most (learning rate, sequence length)?
- Does the model actually follow instructions or just pattern-match?
- What instruction types work well vs poorly?

---

### Phase 1b — RL Fundamentals (weeks 7–10) NEW

Implement RLHF pipeline from scratch: train a reward model on preferences, then optimize policy with PPO. Compare with DPO (simpler, no reward model). Understand the fundamental trade-offs.

This completes the modern LLM training pipeline and provides RL-tuned models for agent experiments.

**Constraints**
- Start from SFT checkpoint from Phase 1a
- Small preference dataset (HH-RLHF subset, ~10k pairs)
- Implement both PPO and DPO for comparison
- Single GPU for reward model, 2-GPU for PPO (if needed)
- Compute cap: AUD $200

**Acceptance criteria**
- Public repo: `llmkit-rl` package with PPO and DPO implementations
- Working reward model with validation accuracy >70%
- PPO training with KL divergence tracking
- DPO training as simpler baseline
- Writeup (2,000 words):
  - Reward model training and evaluation
  - PPO vs DPO comparison (complexity, sample efficiency, final quality)
  - KL divergence analysis and what it means
  - Rollout generation and reward curves
  - When to use which approach
- Honest discussion: what worked, what was surprisingly hard, implementation gotchas

**Key questions to answer**
- Is reward model accuracy predictive of final RL quality?
- How much does PPO improve over SFT baseline?
- Is DPO's simplicity worth the potential quality trade-off?
- What's the right KL penalty weight?
- Do RL'd models actually behave more helpfully/safely?

---

### Phase 2 — Agent Benchmark Replication (weeks 11–14)

Pick one published baseline — τ-bench paper baselines or SWE-agent on a SWE-Bench Verified subset — and reproduce the headline result. Document where you matched, where you diverged, and why.

**Constraints**
- One benchmark, one baseline. Do not attempt to beat the baseline.
- Use frontier model APIs (Claude Sonnet, GPT-4 class). No local model serving.
- Compute/API cap: AUD $200.

**Acceptance criteria**
- Public repo: agent harness, evaluation runner, configuration.
- Writeup (2,000 words): setup, result, deltas from published baseline, failure mode taxonomy from manual inspection of ≥30 trajectories.
- Explicit analysis of undocumented choices in the original paper that affect reproducibility.

---

### Phase 3 — Flow-first vs Reasoning-first Empirical Study (weeks 8–14)

**The interesting one.**

Empirical comparison of two architectural paradigms:

- **Flow-first:** deterministic control flow with LLM nodes.
- **Reasoning-first:** LLM-driven planning with tool access.

Build minimal reference implementations of each. Run both on τ-bench (isolates tool use and multi-turn behaviour). Report success rate, cost, latency, failure mode taxonomy, and — critically — the conditions under which each architecture is preferable.

**Constraints**
- Two architectures, precisely defined and documented *before* any code is written.
- One benchmark, one model family (Claude Sonnet for both arms).
- Pre-register the experimental design: hypotheses, metrics, what would constitute support for each.
- Compute/API cap: AUD $400.

**Acceptance criteria**
- Pre-registration document (one page) committed to the repo before experiments run.
- Public repo: both architectures, benchmark harness, analysis notebooks.
- Writeup (3,000–4,000 words): motivation → prior work → architectures defined → experimental design → results → failure mode analysis → conditions favouring each → limitations → future work.
- Submission to a relevant workshop (stretch). Strong blog post is the floor.

**Pre-commitment:** publish regardless of outcome. A null result, honestly characterised, is publishable and signals research integrity. Do not iterate past the scope to manufacture an interesting result.

---

### Phase 4 — Capability Eval Design (weeks 15–18, optional)

Design a small, novel capability eval suite (30–100 tasks, clean held-out split) targeting an underexplored capability: agent recognition of impossible tasks, calibration on confidence claims, or performance under adversarial tool descriptions. Run across 3–4 frontier models. Frame as a benchmark proposal.

**Skip Phase 4 if:** Phase 3 turns out really interesting, calendar is getting tight, or evals stop being fun.

---

### Phase 5 — SAE-Based Task Boundary Recognition for Agent Scaffolds (weeks 19–22, optional)

**Status:** Draft v1
**Phase window:** Weeks 19–22 (optional, contingent on Phase 3 delivery)

#### Overview

This phase produces a two-tier research artefact connecting mechanistic interpretability to agent scaffold behaviour. The core claim under investigation: residual stream activations in the final layers of a language model contain recoverable signal indicating when the model is operating outside its training distribution — and that signal can be used to route scaffold behaviour in real time.

This is genuinely novel research. Adjacent work (activation steering, probing classifiers, SAE feature analysis) operates either inside the model or as a post-hoc interpretability exercise. This proposal connects the two: using interpreted internal state to influence external scaffold decisions at inference time. No published work has done this cleanly for agent scaffolds specifically.

The scope is deliberately narrow. Of the four candidate scaffold features initially considered — task boundary recognition, uncertainty detection, reasoning vs retrieval discrimination, and loop detection — this phase focuses exclusively on task boundary recognition. Uncertainty detection is the natural extension and is documented as future work. The other two are parked as stretch goals.

#### Key Design Decisions

The following decisions are locked before any code is written. They are recorded here to prevent scope drift and to make the experimental design legible to external reviewers.

**Model:** GPT-2 small (124M parameters). Small enough to train and collect activations cheaply. Large enough to exhibit meaningful superposition and interpretable features. Not a frontier model — this is intentional. The research question is architectural, not capability-dependent.

**Layer selection: final layers only.** Activations are intercepted at the final two transformer blocks (layers 10 and 11 of GPT-2's 12-layer stack). This decision was made on the following reasoning: task boundary recognition is a high-level semantic judgement about whether the input falls within the model's competence. That judgement is most expressed in late-network representations where task-relevant features have been progressively refined through the residual stream. Mid-network layers encode richer superposition but less task-specific signal. Earlier layers encode syntactic and positional features that are less relevant to distribution shift detection. The trade-off accepted: we may miss features that are written early and not preserved to the final layers. This is an acknowledged limitation.

**SAE as external component.** The sparse autoencoder is trained on frozen GPT-2 activations collected after training. It is not part of the model and does not modify inference. At runtime, it sits between the model's residual stream and the scaffold router — intercepting activations, projecting into sparse feature space, and returning a feature activation vector that the scaffold can read.

**Task boundary as binary signal.** The target signal is binary: in-distribution or out-of-distribution. This is the most tractable formulation. It avoids the harder problem of characterising *what kind* of OOD the input is, and produces a clean routing decision for the scaffold (proceed vs escalate/fallback). Uncertainty detection, which requires distinguishing epistemic uncertainty from calibration failure, is explicitly deferred.

**Scaffold routing, not model modification.** The SAE output influences the scaffold's control flow — which tool to call, whether to escalate, whether to request clarification — not the model's internal computation. This is a clean separation that makes the system interpretable and the experiment falsifiable.

#### Tier One — SAE Training and Feature Analysis

**Objective**

Train a sparse autoencoder on GPT-2 small residual stream activations (layers 10–11) and identify features that reliably activate on out-of-distribution inputs relative to a defined in-distribution corpus.

**Dataset construction**

Two corpora are required:

**In-distribution corpus:** A domain-constrained text dataset the GPT-2 model has strong prior on. TinyStories or a filtered OpenWebText subset is suitable. The domain should be narrow enough that OOD is well-defined — conversational English prose, simple factual statements, or a specific topic domain (e.g., news headlines).

**Out-of-distribution corpus:** Inputs that are structurally or semantically outside the in-distribution domain. Candidates: code, mathematical notation, foreign language text, adversarial prompts, highly technical domain text (legal, medical), or nonsense/noise sequences. Multiple OOD categories are useful for characterising which features are general OOD detectors vs domain-specific.

Both corpora should be balanced in token count. Target 50,000–100,000 examples per split. Collect residual stream activations at layers 10 and 11 for every example.

**SAE architecture**

Standard sparse autoencoder with expansion factor of 4–8x. Input dimension matches `d_model` (768 for GPT-2 small). Hidden dimension 3,072–6,144. L1 sparsity penalty on hidden activations. Reconstruction loss (MSE) plus sparsity loss.

The training objective is to find a sparse high-dimensional basis in which the residual stream activations can be approximately reconstructed, with the sparsity constraint forcing clean feature separation.

**Why not a probing classifier directly?** A linear probe trained to classify in vs OOD would be simpler and faster. The SAE approach is chosen because: (1) it surfaces the feature structure rather than just producing a binary output, making the analysis richer and more publishable; (2) the features found may generalise to uncertainty detection and other scaffold signals in future work; (3) it is the current state-of-art approach in mechanistic interpretability and is more legible to a research hiring committee.

**Feature analysis**

After training, identify which SAE features are most predictive of OOD inputs. Approaches:

- Feature activation distributions: which hidden units activate more on OOD than in-distribution inputs
- Logistic regression on SAE feature activations: which features have highest coefficient magnitude for OOD prediction
- Manual inspection: for the top 20 predictive features, what inputs maximally activate them? Are they interpretable?

The analysis should produce: a ranked list of OOD-predictive features, activation statistics for each, example inputs that maximally activate each feature, and an honest assessment of which features are interpretable vs opaque.

**Acceptance criteria for Tier One**

- SAE trained with reconstruction loss below a threshold that confirms the sparse basis is meaningful (to be set after a pilot run)
- At least 5 SAE features identified as reliably OOD-predictive (AUC > 0.75 on held-out test split)
- Manual inspection of top features completed and documented
- Null result documented honestly if no clean features emerge — this is itself a finding about the limits of final-layer SAEs for this task

**Compute budget**

Activation collection: approximately AUD $20–40 (GPU hours for inference over 100K examples). SAE training: small network, CPU-feasible but GPU-accelerated for speed. Total Tier One compute: AUD $80 maximum.

#### Tier Two — Scaffold Integration

**Objective**

Build a minimal agent scaffold that reads SAE feature activations at inference time and uses the task boundary signal to route scaffold behaviour. Evaluate whether the routing improves task success rate and reduces failure modes relative to a baseline scaffold without the SAE router.

**Architecture**

The system has three components:

**Model + activation hook.** A forward pass hook intercepts the residual stream at layer 11 after each token is processed. For generation tasks, this is collected at the final input token before generation begins — the model's "read" of the full context before it starts writing.

**SAE router.** The intercepted activation is pushed through the trained SAE (frozen from Tier One). The output is a sparse feature vector. A lightweight classifier (logistic regression or small MLP trained in Tier One) maps this to a scalar OOD score in [0, 1].

**Scaffold controller.** The OOD score gates scaffold behaviour:

- Score below threshold → proceed with standard tool-call or generation
- Score above threshold → trigger fallback behaviour (request clarification, escalate to human, return "I don't know", or invoke a different tool path)

The threshold is a hyperparameter. The writeup should include a sweep and honest discussion of threshold sensitivity.

**Benchmark**

τ-bench is the primary evaluation benchmark, consistent with Phases 2 and 3. This provides direct comparability with the Phase 3 results and avoids introducing a new benchmark as a confound.

Construct two task splits:

**In-distribution split:** Tasks that fall within GPT-2's competence on the corpus used for SAE training. These should be handled correctly by the baseline scaffold. The SAE router should assign low OOD scores and not intervene.

**Out-of-distribution split:** Tasks that are structurally outside the training distribution. The baseline scaffold should fail or hallucinate. The SAE router should assign high OOD scores and trigger the fallback.

The primary metric is: does the SAE-routed scaffold produce better outcomes on OOD tasks than the baseline, without degrading performance on in-distribution tasks?

**Failure mode analysis**

Manual inspection of at least 30 trajectories per condition. Expected failure modes to document:

- False positives: SAE router flags in-distribution tasks as OOD, triggering unnecessary fallback
- False negatives: SAE router misses genuine OOD inputs, scaffold proceeds and fails
- Threshold sensitivity: does the system degrade gracefully or catastrophically as threshold varies
- Latency cost: the SAE forward pass adds overhead — measure and report

**Pre-registration**

Before running any Tier Two experiments, commit a one-page pre-registration to the repo documenting: hypotheses, success criteria, what would constitute a null result, and what would constitute a positive result. This is consistent with the Phase 3 discipline and signals research integrity.

**Acceptance criteria for Tier Two**

- Measurable improvement in OOD task handling relative to baseline (success rate, hallucination rate, or appropriate fallback rate)
- No statistically significant degradation on in-distribution tasks
- Honest failure mode taxonomy with at least 30 inspected trajectories
- Latency overhead documented
- Null result accepted and published if the routing signal doesn't improve scaffold behaviour

**Compute budget**

Inference for benchmark evaluation: AUD $60–80. Total Phase 5 compute: AUD $140 maximum, well within the overall program envelope.

#### Writeup Structure

The combined artefact is structured as a single paper with two clearly delineated sections, targeting a workshop submission (LangChain Agents, NeurIPS workshops on agent evaluation, or COLM).

**Abstract:** The core claim, the approach, the result.

**1. Motivation:** Why task boundary recognition matters for agent scaffolds. The failure mode it addresses. Why current approaches (prompting, output-level confidence scores) are insufficient.

**2. Background:** Residual stream and superposition (brief). Sparse autoencoders. Existing work on OOD detection, activation steering, probing classifiers. What's missing.

**3. Layer selection rationale:** Why final layers. The trade-off accepted. This section documents the key design decision explicitly.

**4. Tier One — SAE training and feature analysis:** Architecture, dataset, results, feature inspection. Honest about null results.

**5. Tier Two — Scaffold integration:** Architecture diagram, benchmark setup, results, failure mode taxonomy.

**6. Discussion:** What the results mean for scaffold design. Conditions under which SAE-based routing is and isn't useful. The path to uncertainty detection as a natural extension.

**7. Limitations and future work:** Final-layer limitation. GPT-2 scale limitation. Threshold sensitivity. Uncertainty detection, loop detection as next steps.

Target length: 4,000–5,000 words plus figures.

#### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No clean OOD-predictive features emerge from SAE | Medium | High | Document as a finding about final-layer SAEs. Pivot writeup to "negative result with implications for layer selection." |
| SAE routing doesn't improve scaffold performance | Medium | Medium | Pre-registered null result is publishable. The Tier One analysis alone is sufficient for a strong blog post. |
| Latency overhead makes real-time routing impractical | Low | Medium | Report the number honestly. Frame as a research prototype, not a production system. |
| Scope creep into uncertainty detection | High | High | Uncertainty detection is explicitly future work. One mention in the limitations section. No experiments. |
| Phase 3 delays push Phase 5 start past week 19 | Medium | Medium | Phase 5 is optional. If Phase 3 slips, Phase 5 is dropped entirely. Three strong artefacts are better than four weakened ones. |

#### What Success Looks Like

A positive result — SAE features reliably signal OOD inputs and routing improves scaffold outcomes — is a publishable finding with direct implications for production agent design. It opens a research direction (introspection-aware scaffolds) that nobody has explored systematically.

A null result — features don't cleanly separate, or routing doesn't help — is also publishable and arguably more honest about the current limits of SAE-based interpretability for real-time inference. It narrows the search space for future work.

Either outcome, written up rigorously and honestly, is more interesting than another capability eval suite. This is the phase most likely to produce something genuinely novel — not replicating known results, but asking a question nobody has asked cleanly before.

**Phase 5 is contingent on Phase 3 delivery. Do not begin Tier One until the Phase 3 writeup is published. The temptation to start early because the idea is exciting is the exact failure mode this plan is designed to prevent.**

---

## Operating Model

**Cadence**
- Weekend-only. Do not integrate with day-job hours.
- Assume one weekend per fortnight is fully lost. Plan for ~3 productive days per phase-week.
- Sunday evening review: what shipped, what's blocked, is the publish date still credible.

**Tooling**
- Rented H100s by the hour (Lambda, Modal, Runpod). No reserved capacity.
- All code public from day one — easier to share and keep honest.
- Simple landing page linking everything (Quarto or Astro static site).

**Writing discipline**
- Each phase has a hard publish date. Calendar it.
- Write the writeup before you think you're ready — writing exposes gaps in the analysis.
- One round of external review per artefact: one ML systems reviewer, one writing-quality reviewer. Identify them in advance.

**Opportunistic outreach**
- If any of this produces something interesting, might be worth chatting to people about. But that's secondary — finish the work first.

---

## What Success Looks Like

| Outcome | Evidence |
|---------|----------|
| Three finished explorations | Public repos, each with honest writeup of what worked and what didn't |
| Actually learned something | Can explain the technical choices and their trade-offs without handwaving |
| Workshop submission *(stretch)* | Phase 3 might be worth submitting to LangChain Agents, NeurIPS workshops, or COLM if it's interesting enough |
| Had fun | Would do another round of this afterwards |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Starting all phases, finishing none | High | High | Hard publish dates; no starting Phase N+1 until N is published |
| Day-job collapses weekend capacity | Medium | High | 20% buffer per phase; re-baseline rather than compress scope if two consecutive weekends lost |
| Phase 3 null result | Medium | Medium | Pre-commit to publishing regardless; null result is still interesting |
| Compute overrun | Medium | Low–Med | Enforce per-phase caps; prefer ≤3B models for all training experiments |
| Scope creep on Phase 3 | High | High | Pre-register before running; log out-of-scope findings as Phase 5, do not absorb |
| Phase 2 replication fails | Medium | Low | The attempt is the artefact; document the failure thoroughly |
| SFT/RL implementation harder than expected | Medium | Medium | Phases 1a/1b have reduced scope; stub implementations acceptable if documented |
| RL training unstable (PPO divergence) | Medium | Medium | DPO provides simpler fallback; focus writeup on comparison rather than perfect tuning |
| Loses steam halfway through | Medium | Medium | That's fine — even two finished explorations is better than five half-done ones |

---

## Key Assumptions

- Day-job intensity remains manageable. If things get hectic, timelines stretch but scope stays the same.
- Personal compute/API budget of ~AUD $1,190 across the full program (Phases 1–5, including 1a and 1b).
  - Phase 1: AUD $150 (pre-training)
  - Phase 1a: AUD $100 (SFT)
  - Phase 1b: AUD $200 (RL)
  - Phase 2: AUD $200 (agent benchmark)
  - Phase 3: AUD $400 (empirical study)
  - Phase 5: AUD $140 (SAE study, optional)
- Frontier model API access at consumer terms throughout.

---

## Explicit Non-Goals

- Breadth across all research areas. Focusing on **the full LLM pipeline → agents and evals** — understanding the whole stack from pre-training through RL to agent behavior.
- Peer-reviewed publication. Workshop submission if something turns out genuinely novel; otherwise just solid writeups.
- Frontier-scale results. All training experiments use ≤3B parameter models (primarily GPT-2 124M). The questions are interesting at small scale.
- SOTA performance on benchmarks. Goal is understanding the techniques, not beating leaderboards.
- Production-ready implementations. Research-quality code with clear patterns is sufficient; optimization comes later if needed.
- Complex multi-task or multi-dataset training. One dataset per phase for clean comparisons.

---

*Review monthly. Re-baseline if any phase publish date slips by more than two weeks.*
