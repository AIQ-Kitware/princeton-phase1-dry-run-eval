/-
# Contextual drag: a paired accuracy difference, and what a sample of it proves

The card runs a model twice over the same problems -- once on a clean prompt,
once with two of its own failed trajectories injected into the context -- and
asserts `drag = acc_clean - acc_2f >= threshold`.

Two things are worth stating formally, and this file states both.

**The pairing identity.** Because both accuracies are means over the SAME
problem set, the drag is itself a mean of a per-problem difference
(`drag_eq_mean_diff`). That is the whole reason the comparison is legitimate,
and it is not a formality: `unpaired_difference_of_no_effect` exhibits an
accuracy function whose per-problem difference is identically zero and whose
two means still differ, purely from which problems each mean was taken over. If
the two accuracies were measured over different problem sets, `drag` would be a
difference of two independent means and the pairing premise would be false.

**The sampling guarantee.** The pairing identity is what makes the sample-size
question well posed, because it turns the drag from a difference of two
estimates into a single mean of one bounded sequence -- and a mean of a bounded
i.i.d. sequence is exactly what Hoeffding's inequality governs.
`abs_measuredDrag_sub_lt_of_hoeffdingSize_le` closes that: on a paired sample of
at least `hoeffdingSize (b - a) ε δ` problems, the measured drag is within `ε`
of the POPULATION drag with probability at least `1 - δ`. `hoeffdingSize` is no
longer a number the file merely defines; it is the hypothesis of a proved bound.

## The width matters, and it is easy to get wrong

`hoeffdingSize` takes the WIDTH of the interval the per-problem difference lives
in, because the sample size scales with its square -- `hoeffdingSize_two_eq`
records that a width-2 quantity needs FOUR TIMES the sample of a width-1 one.

This is not pedantry about constants. A per-problem accuracy lies in `[0, 1]`,
width 1, and `hoeffdingSize 1 0.05 0.05 ≈ 738`. But the paired DIFFERENCE of two
such accuracies lies in `[-1, 1]`, width 2, and that is the quantity the drag
averages -- so the honest sample size for a tolerance of 0.05 at 95% confidence
is about 2951, not 738. Plugging an accuracy's width into a difference's bound
understates the required sample by a factor of four. The width is a hypothesis
of every theorem below precisely so that it cannot be left implicit.

If per-problem accuracies are graded rather than binary and the difference is
known to stay inside a narrower band, that narrower width may be used and the
sample size shrinks quadratically -- which is a real experimental lever, not a
technicality.

## What this does not formalize

Named plainly, so nobody mistakes what is proved for more than it is:

* **The sample is assumed i.i.d., and the card's is not.** `PairedSample` asks
  that the per-problem differences be independent with a common mean. The card's
  problem set is SELECTED -- the 2F condition needs two of the model's own failed
  trajectories, so only problems the model already failed can appear. Selection
  on failure is selection on something correlated with the per-problem
  difference, which is exactly what `PairedSample.indep` and `PairedSample.mean`
  forbid. **This is now the sharpest gap in the file**, and it is the one that
  decides whether the bound applies to the real experiment: everything below is
  true of the population the filtered problems are drawn from, which is not the
  population the card's claim is about.
* **Identically distributed problems.** `PairedSample.mean` forces every problem
  to have the same expected difference. Real problems differ in difficulty. The
  fix is routine -- Hoeffding needs only independence, and the target becomes the
  average of the per-problem means -- but it changes what `m` denotes, and so
  what the confidence statement is a statement about.
* **Nothing says the drag measures contextual interference.** It is the gap
  between two averages; that the injected trajectories are what caused it is an
  experimental-design claim, not a mathematical one, and no amount of sample size
  converts one into the other.
* **`unpaired_difference_of_no_effect` shows the unpaired comparison CAN
  mislead**, with an explicit witness. It does not show that this pipeline's
  filter does mislead -- that would need the filter's selection mechanism, which
  is the same object the first gap above needs.
* **The threshold is nowhere.** Whether a measured drag clears one is not a
  theorem, and the bound below says nothing about whether the population drag is
  large; it says only how well a sample pins it down.

## Where a real formalization would go

The selection gap, and nothing else, is the next thing to do. Two routes, in
increasing order of fidelity:

1. Condition on the filter. Model the filter as an event `F` in the same space
   and state the bound with respect to `μ[·|F]`. This is cheap -- the theorems
   below apply verbatim to the conditional measure -- and it makes explicit that
   the guarantee is about the filtered population, which is what the card should
   be claiming in the first place.
2. Model the two stages. Draw problems, run the clean condition, filter on
   failure, then run the 2F condition. Here the per-problem differences are no
   longer identically distributed and the conditional independence has to be
   argued rather than assumed. `HasCondSubgaussianMGF` and the Azuma-Hoeffding
   result `measure_sum_ge_le_of_hasCondSubgaussianMGF` in the same Mathlib file
   are built for exactly this shape, and are the reason route 2 is tractable at
   all.

Route 1 is what would let a card honestly discharge `hn_sufficient` today.

## Prior art -- what was needed and what was not

Mathlib's in-tree sub-Gaussian machinery was sufficient; no external dependency
is required. The chain used is `hasSubgaussianMGF_of_mem_Icc` (Hoeffding's lemma
for a bounded random variable) followed by
`HasSubgaussianMGF.measure_sum_range_ge_le_of_iIndepFun` (Hoeffding's inequality
for a sum of independent sub-Gaussians), both in
`Mathlib/Probability/Moments/SubGaussian.lean`. Two-sidedness comes from applying
the one-sided bound to the negated sequence and taking a union bound; the
sub-Gaussian parameter for a difference confined to `[a, b]` is `((b-a)/2)^2`,
which is where the width-squared in `hoeffdingSize` comes from.

**StatsMLlib** (github.com/Lean-MoDS/StatsMLlib) was considered and is not
needed. It advertises Hoeffding, McDiarmid and scalar Bernstein and pins
`v4.33.0` against this development's `v4.33.0-rc2`, so it may still be worth
having for results Mathlib lacks -- McDiarmid in particular, if the drag is ever
replaced by a statistic that is not a mean. Nothing about it has been verified
here, and nothing below depends on it.
-/
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Probability.Moments.SubGaussian

open MeasureTheory ProbabilityTheory

namespace AIQ.Teams.ContextualDrag

variable {ι : Type*}

/-! ## The measured quantity and the pairing identity -/

/-- Mean of `f` over the problems in `P`. Zero on the empty set, which never
arises: the card refuses to report a drag when no problem survives. -/
noncomputable def meanOver (P : Finset ι) (f : ι → ℝ) : ℝ :=
  (∑ i ∈ P, f i) / P.card

/-- Contextual drag: clean accuracy minus 2F accuracy, both over `P`. -/
noncomputable def drag (P : Finset ι) (accClean acc2F : ι → ℝ) : ℝ :=
  meanOver P accClean - meanOver P acc2F

/-- **The pairing identity.** The drag equals the mean of the per-problem
difference. This is what makes the two accuracies comparable, and it holds
precisely because both are taken over the same `P`. It is also what makes the
concentration bound below applicable at all: it exhibits the drag as a mean of
ONE sequence, rather than as a difference of two estimates each carrying its own
error. -/
theorem drag_eq_mean_diff (P : Finset ι) (accClean acc2F : ι → ℝ) :
    drag P accClean acc2F = meanOver P (fun i => accClean i - acc2F i) := by
  unfold drag meanOver
  rw [← sub_div, Finset.sum_sub_distrib]

/-- **A mean inherits the bound on its summands.** The deterministic half of the
sample-size story: whatever interval the per-problem difference lives in, the
drag lives in it too, so the drag has a known width before a single problem is
sampled. That width is the input to `hoeffdingSize`. -/
theorem abs_meanOver_le {P : Finset ι} {f : ι → ℝ} {C : ℝ} (hC : 0 ≤ C)
    (h : ∀ i ∈ P, |f i| ≤ C) : |meanOver P f| ≤ C := by
  rcases P.eq_empty_or_nonempty with rfl | hP
  · simpa [meanOver] using hC
  have hcard : (0 : ℝ) < P.card := by exact_mod_cast hP.card_pos
  rw [meanOver, abs_div, abs_of_nonneg hcard.le, div_le_iff₀ hcard]
  calc |∑ i ∈ P, f i| ≤ ∑ i ∈ P, |f i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ P, C := Finset.sum_le_sum h
    _ = C * P.card := by rw [Finset.sum_const, nsmul_eq_mul]; ring

/-- **Why the pairing is not pedantry.** If the two accuracies are measured over
DIFFERENT problem sets, a nonzero drag can appear when there is no effect at
all: here a single accuracy function -- so the per-problem difference is
identically zero -- still yields a nonzero difference of means, purely from
which problems each mean was taken over.

This is the error the terminal node exists to avoid, and it is why
`acc_clean` is restricted to the problems that survived the filter rather than
being taken over all of them. Note that the two sets here are nested, so this is
not a contrived pathology: it is what happens whenever a subset is compared to
the whole. -/
theorem unpaired_difference_of_no_effect :
    ∃ (Q P : Finset ℕ) (acc : ℕ → ℝ),
      P ⊆ Q ∧ meanOver Q acc - meanOver P acc ≠ 0 := by
  refine ⟨{0, 1}, {0}, fun i => if i = 0 then 1 else 0, ?_, ?_⟩
  · intro x hx
    simp only [Finset.mem_singleton] at hx
    simp [hx]
  · rw [meanOver, meanOver]
    rw [Finset.sum_pair (by norm_num : (0 : ℕ) ≠ 1), Finset.sum_singleton]
    norm_num

/-! ## The sample size

`hoeffdingSize` is a quantity, but from here on it is a quantity that appears as
the hypothesis of a proved bound rather than as a number stated alongside one. -/

/-- Hoeffding's sample size for tolerance `ε` at confidence `1 - δ`, for a
per-problem quantity confined to an interval of width `w`: the number of paired
problems needed before a difference of size `ε` is resolvable.

The width is explicit because the answer is quadratic in it and the wrong width
is the easy mistake. At `ε = δ = 0.05`: about 738 for `w = 1`, about 2951 for
`w = 2`. A paired difference of two `[0, 1]` accuracies has `w = 2`. -/
noncomputable def hoeffdingSize (w ε δ : ℝ) : ℝ := w ^ 2 * Real.log (2 / δ) / (2 * ε ^ 2)

/-- **The factor of four.** A quantity of width 2 -- such as the difference of
two accuracies each in `[0, 1]` -- needs four times the sample of a quantity of
width 1, at the same tolerance and confidence. -/
theorem hoeffdingSize_two_eq (ε δ : ℝ) : hoeffdingSize 2 ε δ = 4 * hoeffdingSize 1 ε δ := by
  unfold hoeffdingSize
  ring

/-- **The sample size does what it is named for.** If `n` reaches
`hoeffdingSize w ε δ`, the two-sided Hoeffding tail at that `n` is at most `δ`.
This is pure algebra -- it is the step that converts an exponential bound into a
confidence level, and it is stated separately because it is where the definition
of `hoeffdingSize` is actually justified. -/
theorem two_mul_exp_le_of_hoeffdingSize_le {w ε δ : ℝ} (hw : 0 < w) (hε : 0 < ε) (hδ : 0 < δ)
    {n : ℕ} (hn : hoeffdingSize w ε δ ≤ n) :
    2 * Real.exp (-(2 * n * ε ^ 2) / w ^ 2) ≤ δ := by
  rw [hoeffdingSize, div_le_iff₀ (by positivity)] at hn
  have hlog : Real.log (2 / δ) ≤ 2 * n * ε ^ 2 / w ^ 2 := by
    rw [le_div_iff₀ (by positivity)]
    nlinarith [hn]
  calc 2 * Real.exp (-(2 * n * ε ^ 2) / w ^ 2) ≤ 2 * Real.exp (-Real.log (2 / δ)) := by
        gcongr
        rw [neg_div]
        linarith
    _ = δ := by
        rw [Real.exp_neg, Real.exp_log (by positivity)]
        field_simp

/-! ## A population, a sample, and a confidence statement

From here the file is about a probability space. `D i` is the per-problem
difference `acc_clean i - acc_2f i` for the `i`-th sampled problem -- a random
variable, because which problem lands in slot `i` is random. The pairing identity
above is what licenses treating it as a single random variable rather than as a
difference of two. -/

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- A paired sample of per-problem differences: independent, confined to
`[a, b]`, each with population mean `m`. `m` is THE POPULATION DRAG -- the
quantity the card's claim is really about -- and `D i` is what the experiment
observes at problem `i`.

Every hypothesis here is a real assumption about the experiment, and the first
two are the ones the card's filter puts at risk. -/
structure PairedSample (μ : Measure Ω) (D : ℕ → Ω → ℝ) (a b m : ℝ) : Prop where
  /-- Distinct problems give independent differences. Selection on failure
  breaks this. -/
  indep : iIndepFun D μ
  meas : ∀ i, AEMeasurable (D i) μ
  /-- The per-problem difference is confined to `[a, b]`; for two accuracies in
  `[0, 1]` this is `[-1, 1]`, of width 2. -/
  bdd : ∀ i, ∀ᵐ ω ∂μ, D i ω ∈ Set.Icc a b
  /-- Every problem has the same expected difference `m`, the population drag. -/
  mean : ∀ i, μ[D i] = m

/-- The drag as measured on the first `n` sampled problems: the same `meanOver`
as above, now over a random sample rather than a fixed problem set. -/
noncomputable def measuredDrag (n : ℕ) (D : ℕ → Ω → ℝ) (ω : Ω) : ℝ :=
  meanOver (Finset.range n) (fun i => D i ω)

omit [MeasurableSpace Ω] in
theorem measuredDrag_eq (n : ℕ) (D : ℕ → Ω → ℝ) (ω : Ω) :
    measuredDrag n D ω = (∑ i ∈ Finset.range n, D i ω) / n := by
  simp [measuredDrag, meanOver]

/-- **Upper tail.** The measured drag overshoots the population drag by `ε` or
more with probability at most `exp (-2 n ε² / (b-a)²)`. -/
theorem measureReal_add_le_measuredDrag [IsProbabilityMeasure μ] (D : ℕ → Ω → ℝ)
    {a b m : ℝ} (hab : a < b) (hD : PairedSample μ D a b m) {n : ℕ} (hn : 0 < n)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | m + ε ≤ measuredDrag n D ω}
      ≤ Real.exp (-(2 * n * ε ^ 2) / (b - a) ^ 2) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  set c : NNReal := (‖b - a‖₊ / 2) ^ 2 with hcdef
  have hcr : (c : ℝ) = (b - a) ^ 2 / 4 := by
    rw [hcdef]
    push_cast
    rw [Real.norm_eq_abs, abs_of_pos (sub_pos.2 hab)]
    ring
  have hsub : ∀ i, HasSubgaussianMGF (fun ω => D i ω - m) c μ := fun i => by
    have h := hasSubgaussianMGF_of_mem_Icc (hD.meas i) (hD.bdd i)
    rwa [hD.mean i] at h
  have hindep' : iIndepFun (fun i ω => D i ω - m) μ := by
    have h := hD.indep.comp (fun _ (x : ℝ) => x - m) (fun _ => measurable_id.sub_const m)
    simpa [Function.comp_def] using h
  have key := HasSubgaussianMGF.measure_sum_range_ge_le_of_iIndepFun
      (X := fun i ω => D i ω - m) hindep' (c := c) (n := n) (fun i _ => hsub i)
      (ε := (n : ℝ) * ε) (by positivity)
  have hset : {ω | m + ε ≤ measuredDrag n D ω}
      = {ω | (n : ℝ) * ε ≤ ∑ i ∈ Finset.range n, (D i ω - m)} := by
    ext ω
    simp only [Set.mem_ofPred_eq, measuredDrag_eq, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul]
    rw [le_div_iff₀ hn']
    constructor <;> intro h <;> linarith
  rw [hset]
  refine key.trans (le_of_eq ?_)
  congr 1
  rw [hcr]
  field_simp
  ring

/-- **Lower tail**, by applying the upper tail to the negated sample. The
negation preserves independence, boundedness (in `[-b, -a]`, the same width) and
the mean-`(-m)` condition, which is why no second Chernoff argument is needed. -/
theorem measureReal_measuredDrag_le_sub [IsProbabilityMeasure μ] (D : ℕ → Ω → ℝ)
    {a b m : ℝ} (hab : a < b) (hD : PairedSample μ D a b m) {n : ℕ} (hn : 0 < n)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | measuredDrag n D ω ≤ m - ε}
      ≤ Real.exp (-(2 * n * ε ^ 2) / (b - a) ^ 2) := by
  have hD' : PairedSample μ (fun i ω => -(D i ω)) (-b) (-a) (-m) := by
    refine ⟨?_, fun i => (hD.meas i).neg, fun i => ?_, fun i => ?_⟩
    · have h := hD.indep.comp (fun _ (x : ℝ) => -x) (fun _ => measurable_neg)
      simpa [Function.comp_def] using h
    · filter_upwards [hD.bdd i] with ω hω
      exact ⟨neg_le_neg hω.2, neg_le_neg hω.1⟩
    · simp [integral_neg, hD.mean i]
  have h := measureReal_add_le_measuredDrag (fun i ω => -(D i ω)) (by linarith : -b < -a) hD' hn hε
  have hset : {ω | measuredDrag n D ω ≤ m - ε}
      = {ω | -m + ε ≤ measuredDrag n (fun i ω => -(D i ω)) ω} := by
    ext ω
    have hneg : measuredDrag n (fun i ω => -(D i ω)) ω = -measuredDrag n D ω := by
      simp [measuredDrag_eq, Finset.sum_neg_distrib, neg_div]
    simp only [Set.mem_ofPred_eq, hneg]
    constructor <;> intro hx <;> linarith
  rw [hset]
  refine h.trans (le_of_eq ?_)
  congr 2
  ring

/-- **Two-sided Hoeffding for the drag.** The measured drag misses the
population drag by `ε` or more with probability at most
`2 exp (-2 n ε² / (b-a)²)`. The width `b - a` enters squared, which is the whole
content of `hoeffdingSize_two_eq`. -/
theorem measureReal_le_abs_measuredDrag_sub [IsProbabilityMeasure μ] (D : ℕ → Ω → ℝ)
    {a b m : ℝ} (hab : a < b) (hD : PairedSample μ D a b m) {n : ℕ} (hn : 0 < n)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ |measuredDrag n D ω - m|}
      ≤ 2 * Real.exp (-(2 * n * ε ^ 2) / (b - a) ^ 2) := by
  have hsubset : {ω | ε ≤ |measuredDrag n D ω - m|}
      ⊆ {ω | m + ε ≤ measuredDrag n D ω} ∪ {ω | measuredDrag n D ω ≤ m - ε} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω
    rcases le_total m (measuredDrag n D ω) with h | h
    · rw [abs_of_nonneg (by linarith)] at hω
      exact Or.inl (by simp only [Set.mem_ofPred_eq]; linarith)
    · rw [abs_of_nonpos (by linarith)] at hω
      exact Or.inr (by simp only [Set.mem_ofPred_eq]; linarith)
  refine (measureReal_mono hsubset).trans ((measureReal_union_le _ _).trans ?_)
  have h1 := measureReal_add_le_measuredDrag D hab hD hn hε
  have h2 := measureReal_measuredDrag_le_sub D hab hD hn hε
  linarith

/-- **The statement the card needs.** On a paired sample of at least
`hoeffdingSize (b - a) ε δ` problems, the measured drag is within `ε` of the
POPULATION drag with probability at least `1 - δ`.

This is what turns `hoeffdingSize` from a number into a guarantee, and it is the
form in which a card could discharge a sufficient-sample-size premise rather
than assume it -- for the population its problems are actually drawn from. Read
the hypotheses before using it: `PairedSample.indep` and `PairedSample.mean` are
assumptions about the sampling, and a filtered problem set does not satisfy them
without further argument. -/
theorem abs_measuredDrag_sub_lt_of_hoeffdingSize_le [IsProbabilityMeasure μ] (D : ℕ → Ω → ℝ)
    {a b m : ℝ} (hab : a < b) (hD : PairedSample μ D a b m) {n : ℕ} (hn0 : 0 < n)
    {ε δ : ℝ} (hε : 0 < ε) (hδ : 0 < δ) (hn : hoeffdingSize (b - a) ε δ ≤ n) :
    1 - δ ≤ μ.real {ω | |measuredDrag n D ω - m| < ε} := by
  have hsum : AEMeasurable (fun ω => ∑ i ∈ Finset.range n, D i ω) μ := by
    have h := Finset.aemeasurable_sum (Finset.range n) fun i (_ : i ∈ Finset.range n) => hD.meas i
    exact h.congr (Filter.Eventually.of_forall fun ω => Finset.sum_apply ω _ _)
  have hmd : AEMeasurable (fun ω => measuredDrag n D ω) μ := by
    simpa [measuredDrag_eq] using hsum.div_const ((n : ℝ))
  have hs : NullMeasurableSet {ω | |measuredDrag n D ω - m| < ε} μ :=
    nullMeasurableSet_lt (hmd.sub_const m).abs aemeasurable_const
  have hcompl : μ.real {ω | ε ≤ |measuredDrag n D ω - m|}
      = 1 - μ.real {ω | |measuredDrag n D ω - m| < ε} := by
    rw [← probReal_compl_eq_one_sub₀ hs]
    congr 1
    ext ω
    simp [not_lt]
  have h := (measureReal_le_abs_measuredDrag_sub D hab hD hn0 hε.le).trans
    (two_mul_exp_le_of_hoeffdingSize_le (by linarith) hε hδ hn)
  rw [hcompl] at h
  linarith

end AIQ.Teams.ContextualDrag
