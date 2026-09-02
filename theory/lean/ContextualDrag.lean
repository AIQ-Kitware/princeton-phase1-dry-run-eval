/-
# Contextual drag: a paired accuracy difference

The card runs a model twice over the same problems -- once on a clean prompt,
once with two of its own failed trajectories injected into the context -- and
asserts `drag = acc_clean - acc_2f >= threshold`.

What is worth stating formally is not the empirical claim but the identity the
card's design depends on: because both accuracies are means over the SAME
problem set, the drag is itself a mean of a per-problem difference. That is the
whole reason the comparison is legitimate. If the two accuracies were measured
over different problem sets, `drag` would be a difference of two independent
means and the pairing premise would be false.

`drag_eq_mean_diff` is proved. The threshold claim is not a theorem -- it is
what the experiment measures -- so it appears here only as a definition, with
the sample-size condition that makes it evidence stated alongside.
-/
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic

namespace AIQ.Teams.ContextualDrag

variable {ι : Type*}

/-- Mean of `f` over the problems in `P`. Zero on the empty set, which never
arises: the card refuses to report a drag when no problem survives. -/
noncomputable def meanOver (P : Finset ι) (f : ι → ℝ) : ℝ :=
  (∑ i ∈ P, f i) / P.card

/-- Contextual drag: clean accuracy minus 2F accuracy, both over `P`. -/
noncomputable def drag (P : Finset ι) (accClean acc2F : ι → ℝ) : ℝ :=
  meanOver P accClean - meanOver P acc2F

/-- **The pairing identity.** The drag equals the mean of the per-problem
difference. This is what makes the two accuracies comparable, and it holds
precisely because both are taken over the same `P`. -/
theorem drag_eq_mean_diff (P : Finset ι) (accClean acc2F : ι → ℝ) :
    drag P accClean acc2F = meanOver P (fun i => accClean i - acc2F i) := by
  unfold drag meanOver
  rw [← sub_div, Finset.sum_sub_distrib]

/-- The card's assertion: the measured drag clears a threshold. A definition,
not a theorem -- whether it holds is what the run decides. -/
def DragExceeds (P : Finset ι) (accClean acc2F : ι → ℝ) (θ : ℝ) : Prop :=
  drag P accClean acc2F ≥ θ

/-- The condition under which `DragExceeds` is evidence about the population
rather than about this sample: the problem set must be large enough that a
difference of size `ε` is resolvable at confidence `1 - δ`. The card does not
currently compute this, which is recorded against
`Hygiene.Inference.threshold_exceeds_sampling_error::hn_sufficient`. -/
def SampleSufficient (P : Finset ι) (ε δ : ℝ) : Prop :=
  (P.card : ℝ) ≥ Real.log (2 / δ) / (2 * ε ^ 2)

end AIQ.Teams.ContextualDrag
