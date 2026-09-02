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

## What this does not formalize

Most of the card. Named plainly, so nobody mistakes this for a guarantee:

* **There is no probability space here.** `meanOver` is an average over a finite
  set. Nothing below is a statement about a population, so `hoeffdingSize`
  is a number this file defines and never connects to a confidence claim.
* **Nothing says the drag measures contextual interference.** It is the gap
  between two averages; that the injected trajectories are what caused it is an
  experimental-design claim, not a mathematical one.
* **`unpaired_difference_of_no_effect` shows the unpaired comparison CAN
  mislead**, with an explicit witness. It does not show that this pipeline's
  filter does mislead -- that would need the filter's selection mechanism.
* The threshold is nowhere. Whether a measured drag clears one is not a theorem.

## Where a real formalization would go

Put a measure on the problem distribution and state the paired Hoeffding bound
over the per-problem DIFFERENCE sequence, then prove that `hoeffdingSize`
queries suffice for the stated tolerance and confidence. That would turn the
sample-size condition from a defined quantity into an actual guarantee, and it
is the piece that would let a card discharge `hn_sufficient` rather than assume
it. Mathlib's `ProbabilityTheory` sub-Gaussian machinery is the obvious start.
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

/-- Hoeffding's sample size for tolerance `ε` at confidence `1 - δ`: the number
of paired problems needed before a difference of size `ε` is resolvable. A
quantity, not a predicate -- at ε = δ = 0.05 it is about 738. -/
noncomputable def hoeffdingSize (ε δ : ℝ) : ℝ := Real.log (2 / δ) / (2 * ε ^ 2)

/-- **Why the pairing is not pedantry.** If the two accuracies are measured over
DIFFERENT problem sets, a nonzero drag can appear when there is no effect at
all: here a single accuracy function -- so the per-problem difference is
identically zero -- still yields a nonzero difference of means, purely from
which problems each mean was taken over.

This is the error the terminal node exists to avoid, and it is why
`acc_clean` is restricted to the problems that survived the filter rather than
being taken over all of them. -/
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

end AIQ.Teams.ContextualDrag
