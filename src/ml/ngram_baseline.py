"""Family-conditioned suffix-backoff n-gram next-step predictor.

For each family, indexes every suffix of length 1..max_order seen in the
training sequences with the counts of the tokens that followed. At inference,
the model looks at the last `max_order` tokens, finds the longest matching
suffix in the index, and returns the top-k most likely next tokens. If no
suffix of length >=1 matches, it backs off to the family's unconditional
token distribution.

This is the "naive baseline" required by docs/EVALUATION_PLAN.md and step 4 of
docs/implementation-plan-en.md.
"""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.data.infineon_loader import Family, FAMILIES, Sequences
from src.ml.beam_search import _beam_complete


@dataclass
class NGramBaseline:
    """Suffix-backoff next-token predictor, one model per family.

    Attributes:
        max_order:   longest suffix length to index (>=1).
        suffix_counts:   family -> suffix(tuple[str, ...]) -> next-token Counter.
        unconditional:   family -> Counter of all tokens (backoff distribution).
    """
    max_order: int = 12
    suffix_counts: dict[Family, dict[tuple[str, ...], Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    unconditional: dict[Family, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def fit(self, sequences_per_family: dict[Family, Sequences]) -> "NGramBaseline":
        for family in FAMILIES:
            if family not in sequences_per_family:
                continue
            for steps in sequences_per_family[family].values():
                self._index_sequence(family, steps)
        return self

    def _index_sequence(self, family: Family, steps: list[str]) -> None:
        n = len(steps)
        for i in range(1, n):
            next_tok = steps[i]
            self.unconditional[family][next_tok] += 1
            for order in range(1, self.max_order + 1):
                start = i - order
                if start < 0:
                    break
                suffix = tuple(steps[start:i])
                self.suffix_counts[family][suffix][next_tok] += 1

    def predict_topk(
        self,
        family: Family,
        prefix: list[str],
        k: int = 5,
    ) -> list[str]:
        """Return the top-k most likely next tokens for the given prefix."""
        return [tok for tok, _ in self.predict_topk_with_scores(family, prefix, k)]

    def predict_topk_with_scores(
        self,
        family: Family,
        prefix: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Top-k tokens with `log(count / total)` scores from the matched suffix.

        Backoff uses the family's unconditional distribution. Needed by beam
        search to compare candidates across different beam expansions.
        """
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}")
        for order in range(min(self.max_order, len(prefix)), 0, -1):
            suffix = tuple(prefix[-order:])
            counts = self.suffix_counts[family].get(suffix)
            if counts:
                total = sum(counts.values())
                return [(tok, math.log(c / total)) for tok, c in counts.most_common(k)]
        counts = self.unconditional[family]
        total = sum(counts.values()) or 1
        return [(tok, math.log(c / total)) for tok, c in counts.most_common(k)]

    def complete(
        self,
        family: Family,
        prefix: list[str],
        *,
        max_steps: int = 200,
        stop_token: str = "SHIP LOT",
        rule_constrained: bool = False,
        candidate_pool: int = 5,
        beam_width: int = 1,
        length_normalize: bool = True,
    ) -> list[str]:
        """Greedy, rule-constrained, or beam-search completion.

        `beam_width=1` (default) preserves the legacy greedy / rule-constrained
        behavior. `beam_width>1` enables beam search: at each step every active
        beam is expanded by its top `candidate_pool` next tokens (filtered by
        the rule validator when `rule_constrained=True`), then the global top
        `beam_width` beams are kept, ranked by cumulative log-prob with
        optional length normalization.

        Rationale for the default `length_normalize=True`: process routes can
        vary 100–151 steps. Without normalization the beam prefers short
        completions that bail early.
        """
        if beam_width > 1:
            return _beam_complete(
                self.predict_topk_with_scores,
                family,
                prefix,
                max_steps=max_steps,
                stop_token=stop_token,
                rule_constrained=rule_constrained,
                candidate_pool=candidate_pool,
                beam_width=beam_width,
                length_normalize=length_normalize,
            )

        out = list(prefix)
        if not rule_constrained:
            for _ in range(max_steps):
                top = self.predict_topk(family, out, k=1)
                if not top:
                    break
                nxt = top[0]
                out.append(nxt)
                if nxt == stop_token:
                    break
            return out

        from src.eval.rule_validator import violation_rules

        base_violations = set(violation_rules(out))
        for _ in range(max_steps):
            cands = self.predict_topk(family, out, k=candidate_pool)
            if not cands:
                break
            chosen: str | None = None
            for cand in cands:
                trial_violations = set(violation_rules(out + [cand]))
                if not (trial_violations - base_violations):
                    chosen = cand
                    base_violations = trial_violations
                    break
            if chosen is None:
                chosen = cands[0]
                base_violations = set(violation_rules(out + [chosen]))
            out.append(chosen)
            if chosen == stop_token:
                break
        return out

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_order": self.max_order,
            "suffix_counts": {
                fam: {k: dict(v) for k, v in sc.items()}
                for fam, sc in self.suffix_counts.items()
            },
            "unconditional": {fam: dict(c) for fam, c in self.unconditional.items()},
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "NGramBaseline":
        with path.open("rb") as f:
            payload = pickle.load(f)
        m = cls(max_order=payload["max_order"])
        for fam, sc in payload["suffix_counts"].items():
            for suffix, counts in sc.items():
                m.suffix_counts[fam][suffix] = Counter(counts)
        for fam, c in payload["unconditional"].items():
            m.unconditional[fam] = Counter(c)
        return m

    def stats(self) -> dict:
        """Compact summary for run logs and metrics.json."""
        return {
            "model": "ngram_suffix_backoff",
            "max_order": self.max_order,
            "indexed_suffixes_per_family": {
                fam: len(sc) for fam, sc in self.suffix_counts.items()
            },
            "vocab_per_family": {
                fam: len(c) for fam, c in self.unconditional.items()
            },
        }

    def stats_json(self) -> str:
        return json.dumps(self.stats(), indent=2, sort_keys=True)
