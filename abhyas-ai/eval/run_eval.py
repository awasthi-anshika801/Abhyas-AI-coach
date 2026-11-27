"""Golden-set check for the answer evaluator.
Strong answers must score higher than medium > weak >= empty. Prints latency stats.
Usage: GEMINI_API_KEY=... python eval/run_eval.py"""
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import llm  # noqa: E402

GOLD = [
    {"q": "Explain the difference between INNER JOIN and LEFT JOIN with an example.",
     "a": {"strong": "INNER JOIN returns only rows with matching keys in both tables. LEFT JOIN returns all rows from the left table and NULLs where the right table has no match. E.g. customers LEFT JOIN orders lists every customer, even those with zero orders, which is how I find customers who never ordered using WHERE o.id IS NULL.",
           "medium": "INNER JOIN gives matching rows and LEFT JOIN gives all rows from the left table too.",
           "weak": "Joins combine tables. Left join is the left one I think.",
           "empty": ""}},
    {"q": "What is overfitting and how do you prevent it?",
     "a": {"strong": "Overfitting is when a model memorizes noise in training data and fails to generalize: low training error, high validation error. Prevent it with more data, regularization (L1/L2, dropout), early stopping, simpler models, cross-validation and feature selection.",
           "medium": "Overfitting is when the model does well on training but badly on test data. Use regularization.",
           "weak": "It means the model is too big.",
           "empty": ""}},
]
ORDER = ["strong", "medium", "weak", "empty"]


def main():
    failed, lat = 0, []
    for g in GOLD:
        scores = {}
        for k in ORDER:
            t0 = time.perf_counter()
            s = llm.evaluate(g["q"], g["a"][k], "Data Analyst", "Generic")
            lat.append((time.perf_counter() - t0) * 1000)
            scores[k] = sum(s[m] for m in ("technical_accuracy", "clarity", "communication", "depth")) / 4
        ok = scores["strong"] > scores["medium"] > scores["weak"] >= scores["empty"]
        failed += not ok
        print(("PASS " if ok else "FAIL ") + g["q"][:50], {k: round(v, 1) for k, v in scores.items()})
    print(f"latency ms: mean={statistics.mean(lat):.0f} max={max(lat):.0f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
