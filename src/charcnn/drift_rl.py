"""CPU-only MMD, ADWIN-style error monitoring, and threshold-only Q-learning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, f1_score


def _subsample(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return values if len(values) <= n else values[rng.choice(len(values), n, replace=False)]


def rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Biased RBF MMD²; fixed sample sizes keep the CPU stage lightweight."""
    xx = np.exp(-gamma * (np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2 * x @ x.T))
    yy = np.exp(-gamma * (np.sum(y * y, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * y @ y.T))
    xy = np.exp(-gamma * (np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * x @ y.T))
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def mmd_report(reference_path: Path, test_embedding_path: Path, output: Path, seed: int) -> dict[str, dict[str, float | bool]]:
    rng = np.random.default_rng(seed)
    reference = np.load(reference_path)["embeddings"].astype(np.float32)
    test = np.load(test_embedding_path)
    sample_n = min(2_000, len(reference))
    ref_sample = _subsample(reference, sample_n, rng)
    pair = np.concatenate([ref_sample[: sample_n // 2], ref_sample[sample_n // 2 :]], axis=0)
    distances = np.sum((pair[::2] - pair[1::2]) ** 2, axis=1)
    gamma = float(1.0 / max(np.median(distances), 1e-8))
    null_scores = []
    for _ in range(20):
        indices = rng.choice(len(reference), size=2 * sample_n, replace=len(reference) < 2 * sample_n)
        null_scores.append(rbf_mmd2(reference[indices[:sample_n]], reference[indices[sample_n:]], gamma))
    threshold = float(np.quantile(null_scores, 0.95))
    result: dict[str, dict[str, float | bool]] = {}
    for key in sorted(test.files):
        current = _subsample(test[key].astype(np.float32), sample_n, rng)
        score = rbf_mmd2(ref_sample, current, gamma)
        month = key.removeprefix("month_").replace("_", "-")
        result[month] = {"mmd2": score, "alert_threshold": threshold, "drift_alert": bool(score > threshold), "sample_rows": int(len(current))}
    (output / "mmd_monthly.json").write_text(json.dumps({"gamma": gamma, "null_threshold_95": threshold, "months": result}, indent=2), encoding="utf-8")
    return result


class ADWINErrorMonitor:
    """Lightweight ADWIN-style adaptive window for a binary error stream.

    It evaluates Hoeffding cuts periodically and discards stale observations on
    a detected mean shift. This is intentionally CPU-friendly for Kaggle.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 128, clock: int = 256, max_window: int = 20_000):
        self.delta, self.min_window, self.clock, self.max_window = delta, min_window, clock, max_window
        self.window: deque[int] = deque()
        self.count = 0

    def update(self, error: int) -> dict | None:
        self.window.append(int(error)); self.count += 1
        if len(self.window) > self.max_window: self.window.popleft()
        if len(self.window) < 2 * self.min_window or self.count % self.clock: return None
        values = np.fromiter(self.window, dtype=np.float64)
        cumulative = np.cumsum(values)
        cuts = np.unique(np.linspace(self.min_window, len(values) - self.min_window, num=20, dtype=int))
        for cut in cuts:
            n0, n1 = cut, len(values) - cut
            mean0, mean1 = cumulative[cut - 1] / n0, (cumulative[-1] - cumulative[cut - 1]) / n1
            epsilon = np.sqrt(0.5 * np.log(4 / self.delta) * (1 / n0 + 1 / n1))
            if abs(mean0 - mean1) > epsilon:
                for _ in range(cut): self.window.popleft()
                return {"old_error_mean": float(mean0), "new_error_mean": float(mean1), "cut": int(cut), "window_after": len(self.window)}
        return None


def adwin_report(stream, output: Path) -> list[dict]:
    detector, alerts = ADWINErrorMonitor(), []
    for index, (item_date, error) in enumerate(zip(stream["date"], stream["error"]), start=1):
        event = detector.update(int(error))
        if event: alerts.append({"index": index, "date": str(item_date), **event})
    with (output / "adwin_alerts.json").open("w", encoding="utf-8") as handle: json.dump(alerts, handle, indent=2)
    return alerts


def threshold_q_learning(stream, monthly_mmd: dict, adwin_alerts: list[dict], baseline_threshold: float, output: Path, seed: int) -> dict:
    """Daily online Q-learning. State/actions affect only the decision threshold."""
    dates = np.asarray(stream["date"], dtype="datetime64[ns]")
    labels, probabilities = np.asarray(stream["true_label"]), np.asarray(stream["phishing_probability"])
    days = np.asarray([str(date)[:10] for date in dates])
    actions = np.asarray([-0.03, 0.0, 0.03])
    q: dict[tuple, np.ndarray] = {}
    threshold, alpha, discount, epsilon = baseline_threshold, 0.20, 0.90, 0.10
    rng, logs = np.random.default_rng(seed), []
    # At the start of a day, the policy may use ADWIN alerts observed through
    # the preceding day. Same-day labels are revealed after prediction and are
    # used only for the next decision, preserving the online sequence.
    adwin_days = {str(event["date"])[:10] for event in adwin_alerts}
    previous_reward, previous_day_had_adwin = 0.0, 0
    for day in np.unique(days):
        mask = days == day; month = day[:7]; drift = int(monthly_mmd[month]["drift_alert"])
        state = (round(threshold, 2), drift, previous_day_had_adwin, int(previous_reward < 0.85))
        values = q.setdefault(state, np.zeros(len(actions)))
        action_index = int(rng.integers(len(actions))) if rng.random() < epsilon else int(np.argmax(values))
        threshold = float(np.clip(threshold + actions[action_index], 0.10, 0.90))
        prediction = (probabilities[mask] >= threshold).astype(int); truth = labels[mask]
        tp = int(((prediction == 1) & (truth == 1)).sum()); fn = int(((prediction == 0) & (truth == 1)).sum()); fp = int(((prediction == 1) & (truth == 0)).sum())
        # This reward explicitly matches the primary reported metric. It
        # balances benign and phishing performance instead of over-rewarding
        # a low threshold's recall gains at the expense of false positives.
        reward = float(f1_score(truth, prediction, average="macro", zero_division=0))
        day_had_adwin = int(day in adwin_days)
        next_state = (round(threshold, 2), drift, day_had_adwin, int(reward < 0.85)); q.setdefault(next_state, np.zeros(len(actions)))
        values[action_index] += alpha * (reward + discount * q[next_state].max() - values[action_index])
        previous_reward = reward
        logs.append({"date": day, "mmd_alert": drift, "previous_day_adwin_alert": previous_day_had_adwin, "action": float(actions[action_index]), "threshold": threshold, "reward_macro_f1": reward, "tp": tp, "fn": fn, "fp": fp, "rows": int(len(truth))})
        previous_day_had_adwin = day_had_adwin
    final_prediction = np.empty_like(labels)
    for row in logs:
        mask = days == row["date"]; final_prediction[mask] = (probabilities[mask] >= row["threshold"]).astype(int)
    report = classification_report(labels, final_prediction, labels=[0, 1], target_names=["benign", "phishing"], output_dict=True, zero_division=0)
    report["macro_f1"] = report["macro avg"]["f1-score"]
    with (output / "rl_threshold_policy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    (output / "rl_adapted_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-artifacts", required=True)
    parser.add_argument("--frozen-results", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/charcnn/drift_rl")
    args = parser.parse_args(argv)
    artifacts, frozen, output = Path(args.training_artifacts), Path(args.frozen_results), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((artifacts / "config.yaml").read_text(encoding="utf-8")) if False else None
    # YAML config is not needed beyond its reproducibility copy in this stage.
    shutil_copy = output / "source_artifact_note.json"
    shutil_copy.write_text(json.dumps({"cnn_weights": "frozen", "label_feedback": "immediate; ADWIN errors and RL rewards update after each daily batch"}, indent=2), encoding="utf-8")
    seed = 2025
    mmd = mmd_report(artifacts / "validation_mmd_reference_embeddings.npz", frozen / "frozen_test_monthly_embeddings.npz", output, seed)
    stream = pq.read_table(frozen / "frozen_test_error_stream.parquet").to_pydict()
    alerts = adwin_report(stream, output)
    threshold = json.loads((artifacts / "chosen_threshold.json").read_text(encoding="utf-8"))["threshold"]
    rl_report = threshold_q_learning(stream, mmd, alerts, threshold, output, seed)
    baseline = json.loads((frozen / "frozen_test_metrics.json").read_text(encoding="utf-8"))
    (output / "baseline_vs_rl.json").write_text(json.dumps({"frozen_fixed_threshold_macro_f1": baseline["macro_f1"], "rl_adapted_macro_f1": rl_report["macro_f1"], "adwin_alert_count": len(alerts)}, indent=2), encoding="utf-8")
    print((output / "baseline_vs_rl.json").read_text())


if __name__ == "__main__":
    main()
