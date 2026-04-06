import base64
import csv
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .paths import get_path_to_evals


class BenchmarkReport:
    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, image: Image.Image, ground_truth: str, predicted: str):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        self.records.append(
            {
                "image_base64": img_str,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": ground_truth == predicted,
            }
        )

    def get_accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r["correct"]) / len(self.records)

    def get_majority_class_accuracy(self) -> float:
        if not self.records:
            return 0.0
        from collections import Counter
        counts = Counter(r["ground_truth"] for r in self.records)
        return counts.most_common(1)[0][1] / len(self.records)

    def to_csv(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = str(Path(get_path_to_evals()) / f"benchmark_{timestamp}.csv")

        with open(csv_file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_base64", "ground_truth", "predicted", "correct"]
            )
            writer.writeheader()
            writer.writerows(self.records)

        return csv_file_path

    def get_confusion_matrix_figure(self) -> plt.Figure:
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        ground_truth = [r["ground_truth"] for r in self.records]
        predicted = [r["predicted"] for r in self.records]
        classes = sorted(set(ground_truth + predicted))

        cm = confusion_matrix(ground_truth, predicted, labels=classes)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title("Defect Detection: Predicted vs Actual", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        print(f"Classes: {classes} | Total: {len(ground_truth)} | Correct: {int(np.trace(cm))}")

        return fig
