import base64
import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Self

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .paths import get_path_to_evals


class EvalReport:
    def __init__(self):
        self.records = []

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

    def to_csv(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_name = f"predictions_{timestamp}.csv"
        csv_file_path = str(Path(get_path_to_evals()) / csv_file_name)

        with open(csv_file_path, "w", newline="") as csvfile:
            fieldnames = ["image_base64", "ground_truth", "predicted", "correct"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(self.records)

        return csv_file_path

    @classmethod
    def from_csv(cls, file_name: str) -> Self:
        file_path = str(Path(get_path_to_evals()) / file_name)
        report = cls()
        csv.field_size_limit(10 * 1024 * 1024)
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            report.records = []
            for row in reader:
                row["correct"] = row["correct"].lower() == "true"
                report.records.append(row)
        return report

    @classmethod
    def from_last_csv(cls) -> Self:
        evals_path = Path(get_path_to_evals())
        csv_files = sorted(
            evals_path.glob("predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not csv_files:
            raise FileNotFoundError("No CSV files found in evals directory")

        return cls.from_csv(csv_files[0].name)

    def print(self, only_misclassified: bool | None = False):
        records_to_show = (
            [record for record in self.records if not record["correct"]]
            if only_misclassified
            else self.records
        )

        n_images = len(records_to_show)
        # n_images = 2
        # print('N images to show: ', n_images)

        cols = 4
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten() if n_images >= 1 else [axes]

        for idx, record in enumerate(records_to_show):
            img_data = base64.b64decode(record["image_base64"])
            img = Image.open(BytesIO(img_data))

            # breakpoint()

            axes[idx].imshow(img)
            axes[idx].axis("off")

            color = "green" if record["correct"] else "red"
            title = f"GT: {record['ground_truth']}\nPred: {record['predicted']}"
            axes[idx].set_title(title, color=color, fontweight="bold")

        for idx in range(n_images, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.close(fig)
        return fig

    def get_accuracy(self) -> float:
        if not self.records:
            return 0.0
        correct_count = sum(1 for record in self.records if record["correct"])
        return correct_count / len(self.records)

    def print_confusion_matrix(self):

        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # Extract ground truth and predicted labels
        ground_truth = [record["ground_truth"] for record in self.records]
        predicted = [record["predicted"] for record in self.records]

        # Get unique classes and sort them
        classes = sorted(list(set(ground_truth + predicted)))

        # Create confusion matrix
        cm = confusion_matrix(ground_truth, predicted, labels=classes)

        # Create the visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Number of Predictions"},
        )

        plt.title(
            "Confusion Matrix: Predicted vs Actual Car Makers",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Actual Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Print some statistics
        print(f"Number of classes: {len(classes)}")
        print(f"Classes: {classes}")
        print(f"Total predictions: {len(ground_truth)}")
        print(f"Correct predictions: {np.trace(cm)}")
        print(f"Accuracy: {np.trace(cm) / len(ground_truth):.3f}")


if __name__ == "__main__":
    from .report import EvalReport

    eval_report = EvalReport.from_last_csv()
    # eval_report = EvalReport.from_csv("predictions_20250924_151458.csv")
    print(f"Loaded {len(eval_report.records)} records from the latest CSV")

    eval_report.print(only_misclassified=True)

    eval_report.print_confusion_matrix()
