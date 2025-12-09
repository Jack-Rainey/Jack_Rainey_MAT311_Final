import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(
    y_true,
    y_pred_baseline,
    y_pred_model,
    baseline_label: str = "Baseline (majority class)",
    model_label: str = "k-NN"
) -> None:
    """
    Plot side-by-side confusion matrices for a baseline model
    and a trained model.

    Parameters
    ----------
    y_true : array-like
        True labels (e.g., y_val).
    y_pred_baseline : array-like
        Predicted labels from the baseline model.
    y_pred_model : array-like
        Predicted labels from the trained model (k-NN in this case).
    baseline_label : str
        Title for the baseline confusion matrix.
    model_label : str
        Title for the trained model confusion matrix.
    """
    conf_baseline = confusion_matrix(y_true, y_pred_baseline)
    conf_model = confusion_matrix(y_true, y_pred_model)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(conf_baseline, annot=True, fmt="d", cmap="Reds", ax=axes[0])
    axes[0].set_title(baseline_label)
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    sns.heatmap(conf_model, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title(model_label)
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_score, label: str) -> float:
    """Plot a ROC curve and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label} (AUC={auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return auc


def plot_performance_comparison(y_test, y_pred_baseline, y_pred_knn) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    baseline_scores = [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline, zero_division=0),
        recall_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_baseline)
    ]
    knn_scores = [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_knn)
    ]
    df = pd.DataFrame({'Metric': metrics, 'k-NN': knn_scores, 'Never Fraud': baseline_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset
    from src.models.train_model import train_models

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    y_test, baseline, knn = train_models(clean)
    plot_confusion_matrices(y_test, baseline, knn)
    plot_performance_comparison(y_test, baseline, knn)
