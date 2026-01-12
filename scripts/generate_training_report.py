#!/usr/bin/env python3
"""
Automatic Training Report Generator

Generates a comprehensive training report with:
- Training/validation metrics over time
- Loss curves (per component for student)
- Learning rate schedule visualization
- Model compression statistics
- Training duration and cost estimation
- TensorBoard embedding visualizations
- Final model performance summary

Usage:
    python generate_training_report.py --mlflow-uri <uri> --run-id <id> --output report.md
    python generate_training_report.py --mlflow-uri <uri> --experiment <name> --output report.md
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import mlflow
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description='Generate training report from MLflow data')
    parser.add_argument('--mlflow-uri', type=str, default='file:///opt/ml/output/mlflow',
                        help='MLflow tracking URI')
    parser.add_argument('--run-id', type=str, help='MLflow run ID')
    parser.add_argument('--experiment', type=str, help='MLflow experiment name (uses latest run)')
    parser.add_argument('--output', type=str, default='training_report.md',
                        help='Output report file path')
    parser.add_argument('--figures-dir', type=str, default='report_figures',
                        help='Directory to save figures')
    return parser.parse_args()


def get_run(mlflow_uri: str, run_id: str = None, experiment: str = None):
    """Get MLflow run by ID or latest from experiment."""
    mlflow.set_tracking_uri(mlflow_uri)

    if run_id:
        return mlflow.get_run(run_id)

    if experiment:
        exp = mlflow.get_experiment_by_name(experiment)
        if not exp:
            raise ValueError(f"Experiment '{experiment}' not found")

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            raise ValueError(f"No runs found in experiment '{experiment}'")

        return mlflow.get_run(runs.iloc[0]['run_id'])

    raise ValueError("Must provide either --run-id or --experiment")


def get_metrics_history(client, run_id: str, metric_name: str) -> Tuple[List[int], List[float]]:
    """Get metric history as (steps, values) lists."""
    history = client.get_metric_history(run_id, metric_name)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    return steps, values


def plot_training_curves(client, run_id: str, params: Dict, figures_dir: Path, training_type: str):
    """Plot training and validation curves."""
    figures_dir.mkdir(exist_ok=True)

    if training_type == 'teacher':
        # Teacher metrics
        metrics_to_plot = [
            ('train_loss', 'Training Loss'),
            ('train_accuracy', 'Training Accuracy'),
            ('val_top1_accuracy', 'Validation Top-1 Accuracy'),
            ('val_top5_accuracy', 'Validation Top-5 Accuracy'),
            ('learning_rate', 'Learning Rate'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('DINOv3 Teacher Training Progress', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, (metric_name, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            try:
                steps, values = get_metrics_history(client, run_id, metric_name)
                if values:
                    ax.plot(steps, values, linewidth=2)
                    ax.set_title(title, fontweight='bold')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(title)
                    ax.grid(True, alpha=0.3)

                    # Add phase separator if applicable
                    if 'epochs_frozen' in params:
                        frozen_epochs = int(params['epochs_frozen'])
                        ax.axvline(x=frozen_epochs, color='red', linestyle='--',
                                   label='Phase 1→2', alpha=0.7)
                        ax.legend()
            except Exception as e:
                ax.text(0.5, 0.5, f'No data: {metric_name}',
                       ha='center', va='center', transform=ax.transAxes)

        # Hide extra subplot
        axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(figures_dir / 'teacher_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    else:  # student
        # Student metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Student Distillation Training Progress', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        # Total loss
        ax = axes[0]
        try:
            steps, values = get_metrics_history(client, run_id, 'total_loss')
            if values:
                ax.plot(steps, values, linewidth=2, color='black', label='Total')
                ax.set_title('Distillation Loss (Total)', fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
        except Exception:
            pass

        # Loss components
        ax = axes[1]
        loss_components = [
            ('feature_loss', 'Feature (35%)', 'blue'),
            ('kl_loss', 'KL (25%)', 'orange'),
            ('attention_loss', 'Attention (25%)', 'green'),
            ('highfreq_loss', 'High-Freq (15%)', 'red'),
        ]
        for metric_name, label, color in loss_components:
            try:
                steps, values = get_metrics_history(client, run_id, metric_name)
                if values:
                    ax.plot(steps, values, linewidth=2, label=label, color=color)
            except Exception:
                pass

        ax.set_title('Distillation Loss Components', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Cosine similarity
        ax = axes[2]
        try:
            steps, values = get_metrics_history(client, run_id, 'cosine_similarity')
            if values:
                ax.plot(steps, values, linewidth=2, color='purple')
                ax.set_title('Teacher-Student Cosine Similarity', fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Similarity')
                ax.grid(True, alpha=0.3)
        except Exception:
            pass

        # Learning rate
        ax = axes[3]
        try:
            steps, values = get_metrics_history(client, run_id, 'learning_rate')
            if values:
                ax.plot(steps, values, linewidth=2, color='brown')
                ax.set_title('Learning Rate Schedule', fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('LR')
                ax.grid(True, alpha=0.3)
        except Exception:
            pass

        plt.tight_layout()
        plt.savefig(figures_dir / 'student_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def calculate_cost(duration_seconds: float, instance_type: str) -> float:
    """Calculate training cost based on duration and instance type."""
    # Pricing per hour (as of 2025)
    instance_pricing = {
        'ml.p4d.24xlarge': 32.77,  # 8x A100 80GB
        'ml.p4de.24xlarge': 40.97,  # 8x A100 80GB (on-demand)
        'ml.g5.xlarge': 1.006,      # 1x A10G 24GB
        'ml.g5.12xlarge': 5.672,    # 4x A10G 24GB
    }

    price_per_hour = instance_pricing.get(instance_type, 0)
    hours = duration_seconds / 3600
    return price_per_hour * hours


def generate_report(run, figures_dir: Path, output_path: Path):
    """Generate markdown training report."""
    params = run.data.params
    metrics = run.data.metrics
    info = run.info

    # Detect training type
    training_type = 'teacher' if 'dinov3' in params.get('model', '').lower() else 'student'

    # Calculate duration
    start_time = datetime.fromtimestamp(info.start_time / 1000)
    end_time = datetime.fromtimestamp(info.end_time / 1000) if info.end_time else datetime.now()
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]  # Remove microseconds

    # Calculate cost
    instance_type = params.get('instance_type', 'ml.p4d.24xlarge')
    cost = calculate_cost(duration.total_seconds(), instance_type)

    # Generate report
    report = []
    report.append(f"# Training Report: {training_type.title()}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n---\n")

    # Run information
    report.append(f"## Run Information")
    report.append(f"- **Run ID**: `{info.run_id}`")
    report.append(f"- **Status**: {info.status}")
    report.append(f"- **Start Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **End Time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Duration**: {duration_str}")
    report.append(f"- **Instance Type**: {instance_type}")
    report.append(f"- **Estimated Cost**: **${cost:.2f}**")
    report.append("")

    # Hyperparameters
    report.append(f"## Hyperparameters")
    report.append("")
    if training_type == 'teacher':
        report.append(f"| Parameter | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| Model | {params.get('model', 'N/A')} |")
        report.append(f"| Embedding Dim | {params.get('embedding_dim', 'N/A')} |")
        report.append(f"| Epochs (Frozen) | {params.get('epochs_frozen', 'N/A')} |")
        report.append(f"| Epochs (Unfrozen) | {params.get('epochs_unfrozen', 'N/A')} |")
        report.append(f"| Batch Size | {params.get('batch_size', 'N/A')} |")
        report.append(f"| LR (Frozen) | {params.get('lr_frozen', 'N/A')} |")
        report.append(f"| LR (Unfrozen) | {params.get('lr_unfrozen', 'N/A')} |")
        report.append(f"| ArcFace Margin | {params.get('arcface_margin', 'N/A')} |")
        report.append(f"| ArcFace Scale | {params.get('arcface_scale', 'N/A')} |")
        report.append(f"| Num Classes | {params.get('num_classes', 'N/A')} |")
    else:
        report.append(f"| Parameter | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| Student Model | {params.get('student_model', 'N/A')} |")
        report.append(f"| Teacher Model | {params.get('teacher_model', 'N/A')} |")
        report.append(f"| Stage | {params.get('stage', 'N/A')} |")
        report.append(f"| Embedding Dim | {params.get('embedding_dim', 'N/A')} |")
        report.append(f"| Epochs | {params.get('epochs', 'N/A')} |")
        report.append(f"| Batch Size | {params.get('batch_size', 'N/A')} |")
        report.append(f"| Learning Rate | {params.get('lr', 'N/A')} |")
        report.append(f"| α (Feature) | {params.get('alpha_feature', 'N/A')} (35%) |")
        report.append(f"| α (KL) | {params.get('alpha_kl', 'N/A')} (25%) |")
        report.append(f"| α (Attention) | {params.get('alpha_attention', 'N/A')} (25%) |")
        report.append(f"| α (High-Freq) | {params.get('alpha_highfreq', 'N/A')} (15%) |")

    report.append("")

    # Model statistics
    report.append(f"## Model Statistics")
    report.append("")
    if training_type == 'teacher':
        total_params = params.get('total_params', 'N/A')
        phase1_params = params.get('trainable_params_phase1', 'N/A')
        phase2_params = params.get('trainable_params_phase2', 'N/A')

        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Total Parameters | {total_params:,} if isinstance(total_params, int) else total_params |")
        report.append(f"| Trainable (Phase 1) | {phase1_params:,} if isinstance(phase1_params, int) else phase1_params |")
        report.append(f"| Trainable (Phase 2) | {phase2_params:,} if isinstance(phase2_params, int) else phase2_params |")
    else:
        student_params = params.get('student_total_params', 'N/A')
        teacher_params = 304_000_000  # DINOv3-ViT-L/16

        if isinstance(student_params, (int, float)):
            compression_ratio = (1 - student_params / teacher_params) * 100
            report.append(f"| Metric | Value |")
            report.append(f"|--------|-------|")
            report.append(f"| Student Parameters | {int(student_params):,} (28M) |")
            report.append(f"| Teacher Parameters | {teacher_params:,} (304M) |")
            report.append(f"| **Compression** | **{compression_ratio:.1f}% reduction** |")
            report.append(f"| **Size Ratio** | **{teacher_params/student_params:.1f}x smaller** |")

    report.append("")

    # Performance metrics
    report.append(f"## Final Performance")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")

    if training_type == 'teacher':
        for metric in ['best_val_top1_phase1', 'best_val_top1_phase2', 'val_top1_accuracy', 'val_top5_accuracy']:
            if metric in metrics:
                value = metrics[metric]
                if 'accuracy' in metric:
                    report.append(f"| {metric.replace('_', ' ').title()} | {value*100:.2f}% |")
                else:
                    report.append(f"| {metric.replace('_', ' ').title()} | {value:.4f} |")
    else:
        for metric in ['cosine_similarity', 'mse', 'total_loss']:
            if metric in metrics:
                value = metrics[metric]
                report.append(f"| {metric.replace('_', ' ').title()} | {value:.4f} |")

    report.append("")

    # Training curves
    report.append(f"## Training Curves")
    report.append("")
    if training_type == 'teacher':
        report.append(f"![Training Curves](report_figures/teacher_training_curves.png)")
    else:
        report.append(f"![Training Curves](report_figures/student_training_curves.png)")

    report.append("")

    # TensorBoard
    report.append(f"## Visualizations")
    report.append("")
    report.append(f"TensorBoard logs available at: `/opt/ml/output/tensorboard/`")
    report.append("")
    report.append(f"To view embeddings:")
    report.append(f"```bash")
    report.append(f"tensorboard --logdir=/opt/ml/output/tensorboard/")
    report.append(f"```")
    report.append("")

    # Next steps
    report.append(f"## Next Steps")
    report.append("")
    if training_type == 'teacher':
        report.append(f"1. ✅ Teacher model training complete")
        report.append(f"2. 🚀 **Launch student distillation training** (Stage 1 + Stage 2)")
        report.append(f"3. Export student to ONNX for edge deployment")
        report.append(f"4. Compile with Hailo SDK for Raspberry Pi + Hailo-8L")
    else:
        stage = params.get('stage', 'unknown')
        if stage == 'stage1':
            report.append(f"1. ✅ Stage 1 (general distillation) complete")
            report.append(f"2. 🚀 **Launch Stage 2 training** (task-specific fine-tuning)")
        else:
            report.append(f"1. ✅ Student distillation complete (Stage 1 + Stage 2)")
            report.append(f"2. 🚀 **Export to ONNX** for edge deployment")
            report.append(f"3. Compile with Hailo SDK for Raspberry Pi + Hailo-8L")
            report.append(f"4. Build reference database with embeddings")

    report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✅ Report generated: {output_path}")


def main():
    args = parse_args()

    print(f"Generating training report...")
    print(f"MLflow URI: {args.mlflow_uri}")

    # Get run
    run = get_run(args.mlflow_uri, args.run_id, args.experiment)
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")

    # Create figures directory
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    # Detect training type
    params = run.data.params
    training_type = 'teacher' if 'dinov3' in params.get('model', '').lower() else 'student'

    # Plot training curves
    print(f"Generating training curves...")
    client = mlflow.tracking.MlflowClient(tracking_uri=args.mlflow_uri)
    plot_training_curves(client, run.info.run_id, params, figures_dir, training_type)

    # Generate report
    print(f"Generating report...")
    output_path = Path(args.output)
    generate_report(run, figures_dir, output_path)

    print(f"\n" + "=" * 60)
    print(f"✅ Training report complete!")
    print(f"=" * 60)
    print(f"Report: {output_path}")
    print(f"Figures: {figures_dir}/")
    print(f"=" * 60)


if __name__ == '__main__':
    main()
