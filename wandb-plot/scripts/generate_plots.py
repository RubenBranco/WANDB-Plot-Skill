#!/usr/bin/env python3
"""Generate line plots from W&B run metrics using matplotlib.

This script generates publication-quality line plots from metric data,
with intelligent axis scaling and optional smoothing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from scripts.wandb_utils import (
    get_run,
    setup_logging,
    WandBAuthError,
    progress_wrap,
    resolve_output_dir,
    safe_filename,
    write_metadata_json,
)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: str,
    smooth: Optional[int] = None
):
    """
    Generate a single metric plot.

    Args:
        df: DataFrame with metric data
        metric: Name of metric column to plot
        output_path: Path to save plot
        smooth: Optional rolling average window size

    Raises:
        ValueError: If metric not in DataFrame
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in data")

    plt.figure(figsize=(10, 6))

    # Determine x-axis
    if '_step' in df.columns:
        x_data = df['_step']
        x_label = 'Step'
    elif '_timestamp' in df.columns:
        x_data = df['_timestamp']
        x_label = 'Timestamp'
    else:
        x_data = df.index
        x_label = 'Index'

    # Get y data
    y_data = df[metric]

    # Remove NaN values
    mask = ~y_data.isna()
    x_data = x_data[mask]
    y_data = y_data[mask]

    if len(y_data) == 0:
        raise ValueError(f"Metric '{metric}' has no valid data points")

    # Apply smoothing if requested
    if smooth and smooth > 1:
        # Convert to series for rolling
        y_series = pd.Series(y_data.values)
        y_smoothed = y_series.rolling(window=smooth, min_periods=1).mean()
        y_data_smoothed = y_smoothed.values

        # Plot both original (faint) and smoothed
        plt.plot(x_data, y_data, linewidth=1, alpha=0.3, color='blue', label='Original')
        plt.plot(x_data, y_data_smoothed, linewidth=2, alpha=0.9, color='blue', label=f'Smoothed (window={smooth})')
        plt.legend()
    else:
        # Plot original data
        plt.plot(x_data, y_data, linewidth=2, alpha=0.8)

    # Labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} over time', fontsize=14, pad=20)

    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Intelligent y-axis scaling
    if 'loss' in metric.lower():
        # Use log scale for loss metrics if values span multiple orders of magnitude
        positive = y_data[y_data > 0]
        if not positive.empty:
            y_min, y_max = positive.min(), positive.max()
        else:
            y_min, y_max = None, None
        if y_min and y_max and y_min > 0 and (y_max / y_min) > 10:  # Span >1 order
            plt.yscale('log')
            plt.ylabel(f'{metric} (log scale)', fontsize=12)
    elif 'acc' in metric.lower() or 'accuracy' in metric.lower():
        # For accuracy metrics, check if in 0-1 range
        if y_data.max() <= 1.0 and y_data.min() >= 0.0:
            plt.ylim(-0.05, 1.05)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")


def generate_plots(
    entity_project: str,
    run_id: str,
    metrics: List[str],
    full_resolution: bool = False,
    output_dir: Optional[str] = None,
    smooth: Optional[int] = None
) -> List[str]:
    """
    Generate plots from metric data.

    Args:
        entity_project: Project in format "entity/project" or "project"
        run_id: Run ID or name
        metrics: List of metric names to plot
        full_resolution: If True, use scan_history() for all data points
        output_dir: Optional custom output directory
        smooth: Optional rolling average window size

    Returns:
        List of generated file paths

    Raises:
        WandBAuthError: If not authenticated
        ValueError: If run or metrics not found

    Example:
        >>> plots = generate_plots(
        ...     "my-org/my-project",
        ...     "abc123",
        ...     ["loss", "accuracy"],
        ...     smooth=10
        ... )
        >>> print(f"Generated {len(plots)} plots")
    """
    logger = logging.getLogger(__name__)
    run = get_run(entity_project, run_id)

    # Fetch history data
    keys = list(dict.fromkeys(metrics + ["_step", "_timestamp"]))
    try:
        if full_resolution:
            print("Fetching full resolution data (this may take a while)...")
            logger.info("Fetching full resolution data for run %s", run_id)
            # scan_history() returns all data points
            history_data = []
            try:
                iterator = run.scan_history(keys=keys)
            except TypeError:
                iterator = run.scan_history()
            for row in progress_wrap(iterator, "Fetching history"):
                history_data.append(row)
            df = pd.DataFrame(history_data)
        else:
            # history() returns sampled data (~500 points)
            print("Fetching sampled data...")
            logger.info("Fetching sampled data for run %s", run_id)
            try:
                df = run.history(keys=keys)
            except TypeError:
                df = run.history()
    except Exception as e:
        raise ValueError(f"Error fetching run history: {str(e)}") from e

    if df.empty:
        raise ValueError("Run has no history data")

    # Verify metrics exist
    available_metrics = [col for col in df.columns if not col.startswith('_')]
    missing_metrics = [m for m in metrics if m not in df.columns]

    if missing_metrics:
        error_msg = f"Metrics not found: {', '.join(missing_metrics)}\n"
        error_msg += f"Available metrics: {', '.join(available_metrics[:10])}"
        if len(available_metrics) > 10:
            error_msg += f" (+{len(available_metrics) - 10} more)"
        raise ValueError(error_msg)

    # Determine output directory (only after validation)
    output_path = resolve_output_dir(entity_project, run, output_dir=output_dir)

    # Generate plots
    generated_paths = []

    for metric in progress_wrap(metrics, "Generating plots"):
        # Create safe filename
        safe_metric_name = safe_filename(metric)
        output_file = output_path / f"{safe_metric_name}.png"

        try:
            plot_metric(df, metric, str(output_file), smooth=smooth)
            generated_paths.append(str(output_file))
        except Exception as e:
            logger.warning("Failed to generate plot for '%s': %s", metric, e)
            print(f"Warning: Failed to generate plot for '{metric}': {e}", file=sys.stderr)
            continue

    # Create metadata file
    if generated_paths:
        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "entity": getattr(run, "entity", None),
            "project": getattr(run, "project", None),
            "generation_timestamp": datetime.now().isoformat(),
            "full_resolution": full_resolution,
            "smoothing_window": smooth,
            "metrics_plotted": metrics,
            "plots_generated": [str(Path(p).name) for p in generated_paths],
            "plot_count": len(generated_paths),
            "data_points": len(df)
        }

        write_metadata_json(output_path, metadata, merge=True)

    return generated_paths


def main():
    """Main entry point for the script."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate line plots from W&B run metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate loss and accuracy plots
  %(prog)s my-org/my-project abc123 --metrics loss,accuracy

  # Full resolution with smoothing
  %(prog)s my-org/my-project abc123 --metrics loss --full-res --smooth 10

  # Multiple metrics
  %(prog)s my-org/my-project abc123 --metrics train/loss,val/loss,train/accuracy

  # Custom output directory
  %(prog)s my-org/my-project abc123 --metrics loss --output ./my_plots
        """
    )

    parser.add_argument(
        "entity_project",
        help="W&B project in format 'entity/project' or 'project'"
    )
    parser.add_argument(
        "run_id",
        help="Run ID or name"
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated list of metrics to plot (e.g., 'loss,accuracy')"
    )
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Use full resolution data (slower, all data points)"
    )
    parser.add_argument(
        "--output",
        help="Custom output directory (default: wandb_plots/<entity>_<project>/<run>/)"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        help="Apply rolling average smoothing with specified window size"
    )

    args = parser.parse_args()

    # Parse metrics
    metrics_list = [m.strip() for m in args.metrics.split(',') if m.strip()]

    if not metrics_list:
        print("Error: No metrics specified", file=sys.stderr)
        return 1

    try:
        # Generate plots
        generated = generate_plots(
            args.entity_project,
            args.run_id,
            metrics_list,
            full_resolution=args.full_res,
            output_dir=args.output,
            smooth=args.smooth
        )

        if generated:
            print(f"\nSuccessfully generated {len(generated)} plot(s):")
            for path in generated:
                print(f"  - {path}")
        else:
            print("\nNo plots were generated.")

        return 0

    except WandBAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
