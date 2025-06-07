import matplotlib.pyplot as plt
import numpy as np


def create_charts():
    """
    Generates and displays four key charts for comparing the performance
    of the baseline and alternative models for the TSP.
    """
    # --- Data from the 'Summary of Best Performing Parameters' sheet ---
    # This data is based on the best run for each model type at each problem size.
    problem_sizes = np.array([20, 50, 100, 200, 500])

    # Best Average Tour Lengths
    baseline_tour_lengths = [4.1690, 6.3447, 8.7508, 12.0735, 12.5700]
    alternative_tour_lengths = [4.1812, 6.3332, 8.7169, 12.1566, 18.6]

    # Optimality Gaps (%) from the 'Optimality Gap Analysis' sheet.
    # The N=500 baseline gap is anomalous and should be discussed as such in the report.
    baseline_gaps = [9.7, 11.3, 12.2, 19.5, -20.9]  # The -20.9% is anomalous
    alternative_gaps = [10.0, 11.1, 11.8, 20.4, 17.0]

    # Average Inference Times (in seconds, from evaluation logs)
    # These are combined from various runs for a representative plot.
    # Note: 2-Opt time dominates, so the times are similar for both models.
    inference_times = [0.006, 0.035, 0.27, 3.3, 35.0]  # N=500 is an estimate based on scaling trends

    # --- Chart Styling ---
    plt.style.use('seaborn-v0_8-whitegrid')
    bar_width = 0.35
    index = np.arange(len(problem_sizes))

    # --- 1. Bar Chart: Best Tour Length Comparison ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    bar1 = ax1.bar(index - bar_width / 2, baseline_tour_lengths, bar_width, label='Baseline Loss',
                   color='cornflowerblue')
    bar2 = ax1.bar(index + bar_width / 2, alternative_tour_lengths, bar_width, label='Alternative Loss', color='salmon')

    ax1.set_xlabel('Problem Size (N)')
    ax1.set_ylabel('Best Average Tour Length')
    ax1.set_title('Comparison of Best Average Tour Lengths')
    ax1.set_xticks(index)
    ax1.set_xticklabels(problem_sizes)
    ax1.legend()

    ax1.bar_label(bar1, fmt='%.2f', padding=3)
    ax1.bar_label(bar2, fmt='%.2f', padding=3)

    # --- 2. Line Chart: Tour Length Scalability ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(problem_sizes, baseline_tour_lengths, marker='o', linestyle='-', label='Baseline Loss',
             color='cornflowerblue')
    ax2.plot(problem_sizes, alternative_tour_lengths, marker='s', linestyle='--', label='Alternative Loss',
             color='salmon')

    # Annotate the anomalous N=500 baseline point
    ax2.annotate('Anomalous Result', xy=(500, 12.57), xytext=(400, 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    ax2.set_xlabel('Problem Size (N)')
    ax2.set_ylabel('Best Average Tour Length')
    ax2.set_title('Scalability of Tour Length')
    ax2.legend()
    ax2.set_xscale('log')  # Use log scale for better visualization of N

    # --- 3. Line Chart: Optimality Gap Scalability ---
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(problem_sizes, baseline_gaps, marker='o', linestyle='-', label='Baseline Loss', color='cornflowerblue')
    ax3.plot(problem_sizes, alternative_gaps, marker='s', linestyle='--', label='Alternative Loss', color='salmon')
    ax3.axhline(0, color='grey', linewidth=0.8, linestyle='--')  # Add a line at 0% gap

    ax3.set_xlabel('Problem Size (N)')
    ax3.set_ylabel('Optimality Gap (%)')
    ax3.set_title('Scalability of Optimality Gap')
    ax3.legend()
    ax3.set_xscale('log')

    # --- 4. Line Chart: Inference Time Scalability ---
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    ax4.plot(problem_sizes, inference_times, marker='o', linestyle='-', color='darkcyan')

    ax4.set_xlabel('Problem Size (N)')
    ax4.set_ylabel('Average Inference Time (seconds)')
    ax4.set_title('Scalability of Inference Time (Heatmap + 2-Opt)')
    ax4.set_yscale('log')  # Log scale for time is essential due to large increase
    ax4.set_xscale('log')

    # --- Display all charts ---
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Ensure you have matplotlib installed: pip install matplotlib
    create_charts()
