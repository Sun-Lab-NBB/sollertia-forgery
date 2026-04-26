from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import evaluate_bleaching

animal_root = Path("/home/data/Data/MaalstroomicFlow/void/11")
session_paths = tuple(sorted(path for path in animal_root.iterdir() if path.is_dir()))

print(f"Discovered {len(session_paths)} sessions for animal 11:")
for path in session_paths:
    print(f"  {path.name}")

report = evaluate_bleaching(session_paths=session_paths)

print()
print(f"Registered cells across all sessions: {report.cell_count}")
print(f"Population F0 trend: {[float(value) for value in report.f0_population_trend]}")
print(f"F0 fractional loss (last vs first): {report.f0_fractional_loss:.1%}")
if report.f0_decay_fit.fit_succeeded:
    print(
        f"F0 decay fit: amplitude={report.f0_decay_fit.amplitude:.2f}, "
        f"tau={report.f0_decay_fit.tau_days:.2f} days, offset={report.f0_decay_fit.offset:.2f}"
    )
else:
    print("F0 decay fit did not converge.")
print(f"SNR population trend: {[float(value) for value in report.snr_population_trend]}")
print(f"SNR paired Wilcoxon p-values vs session 0: {[float(value) for value in report.snr_paired_p_values]}")

print()
print("Per-session within-session fractional drops:")
for session in report.sessions:
    print(
        f"  day {session.days_since_first:5.2f} | within-session drop {session.within_session_fractional_drop:6.1%} | "
        f"sampling rate {session.sampling_rate_hz:5.2f} Hz | {session.session_path.name}"
    )

print()
if report.flagged_sessions:
    print(f"Flagged sessions ({len(report.flagged_sessions)}):")
    for path in report.flagged_sessions:
        print(f"  {path.name}")
else:
    print("No sessions exceeded any threshold.")

output_directory = Path(__file__).parent / "bleaching_animal_11"
output_directory.mkdir(exist_ok=True)
report.plot_baseline_trend().savefig(output_directory / "baseline_trend.png", bbox_inches="tight")
report.plot_within_session().savefig(output_directory / "within_session.png", bbox_inches="tight")
report.plot_snr_distributions().savefig(output_directory / "snr_distributions.png", bbox_inches="tight")
plt.close("all")

print()
print(f"Diagnostic figures saved to: {output_directory}")
