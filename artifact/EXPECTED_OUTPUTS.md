# Expected Outputs (Checklist)

After running `bash scripts/run_stage012345_full_eval.sh 2`:

## Stage 0
- `results/w1_synth_small/metrics_rank*.csv`
- `results/w1_synth_small/events_rank*.jsonl`
- `results/w1_synth_small/metrics_merged.csv`
- `results/w1_synth_small/figures/*.pdf`
- `results/w1_synth_small/tables/*.csv`

## Stage 2 and Stage 3 combined workloads
For each combined run directory under:
- `results/stage2_*_combined`
- `results/stage3_*_combined`

required files:
- `metrics_merged.csv`
- `figures/fig1_bytes_vs_rmax.pdf`
- `figures/fig2_latency_quantiles.pdf`
- `figures/fig3_xva_error_cdf.pdf`
- `figures/fig4_fallback_vs_shock.pdf`
- `tables/table1_main.csv`
- `tables/table2_lower_bound.csv`

## Stage 4
- `results/stage4_final/summary_stage_metrics.csv`
- `results/stage4_final/paper_tables/table_stage2_vs_stage3_delta.csv`
- `results/stage4_final/paper_tables/table_stage3_final_story.csv`
- `results/stage4_final/paper_figures/fig_stage2_vs_stage3_bytes_reduction.pdf`
- `results/stage4_final/paper_figures/fig_stage2_vs_stage3_p99_speedup.pdf`
- `results/stage4_final/paper_figures/fig_stage2_vs_stage3_fallback_rate.pdf`
- `results/stage4_final/paper_figures/fig_stage2_vs_stage3_max_shadow_err_bps.pdf`

## Stage 5
- `results/stage5_final/ablation_summary.csv`
- `results/stage5_final/gates_stage5.csv`
- `results/stage5_final/stage5_report.md`
- `results/stage5_final/paper_tables/table_stage5_ablation_summary.csv`
- `results/stage5_final/paper_tables/table_stage5_gate_audit.csv`
- `results/stage5_final/paper_figures/fig_stage5_speedup_vs_error.pdf`
- `results/stage5_final/paper_figures/fig_stage5_bytes_vs_fallback.pdf`

## Final artifact package
- `results/artifact_bundle_latest/`
- `results/artifact_bundle_latest/MANIFEST.csv`
- `results/artifact_bundle_latest.zip`
