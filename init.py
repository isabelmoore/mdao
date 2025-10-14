# --- knobs (kept local so you can tweak quickly) ---
RANGE_KEY = "range"   # or "range_nm" if you renamed it
TOP_K     = 10

# --- compact peek at first 10 results ---
print("\nSample results (first 10):")
for i, res in enumerate(results[:10], 1):
    print("-" * 80)
    for k in sorted(res.keys()):
        v = res[k]
        print(f"{k:30}: {v}")

# --- basic counters + success filter ---
total = len(results)
failed = sum(1 for r in results if int(r.get("status", 0)) == 0)
print(f"\nLoading complete. {failed} / {total} simulations failed (status == 0)")

successful = [r for r in results if int(r.get("status", 0)) == 1 and r.get(RANGE_KEY) is not None]
success_sorted = sorted(successful, key=lambda r: float(r.get(RANGE_KEY, float("-inf"))), reverse=True)

# --- top-K table (stable column order: case_id, params…, range, status) ---
print("\nTop Successful Cases by Range:")
if success_sorted:
    param_names = [spec.name for spec in input_params]  # include numeric + categorical if any
    columns = ["case_id", *param_names, RANGE_KEY, "status"]

    table = PrettyTable()
    table.field_names = columns

    for r in success_sorted[:TOP_K]:
        row = []
        for c in columns:
            val = r.get(c)
            row.append(f"{val:.4f}" if isinstance(val, float) else val)
        table.add_row(row)
    print(table)
else:
    print("No successful results found (status == 1). Skipping table and plots.")

# --- plotting (only if we have successes) ---
if success_sorted:
    # numeric params only for scatter vs RANGE_KEY
    numeric_param_names = [spec.name for spec in input_params if isinstance(spec, NumericSpec)]
    if numeric_param_names:
        y_arr = np.asarray([float(r[RANGE_KEY]) for r in success_sorted], dtype=float)

        n = len(numeric_param_names)
        fig, axs = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)
        if n == 1:
            axs = [axs]

        for ax, pname in zip(axs, numeric_param_names):
            x_arr = np.asarray([float(r[pname]) for r in success_sorted], dtype=float)

            sc = ax.scatter(x_arr, y_arr, c=y_arr, cmap="viridis_r", s=18)
            ax.set_title(f"{pname} vs {RANGE_KEY}")
            ax.set_xlabel(pname)
            ax.set_ylabel(RANGE_KEY)
            ax.grid(True, alpha=0.3)

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(RANGE_KEY)

            # annotate max
            idx = int(np.argmax(y_arr))
            ax.annotate(
                f"max: {x_arr[idx]:.2f}, {y_arr[idx]:.2f}",
                xy=(x_arr[idx], y_arr[idx]),
                xytext=(x_arr[idx], y_arr[idx] * 1.01),
                arrowprops=dict(arrowstyle="->"),
            )

            # gentle padding if ranges are non-degenerate
            if np.ptp(x_arr) > 0:
                xr = np.ptp(x_arr)
                ax.set_xlim(x_arr.min() - 0.1 * xr, x_arr.max() + 0.1 * xr)
            if np.ptp(y_arr) > 0:
                yr = np.ptp(y_arr)
                ax.set_ylim(y_arr.min() - 0.1 * yr, y_arr.max() + 0.1 * yr)

        fig.suptitle("Successful Cases per Parameter", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(report_dir / "success_parameters.png", dpi=200)
        plt.close(fig)
        print("Saved scatter grid to success_parameters.png")
    else:
        print("No numeric parameter data available to plot. Skipping plots.")
