# --- Filter + sort successful results ---
successful_results = [res for res in results if res.get('status') == 1]
sorted_successful = sorted(successful_results, key=lambda r: r.get('range', float('-inf')), reverse=True)

print("Top 10 Successful Cases by Range:")
table = PrettyTable()

if sorted_successful:
    # Build table headers from the first row, excluding some keys
    sample_case = sorted_successful[0]
    excluded_keys = {'status', 'case_id', 'comments'}
    field_names = [k for k in sample_case.keys() if k not in excluded_keys]
    table.field_names = field_names

    # Add up to 10 rows, formatting floats
    for result in sorted_successful[:10]:
        row = []
        for key in field_names:
            val = result.get(key)
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(val)
        table.add_row(row)

    print(table)
else:
    print("No successful results found (status == 1). Skipping table and plots.")
    # Early exit if you prefer; or just skip plotting below
    # raise SystemExit

# --- Build plotting data ONLY from numeric specs you care about ---
# Assumes you have a list `input_params` with objects having `.name` for numeric specs.
# If you don't, you can hardcode a list like: numeric_param_names = ["param_a", "param_b", ...]
try:
    numeric_param_names = [spec.name for spec in input_params if isinstance(spec, NumericSpec)]
except NameError:
    # Fallback: attempt to infer numeric columns heuristically (float/int) from the first success
    if sorted_successful:
        candidate_keys = [k for k, v in sorted_successful[0].items() if k not in {'status', 'case_id', 'comments', 'range'}]
        # Keep keys whose values are numeric in most rows
        numeric_param_names = []
        for k in candidate_keys:
            vals = [r.get(k) for r in sorted_successful[:50]]
            nums = [x for x in vals if isinstance(x, (int, float))]
            if len(nums) >= max(3, int(0.6 * len(vals))):
                numeric_param_names.append(k)
    else:
        numeric_param_names = []

# Prepare X (param values) and Y (target) aligned with sorted_successful order
successful_targets = np.array([float(r['range']) for r in sorted_successful], dtype=float)

# Collect values for each numeric parameter
successful_cases = {name: [] for name in numeric_param_names}
for r in sorted_successful:
    for name in numeric_param_names:
        v = r.get(name)
        if v is not None:
            try:
                successful_cases[name].append(float(v))
            except (TypeError, ValueError):
                # Skip non-numeric entries
                pass

# Remove parameters that ended up with no data
successful_cases = {k: v for k, v in successful_cases.items() if len(v) > 0}

if not successful_cases:
    print("No numeric parameter data available to plot. Skipping plots.")
else:
    n_params = len(successful_cases)

    # Create one subplot per parameter; handle the single-axes case
    fig, axs = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=False)
    if n_params == 1:
        axs = [axs]

    # Note: targets list must match x-values length per param (filter to common indices)
    # Here we’ll assume every success has a value for each param we kept above.
    # If that’s not true, we’ll filter pairs per param below.
    for ax, (name, xvals) in zip(axs, successful_cases.items()):
        # Build aligned pairs (x_i, y_i) only where x_i is present
        yvals = []
        x_aligned = []
        for x, r in zip(xvals, sorted_successful):
            # Each x came from one row in order; ensure we have a range value
            try:
                y = float(r['range'])
                x_aligned.append(float(x))
                yvals.append(y)
            except (KeyError, TypeError, ValueError):
                continue

        if len(x_aligned) == 0:
            print(f"No aligned data to plot for parameter '{name}'.")
            continue

        x_arr = np.array(x_aligned, dtype=float)
        y_arr = np.array(yvals, dtype=float)

        ax.scatter(x_arr, y_arr, marker='o', alpha=0.7, label='Successful Cases')
        ax.set_title(f"Successful Cases for {name}")
        ax.set_ylabel("Target Value (Range)")

        # Smart axis limits
        xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
        ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
        if xmax > xmin:
            ax.set_xlim(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
        if ymax > ymin:
            ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Parameter Value")
    plt.tight_layout()
    plt.savefig("parameters.png", dpi=200)
    print("Saved scatter grid to parameters.png")
