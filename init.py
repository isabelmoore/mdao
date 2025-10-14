
    separator = "-" * 80
    for res in results[:10]:
        print(separator)
        for key, value in res.items():
            print(f"{key:30}: {value}")
            
    # --- Filter + sort successful results ---
    zero_count = sum(1 for res in results if res.get('status') == 0.0)
    print(f"\nLoading complete. {zero_count} / {len(results)} simulations failed (evaluated to 0 NM range)")

    successful_results = [res for res in results if res.get('status') == 1]
    sorted_successful = sorted(successful_results, key=lambda r: r.get('range', float('-inf')), reverse=True)

    print("Top 10 Successful Cases by Range:")
    table = PrettyTable()

    if sorted_successful:
        sample_case = sorted_successful[0]
        excluded_keys = {'status', 'case_id', 'comments'}
        field_names = [k for k in sample_case.keys() if k not in excluded_keys]
        table.field_names = field_names

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

    try:
        numeric_param_names = [spec.name for spec in input_params if isinstance(spec, NumericSpec)]
    except NameError:
        if sorted_successful:
            candidate_keys = [k for k, v in sorted_successful[0].items() if k not in {'status', 'case_id', 'comments', 'range'}]
            numeric_param_names = []
            for k in candidate_keys:
                vals = [r.get(k) for r in sorted_successful[:50]]
                nums = [x for x in vals if isinstance(x, (int, float))]
                if len(nums) >= max(3, int(0.6 * len(vals))):
                    numeric_param_names.append(k)
        else:
            numeric_param_names = []

    successful_cases = {name: [] for name in numeric_param_names}
    for r in sorted_successful:
        for name in numeric_param_names:
            v = r.get(name)
            if v is not None:
                try:
                    successful_cases[name].append(float(v))
                except (TypeError, ValueError):
                    pass

    successful_cases = {k: v for k, v in successful_cases.items() if len(v) > 0}
    if not successful_cases:
        print("No numeric parameter data available to plot. Skipping plots.")
    else:
        n_params = len(successful_cases)

        fig, axs = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=False)
        if n_params == 1:
            axs = [axs]

        # Loop over each parameter 
        for ax, (name, xvals) in zip(axs, successful_cases.items()):
            yvals = []
            x_aligned = []
            for x, r in zip(xvals, sorted_successful):
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
            
            sc = ax.scatter(x_arr, y_arr, marker='o', c=y_arr, cmap='viridis_r', label='Successful Cases')
            
            ax.set_title(f"Parameter {name}")
            ax.set_ylabel("Target Value (Range)")
            xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
            ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
            if xmax > xmin:
                ax.set_xlim(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
            if ymax > ymin:
                ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
            ax.grid(True)
            
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Target Value (Range)")
            idx_max = np.argmax(y_arr)
            max_val = y_arr[idx_max]
            max_x = x_arr[idx_max]
            
            ax.annotate(f"max: {max_x:.2f}, range: {max_val:.2f}",
                        xy=(max_x, max_val), 
                        xytext=(max_x, max_val + 0.05*(ymax-ymin)),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=9,
                        color='red')

        axs[-1].set_xlabel("Parameter Value")
        fig.suptitle("Cases for Each Parameter", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 1])        
        plt.savefig(report_dir / "success_parameters.png", dpi=200)
        print("Saved scatter grid to parameters.png")
