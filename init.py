# === Per-parameter 1D density plots (hist + optional KDE) ===
try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

if successful_cases:
    param_names = list(successful_cases.keys())
    m = len(param_names)

    # sensible layout
    fig_h, axs_h = plt.subplots(m, 1, figsize=(9, 2.5*m), sharex=False)
    if m == 1:
        axs_h = [axs_h]

    for ax, name in zip(axs_h, param_names):
        x = np.asarray(successful_cases[name], dtype=float)
        if x.size == 0:
            ax.text(0.5, 0.5, f"No data for {name}", ha='center', va='center')
            continue

        # choose bins ~ sqrt(n), clipped to [10, 60]
        bins = max(10, min(60, int(np.sqrt(len(x)))))

        # handle degenerate constant arrays
        if np.allclose(x.min(), x.max()):
            ax.hist(x, bins=3, density=True, alpha=0.7)
            ax.set_title(f"Density (hist) — {name} (constant values)")
            ax.set_xlabel(name); ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
        else:
            # histogram as density
            ax.hist(x, bins=bins, density=True, alpha=0.35, label="Hist (density)")

            # KDE overlay if available
            if _HAS_SCIPY and len(x) >= 5:
                try:
                    kde = gaussian_kde(x)
                    x_grid = np.linspace(x.min(), x.max(), 400)
                    y = kde.evaluate(x_grid)
                    ax.plot(x_grid, y, linewidth=2, label="KDE")
                except Exception as e:
                    print(f"[KDE skipped] {name}: {e}")

            ax.set_title(f"Density for {name}")
            ax.set_xlabel(name); ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig("parameters_hist_density.png", dpi=200)
    print("Saved per-parameter hist/KDE grid to parameters_hist_density.png")
