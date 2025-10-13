# ----- 2D projection over rotating parameter pairs (works for 4+ params) -----
if len(keys) >= 2 and X_train.size > 0:
    all_pairs = list(combinations(keys, 2))  # e.g., 6 pairs for 4 params
    pairs_per_iter = 2                        # render at most 2 pairs each iter
    start = (it * pairs_per_iter) % len(all_pairs)
    sel_pairs = [all_pairs[(start + j) % len(all_pairs)] for j in range(pairs_per_iter)]

    # Fix other params at current best (or midpoints if none yet)
    if optimizer.max and 'params' in optimizer.max:
        fixed_defaults = {k: float(optimizer.max['params'][k]) for k in keys}
    else:
        fixed_defaults = {k: float((param_bounds[k][0] + param_bounds[k][1]) / 2.0) for k in keys}

    for (p1, p2) in sel_pairs:
        p1_lin = np.linspace(param_bounds[p1][0], param_bounds[p1][1], grid_points)
        p2_lin = np.linspace(param_bounds[p2][0], param_bounds[p2][1], grid_points)
        X1, X2 = np.meshgrid(p1_lin, p2_lin)

        # Build query grid with other params fixed
        X_query = []
        for a, b in zip(X1.ravel(), X2.ravel()):
            row = []
            for k in keys:
                if k == p1:
                    row.append(a)
                elif k == p2:
                    row.append(b)
                else:
                    row.append(fixed_defaults[k])
            X_query.append(row)
        X_query = np.array(X_query, dtype=float)

        Z_mu, Z_std = gp.predict(X_query, return_std=True)
        y_max = float(np.max(y_train)) if y_train.size else 0.0
        Z_acq = acq.utility(X_query, gp, y_max)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axs[0].contourf(X1, X2, Z_mu.reshape(X1.shape), levels=50)
        axs[0].set_title(f'GP Predicted Mean ({p1} vs {p2})')
        axs[0].set_xlabel(p1); axs[0].set_ylabel(p2)
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].contourf(X1, X2, Z_std.reshape(X1.shape), levels=50)
        axs[1].set_title('GP Predictive Std')
        axs[1].set_xlabel(p1); axs[1].set_ylabel(p2)
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].contourf(X1, X2, Z_acq.reshape(X1.shape), levels=50)
        axs[2].set_title(f'Acquisition ({acq.kind.upper()})')
        axs[2].set_xlabel(p1); axs[2].set_ylabel(p2)
        fig.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        frame_filename = f"surface_{p1}_{p2}_iter_{it+1}.png"
        plt.savefig(frame_filename, dpi=120)
        plt.close(fig)
        frames.append(frame_filename)
