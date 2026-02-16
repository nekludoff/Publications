import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def simulate_node_stream(K_max: int) -> np.ndarray:
    """Ideal node-level canonical stream indices: 1..K_max."""
    return np.arange(1, K_max + 1, dtype=int)

def channel(arr_indices: np.ndarray,
            p_loss: float,
            p_dup: float,
            delay_mean: float,
            delay_std: float,
            reorder: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Channel effects:
      - loss: drop each index independently with prob p_loss
      - duplication: duplicate kept indices with prob p_dup (one extra copy)
      - delay: gaussian delay truncated at 0
      - reorder: arrivals sorted by arrival time (out-of-order indices)
    Returns:
      idx_arr: indices as received (possibly with duplicates)
      t_arr: arrival times for each received element
    """
    keep = rng.random(arr_indices.size) > p_loss
    idx = arr_indices[keep]

    dup_mask = rng.random(idx.size) < p_dup
    if dup_mask.any():
        idx = np.concatenate([idx, idx[dup_mask]])

    delays = rng.normal(delay_mean, delay_std, size=idx.size)
    delays = np.clip(delays, 0.0, None)

    # emission time ~ index (only to create a flow; reconstruction ignores it)
    emit_time = idx.astype(float)
    t_arr = emit_time + delays

    if reorder:
        order = np.argsort(t_arr)
        idx = idx[order]
        t_arr = t_arr[order]

    return idx, t_arr

def reconstruct_by_index(idx_arr: np.ndarray) -> np.ndarray:
    """Rec: remove duplicates, sort by index."""
    if idx_arr.size == 0:
        return np.array([], dtype=int)
    return np.unique(idx_arr)  # sorted unique

def merge_streams(streams: list[np.ndarray]) -> np.ndarray:
    """Merge: union of indices across nodes, sorted."""
    nonempty = [s for s in streams if s.size > 0]
    if not nonempty:
        return np.array([], dtype=int)
    all_idx = np.concatenate(nonempty, axis=0)
    return np.unique(all_idx)

def coverage(reconstructed: np.ndarray, K_max: int) -> float:
    """Fraction of indices recovered in [1..K_max]."""
    return reconstructed.size / float(K_max)

def prefix_growth_over_time(arrivals: list[tuple[np.ndarray, np.ndarray]],
                            K_max: int,
                            t_grid: np.ndarray) -> np.ndarray:
    """
    For each time T, take arrivals up to T, Rec per node, then Merge.
    """
    cov = np.zeros_like(t_grid, dtype=float)
    for i, T in enumerate(t_grid):
        rec_nodes = []
        for idx_arr, t_arr in arrivals:
            mask = t_arr <= T
            rec_nodes.append(reconstruct_by_index(idx_arr[mask]))
        merged = merge_streams(rec_nodes)
        cov[i] = coverage(merged, K_max)
    return cov

# -----------------------------
# Experiment 1: Coverage vs loss (Monte Carlo)
# -----------------------------
def experiment_coverage_vs_loss(K_max=500, M_nodes=4, trials=200,
                                p_dup=0.02, delay_mean=3.0, delay_std=1.5):
    loss_grid = np.linspace(0.0, 0.6, 13)
    means, stds = [], []
    base = simulate_node_stream(K_max)

    for p_loss in loss_grid:
        covs = []
        for _ in range(trials):
            rec_nodes = []
            for _j in range(M_nodes):
                idx_arr, _t_arr = channel(base, p_loss=p_loss, p_dup=p_dup,
                                          delay_mean=delay_mean, delay_std=delay_std,
                                          reorder=True)
                rec_nodes.append(reconstruct_by_index(idx_arr))
            merged = merge_streams(rec_nodes)
            covs.append(coverage(merged, K_max))
        covs = np.array(covs)
        means.append(covs.mean())
        stds.append(covs.std(ddof=1))
    return loss_grid, np.array(means), np.array(stds)

loss_grid, cov_mean, cov_std = experiment_coverage_vs_loss()

plt.figure()
plt.errorbar(loss_grid, cov_mean, yerr=cov_std, fmt='o-')
plt.xlabel('Loss probability p_loss')
plt.ylabel('Recovered coverage (fraction of indices)')
plt.title('Reconstruction coverage vs loss (Monte Carlo)')
plt.ylim(0, 1.02)
plt.show()

# -----------------------------
# Experiment 2: Prefix growth vs latency (single run)
# -----------------------------
def experiment_prefix_growth(K_max=500, M_nodes=4, p_loss=0.1, p_dup=0.02,
                             delay_mean=8.0, delay_std=4.0,
                             T_max=560.0, n_points=140):
    base = simulate_node_stream(K_max)
    arrivals = []
    for _j in range(M_nodes):
        idx_arr, t_arr = channel(base, p_loss=p_loss, p_dup=p_dup,
                                 delay_mean=delay_mean, delay_std=delay_std,
                                 reorder=True)
        arrivals.append((idx_arr, t_arr))

    t_grid = np.linspace(0.0, T_max, n_points)
    cov_t = prefix_growth_over_time(arrivals, K_max, t_grid)
    return t_grid, cov_t

t_grid, cov_t = experiment_prefix_growth()

plt.figure()
plt.plot(t_grid, cov_t)
plt.xlabel('Observer time (arbitrary units)')
plt.ylabel('Recovered coverage (fraction of indices)')
plt.title('Prefix growth under latency/reordering (single run)')
plt.ylim(0, 1.02)
plt.show()


# -----------------------------
# Helper metric: time-to-threshold
# -----------------------------
def time_to_coverage(t_grid: np.ndarray, cov_t: np.ndarray, target: float) -> float:
    idx = np.searchsorted(cov_t, target, side='left')
    if idx >= len(t_grid):
        return float('nan')
    return float(t_grid[idx])

print("Example: time-to-90% coverage =", time_to_coverage(t_grid, cov_t, 0.9))

# -----------------------------
# Experiment 3A: Scaling vs number of nodes (Monte Carlo)
#   Metric: time-to-90% coverage
# -----------------------------
def experiment_time_to_target_vs_nodes(
    K_max=800,
    p_loss=0.1,
    p_dup=0.02,
    delay_mean=10.0,
    delay_std=5.0,
    target=0.95,
    T_max=1600.0,
    n_points=220,
    trials=120,
    nodes_grid=(1, 2, 3, 4, 6, 8, 10, 12)
):
    base = simulate_node_stream(K_max)
    t_grid = np.linspace(0.0, T_max, n_points)

    means, stds = [], []
    for M_nodes in nodes_grid:
        tts = []
        for _ in range(trials):
            arrivals = []
            for _j in range(M_nodes):
                idx_arr, t_arr = channel(
                    base,
                    p_loss=p_loss,
                    p_dup=p_dup,
                    delay_mean=delay_mean,
                    delay_std=delay_std,
                    reorder=True
                )
                arrivals.append((idx_arr, t_arr))

            cov_t = prefix_growth_over_time(arrivals, K_max, t_grid)
            tt = time_to_coverage(t_grid, cov_t, target)
            tts.append(tt)

        tts = np.array(tts, dtype=float)
        means.append(np.nanmean(tts))
        stds.append(np.nanstd(tts, ddof=1))
    return np.array(nodes_grid, dtype=int), np.array(means), np.array(stds)

nodes_grid, tt_mean, tt_std = experiment_time_to_target_vs_nodes()

plt.figure()
plt.errorbar(nodes_grid, tt_mean, yerr=tt_std, fmt='o-')
plt.xlabel('Number of reception nodes (M)')
plt.ylabel('Time to reach 95% coverage')
plt.title('Scaling: time-to-95% coverage vs number of nodes (Monte Carlo)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


# -----------------------------
# Experiment 3B: Scaling vs mean latency (Monte Carlo)
#   Metric: time-to-90% coverage
# -----------------------------
def experiment_time_to_target_vs_latency(
    K_max=800,
    M_nodes=10,
    p_loss=0.1,
    p_dup=0.02,
    delay_std=5.0,
    target=0.95,
    T_max=1300.0,
    n_points=260,
    trials=120,
    latency_grid=(0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 28.0)
):
    base = simulate_node_stream(K_max)
    t_grid = np.linspace(0.0, T_max, n_points)

    means, stds = [], []
    for delay_mean in latency_grid:
        tts = []
        for _ in range(trials):
            arrivals = []
            for _j in range(M_nodes):
                idx_arr, t_arr = channel(
                    base,
                    p_loss=p_loss,
                    p_dup=p_dup,
                    delay_mean=delay_mean,
                    delay_std=delay_std,
                    reorder=True
                )
                arrivals.append((idx_arr, t_arr))

            cov_t = prefix_growth_over_time(arrivals, K_max, t_grid)
            tt = time_to_coverage(t_grid, cov_t, target)
            tts.append(tt)

        tts = np.array(tts, dtype=float)
        means.append(np.nanmean(tts))
        stds.append(np.nanstd(tts, ddof=1))
    return np.array(latency_grid, dtype=float), np.array(means), np.array(stds)

lat_grid, ttL_mean, ttL_std = experiment_time_to_target_vs_latency()

plt.figure()
plt.errorbar(lat_grid, ttL_mean, yerr=ttL_std, fmt='o-')
plt.xlabel('Mean channel delay (delay_mean)')
plt.ylabel('Time to reach 95% coverage')
plt.title('Scaling: time-to-95% coverage vs mean latency (Monte Carlo)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
