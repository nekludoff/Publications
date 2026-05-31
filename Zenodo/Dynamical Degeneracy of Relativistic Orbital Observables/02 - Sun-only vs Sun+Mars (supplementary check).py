import math
import numpy as np
from scipy.integrate import solve_ivp

# =========================================================
# Constants
# =========================================================

G = 6.67430e-11
M_SUN = 1.98847e30
M_MARS = 6.4171e23

MU_SUN = G * M_SUN
MU_MARS = G * M_MARS

CSTAR = 299_792_458.0
ARCSEC_PER_RAD = 206264.80624709636

# Solver settings
N_ORBITS = 80
RTOL = 1e-10
ATOL = 1e-12
MAX_STEP = 2000.0

# Mercury orbit
A_MERCURY = 5.790905e10
E_MERCURY = 0.205630

# Mars orbit (simplified prescribed Kepler orbit)
A_MARS = 2.2794382e11
E_MARS = 0.0934
MARS_PHASE_THETA0 = 0.0

# Sector test
N_SECTORS = 16
ORBIT_INDEX = 10

# =========================================================
# Channel parameters
# =========================================================

# Sun -> Mercury
ETA1_SUN = 6.0
ETA2_SUN = 0.0
ETA3_SUN = 0.0
ETA4_SUN = 0.0

# Mars -> Mercury
ETA1_MARS = 1.0
ETA2_MARS = 0.0
ETA3_MARS = 0.0
ETA4_MARS = 0.0


# =========================================================
# Kepler helpers
# =========================================================

def solve_kepler(M, e, tol=1e-13, max_iter=50):
    M = (M + 2.0 * math.pi) % (2.0 * math.pi)
    E = M if e < 0.8 else math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def kepler_state(a, e, mu, t, theta0=0.0):
    n = math.sqrt(mu / a**3)
    M = n * t
    E = solve_kepler(M, e)

    cosE = math.cos(E)
    sinE = math.sin(E)
    r = a * (1.0 - e * cosE)

    x_orb = a * (cosE - e)
    y_orb = a * math.sqrt(1.0 - e * e) * sinE

    denom = 1.0 - e * cosE
    vx_orb = -a * n * sinE / denom
    vy_orb =  a * n * math.sqrt(1.0 - e * e) * cosE / denom

    c = math.cos(theta0)
    s = math.sin(theta0)

    x = c * x_orb - s * y_orb
    y = s * x_orb + c * y_orb
    vx = c * vx_orb - s * vy_orb
    vy = s * vx_orb + c * vy_orb

    return x, y, vx, vy, r


# =========================================================
# Channel acceleration
# =========================================================

def channel_accel(dx, dy, dvx, dvy, mu_src, cstar, eta1, eta2, eta3, eta4):
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    r3 = r2 * r

    v2 = dvx * dvx + dvy * dvy
    vr = (dx * dvx + dy * dvy) / r

    channel = (
        1.0
        + eta1 * mu_src / (r * cstar * cstar)
        + eta2 * v2 / (cstar * cstar)
        + eta3 * vr * vr / (cstar * cstar)
    )

    ax = -mu_src * dx / r3 * channel
    ay = -mu_src * dy / r3 * channel

    anis = eta4 * mu_src / (r2 * cstar * cstar) * vr
    ax += anis * dvx
    ay += anis * dvy

    return ax, ay


# =========================================================
# Initial conditions
# =========================================================

def mercury_initial_conditions():
    r_p = A_MERCURY * (1.0 - E_MERCURY)
    v_p = math.sqrt(MU_SUN * (1.0 + E_MERCURY) / (A_MERCURY * (1.0 - E_MERCURY)))
    y0 = [r_p, 0.0, 0.0, v_p]
    period = 2.0 * math.pi * math.sqrt(A_MERCURY**3 / MU_SUN)
    return y0, period


# =========================================================
# Build RHS
# =========================================================

def make_rhs(include_mars: bool):
    def rhs(t, state):
        x, y, vx, vy = state

        # Sun at origin
        x_s, y_s, vx_s, vy_s = 0.0, 0.0, 0.0, 0.0

        # Sun -> Mercury
        ax_s, ay_s = channel_accel(
            x - x_s, y - y_s,
            vx - vx_s, vy - vy_s,
            MU_SUN, CSTAR,
            ETA1_SUN, ETA2_SUN, ETA3_SUN, ETA4_SUN
        )

        ax = ax_s
        ay = ay_s

        if include_mars:
            # Mars prescribed Kepler orbit
            x_m, y_m, vx_m, vy_m, _ = kepler_state(
                A_MARS, E_MARS, MU_SUN, t, theta0=MARS_PHASE_THETA0
            )

            ax_m, ay_m = channel_accel(
                x - x_m, y - y_m,
                vx - vx_m, vy - vy_m,
                MU_MARS, CSTAR,
                ETA1_MARS, ETA2_MARS, ETA3_MARS, ETA4_MARS
            )

            ax += ax_m
            ay += ay_m

        return [vx, vy, ax, ay]

    return rhs


# =========================================================
# Perihelion finder
# =========================================================

def find_perihelia(data):
    t = data["t"]
    x = data["x"]
    y = data["y"]
    theta = data["theta"]
    vr = data["vr"]

    peri_times = []
    peri_angles = []
    peri_theta_unwrapped = []

    for i in range(len(t) - 1):
        v1 = vr[i]
        v2 = vr[i + 1]

        if v1 < 0.0 and v2 > 0.0:
            t1 = t[i]
            t2 = t[i + 1]
            w = -v1 / (v2 - v1)

            t_zero = t1 + w * (t2 - t1)
            x_zero = x[i] + w * (x[i + 1] - x[i])
            y_zero = y[i] + w * (y[i + 1] - y[i])
            th_zero = theta[i] + w * (theta[i + 1] - theta[i])

            peri_times.append(t_zero)
            peri_angles.append(math.atan2(y_zero, x_zero))
            peri_theta_unwrapped.append(th_zero)

    return {
        "peri_times": np.array(peri_times),
        "peri_angles": np.unwrap(np.array(peri_angles)),
        "peri_theta_unwrapped": np.array(peri_theta_unwrapped),
    }


# =========================================================
# Sector times
# =========================================================

def compute_sector_times_exact(data, peri, orbit_index=ORBIT_INDEX, n_sectors=N_SECTORS):
    t = data["t"]
    theta = data["theta"]

    peri_t = peri["peri_times"]
    peri_th = peri["peri_theta_unwrapped"]

    if orbit_index + 1 >= len(peri_t):
        raise ValueError("orbit_index too large for detected perihelia")

    t0 = peri_t[orbit_index]
    t1 = peri_t[orbit_index + 1]
    th0 = peri_th[orbit_index]
    th1 = peri_th[orbit_index + 1]

    mask = (t >= t0) & (t <= t1)
    t_seg = t[mask]
    th_seg = theta[mask]

    if len(t_seg) == 0:
        raise RuntimeError("Empty segment after masking")

    if t_seg[0] > t0:
        th_start = np.interp(t0, t, theta)
        t_seg = np.insert(t_seg, 0, t0)
        th_seg = np.insert(th_seg, 0, th_start)
    else:
        t_seg[0] = t0
        th_seg[0] = np.interp(t0, t, theta)

    if t_seg[-1] < t1:
        th_end = np.interp(t1, t, theta)
        t_seg = np.append(t_seg, t1)
        th_seg = np.append(th_seg, th_end)
    else:
        t_seg[-1] = t1
        th_seg[-1] = np.interp(t1, t, theta)

    phi = th_seg - th0
    total_span = th1 - th0
    phi *= (2.0 * math.pi / total_span)

    sector_edges = np.linspace(0.0, 2.0 * math.pi, n_sectors + 1)
    sector_times = []

    for i in range(n_sectors):
        p0 = sector_edges[i]
        p1 = sector_edges[i + 1]
        ts0 = np.interp(p0, phi, t_seg)
        ts1 = np.interp(p1, phi, t_seg)
        sector_times.append(ts1 - ts0)

    return np.array(sector_times)


# =========================================================
# Run case
# =========================================================

def run_case(label: str, include_mars: bool):
    y0, period = mercury_initial_conditions()

    sol = solve_ivp(
        make_rhs(include_mars),
        (0.0, N_ORBITS * period),
        y0,
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )

    if not sol.success:
        raise RuntimeError(f"{label}: {sol.message}")

    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    vx = sol.y[2]
    vy = sol.y[3]

    r = np.sqrt(x * x + y * y)
    theta = np.unwrap(np.arctan2(y, x))
    vr = (x * vx + y * vy) / r

    data = {
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "r": r,
        "theta": theta,
        "vr": vr,
    }

    peri = find_perihelia(data)

    if len(peri["peri_times"]) < 3:
        raise RuntimeError(f"{label}: too few perihelia found")

    delta_varpi = np.diff(peri["peri_angles"])
    mean_rad = float(np.mean(delta_varpi))
    std_rad = float(np.std(delta_varpi))
    mean_arcsec_orbit = mean_rad * ARCSEC_PER_RAD
    orbits_per_century = 36525.0 * 86400.0 / period
    mean_arcsec_century = mean_arcsec_orbit * orbits_per_century

    sector_times = compute_sector_times_exact(data, peri)

    return {
        "label": label,
        "samples": len(t),
        "peri_count": len(peri["peri_times"]),
        "mean_rad": mean_rad,
        "std_rad": std_rad,
        "mean_arcsec_orbit": mean_arcsec_orbit,
        "mean_arcsec_century": mean_arcsec_century,
        "delta_varpi": delta_varpi,
        "sector_times": sector_times,
        "peri_times": peri["peri_times"],
        "peri_angles": peri["peri_angles"],
    }


# =========================================================
# Main
# =========================================================

def main():
    print("=== Compare Sun-only vs Sun+Mars ===")
    print()
    print("Sun channel params:")
    print(f"  eta1={ETA1_SUN}, eta2={ETA2_SUN}, eta3={ETA3_SUN}, eta4={ETA4_SUN}")
    print("Mars channel params:")
    print(f"  eta1={ETA1_MARS}, eta2={ETA2_MARS}, eta3={ETA3_MARS}, eta4={ETA4_MARS}")
    print()

    sun_only = run_case("Sun-only", include_mars=False)
    sun_mars = run_case("Sun+Mars", include_mars=True)

    for res in [sun_only, sun_mars]:
        print(f"--- {res['label']} ---")
        print(f"Samples: {res['samples']}")
        print(f"Perihelia found: {res['peri_count']}")
        print("Perihelion precession:")
        print(f"  mean = {res['mean_rad']:.12e} rad/orbit")
        print(f"  std  = {res['std_rad']:.12e} rad/orbit")
        print(f"  mean = {res['mean_arcsec_orbit']:.6f} arcsec/orbit")
        print(f"  mean = {res['mean_arcsec_century']:.6f} arcsec/century")
        print()

    print("=== Differences: Sun+Mars minus Sun-only ===")
    d_mean_century = sun_mars["mean_arcsec_century"] - sun_only["mean_arcsec_century"]
    d_mean_orbit = sun_mars["mean_arcsec_orbit"] - sun_only["mean_arcsec_orbit"]
    d_std_rad = sun_mars["std_rad"] - sun_only["std_rad"]

    print(f"Δ mean precession = {d_mean_orbit:.6f} arcsec/orbit")
    print(f"Δ mean precession = {d_mean_century:.6f} arcsec/century")
    print(f"Δ std(rad/orbit)  = {d_std_rad:.12e}")
    print()

    print("=== First 20 delta_varpi values (arcsec/orbit) ===")
    nshow = min(20, len(sun_only["delta_varpi"]), len(sun_mars["delta_varpi"]))
    print(f"{'orbit':>5} {'Sun-only':>16} {'Sun+Mars':>16} {'difference':>16}")
    print("-" * 60)
    for i in range(nshow):
        s1 = sun_only["delta_varpi"][i] * ARCSEC_PER_RAD
        s2 = sun_mars["delta_varpi"][i] * ARCSEC_PER_RAD
        print(f"{i:5d} {s1:16.9f} {s2:16.9f} {s2 - s1:16.9f}")
    print()

    print(f"=== Sector times on orbit index {ORBIT_INDEX} ===")
    print(f"{'sector':>6} {'Sun-only [s]':>16} {'Sun+Mars [s]':>16} {'difference [s]':>16}")
    print("-" * 62)
    sec_diff = sun_mars["sector_times"] - sun_only["sector_times"]
    for i in range(N_SECTORS):
        print(
            f"{i:6d} "
            f"{sun_only['sector_times'][i]:16.6f} "
            f"{sun_mars['sector_times'][i]:16.6f} "
            f"{sec_diff[i]:16.6f}"
        )
    print()
    print(f"max |Δsector_time| = {np.max(np.abs(sec_diff)):.6f} s")
    print(f"rms  Δsector_time  = {np.sqrt(np.mean(sec_diff**2)):.6f} s")


if __name__ == "__main__":
    main()