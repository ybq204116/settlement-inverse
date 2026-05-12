import argparse
import os
from pathlib import Path

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import matplotlib.pyplot as plt

try:
    import deepxde as dde
except ImportError as exc:
    raise SystemExit(
        "DeepXDE is not installed in this Python environment. "
        "Install DeepXDE with a PyTorch backend before running inverse.py."
    ) from exc


# Paper example constants.
R_TUNNEL = 1.5
H = 7.5
E_TRUE = 10.0
NU_TRUE = 0.2
UR = 1.0
UE = 2.0

X_MIN, X_MAX = -15.0, 15.0
Z_MIN, Z_MAX = 0.0, 15.0

DEFAULT_SEED = 2026

# These variables are initialized in build_and_train so command line initial
# guesses can be used while PDE/operator callbacks still see the same objects.
E_VAR = None
NU_VAR = None


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def vb_displacement(x, z, r=R_TUNNEL, h=H, nu=NU_TRUE, ur=UR, ue=UE):
    """Verruijt-Booker displacement solution in an elastic half-plane."""
    x = np.asarray(x)
    z = np.asarray(z)

    k = nu / (1.0 + nu)
    m = 1.0 / (1.0 - 2.0 * nu)

    z1 = z - h
    z2 = z + h
    r1_sq = x**2 + z1**2
    r2_sq = x**2 + z2**2
    r1_4 = r1_sq**2
    r2_4 = r2_sq**2
    r2_6 = r2_sq**3

    ux_12 = (
        -ur * r**3 * x * (1.0 / r1_sq + 1.0 / r2_sq)
        + ue * r**3 * x * ((x**2 - k * z1**2) / r1_4 + (x**2 - k * z2**2) / r2_4)
    )
    uz_12 = (
        -ur * r**3 * (z1 / r1_sq + z2 / r2_sq)
        + ue
        * r**3
        * (z1 * (k * x**2 - z1**2) / r1_4 + z2 * (k * x**2 - z2**2) / r2_4)
    )

    ux_34 = (
        -2.0 * ur * r**3 * x / m * (1.0 / r2_sq - 2.0 * m * z * z2 / r2_4)
        - 4.0
        * ue
        * r**3
        * x
        * h
        / (1.0 + m)
        * (z2 / r2_4 + m * z * (x**2 - 3.0 * z2**2) / r2_6)
    )
    uz_34 = (
        2.0
        * ur
        * r**3
        / m
        * ((m + 1.0) * z2 / r2_sq - m * z * (x**2 - z2**2) / r2_4)
        - 2.0
        * ue
        * r**3
        * h
        * (
            (x**2 - z2**2) / r2_4
            + (m / (m + 1.0)) * (2.0 * z * z2 * (3.0 * x**2 - z2**2) / r2_6)
        )
    )

    return ux_12 + ux_34, uz_12 + uz_34


def variable_to_float(value):
    """Return a Python float from a DeepXDE backend variable/tensor."""
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    try:
        return float(dde.backend.to_numpy(value).item())
    except AttributeError:
        return float(value)


def current_material_values():
    return variable_to_float(E_VAR), variable_to_float(NU_VAR)


def lame_parameters():
    e = E_VAR
    nu = NU_VAR
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    return lam, mu


def strain_components(x, u):
    ux_x = dde.grad.jacobian(u, x, i=0, j=0)
    ux_z = dde.grad.jacobian(u, x, i=0, j=1)
    uz_x = dde.grad.jacobian(u, x, i=1, j=0)
    uz_z = dde.grad.jacobian(u, x, i=1, j=1)

    eps_xx = ux_x
    eps_zz = uz_z
    eps_xz = 0.5 * (ux_z + uz_x)
    return eps_xx, eps_zz, eps_xz


def stress_components(x, u):
    eps_xx, eps_zz, eps_xz = strain_components(x, u)
    lam, mu = lame_parameters()
    tr_eps = eps_xx + eps_zz

    sigma_xx = lam * tr_eps + 2.0 * mu * eps_xx
    sigma_zz = lam * tr_eps + 2.0 * mu * eps_zz
    sigma_xz = 2.0 * mu * eps_xz
    return sigma_xx, sigma_zz, sigma_xz


def pde(x, u):
    sigma_xx, sigma_zz, sigma_xz = stress_components(x, u)

    dsxx_dx = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
    dsxz_dz = dde.grad.jacobian(sigma_xz, x, i=0, j=1)
    dsxz_dx = dde.grad.jacobian(sigma_xz, x, i=0, j=0)
    dszz_dz = dde.grad.jacobian(sigma_zz, x, i=0, j=1)

    balance_x = dsxx_dx + dsxz_dz
    balance_z = dsxz_dx + dszz_dz
    return [balance_x, balance_z]


def top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], Z_MIN)


def operator_sigma_zz(x, u, _):
    _, sigma_zz, _ = stress_components(x, u)
    return sigma_zz


def operator_sigma_xz(x, u, _):
    _, _, sigma_xz = stress_components(x, u)
    return sigma_xz


def inside_domain(points):
    x = points[:, 0]
    z = points[:, 1]
    in_rect = (X_MIN <= x) & (x <= X_MAX) & (Z_MIN <= z) & (z <= Z_MAX)
    outside_tunnel = x**2 + (z - H) ** 2 >= R_TUNNEL**2
    return in_rect & outside_tunnel


def sample_inverse_points(n, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    chunks = []
    total = 0
    while total < n:
        trial_count = max(2 * (n - total), 256)
        trial = np.column_stack(
            [
                rng.uniform(X_MIN, X_MAX, trial_count),
                rng.uniform(Z_MIN, Z_MAX, trial_count),
            ]
        )
        valid = trial[inside_domain(trial)]
        chunks.append(valid)
        total += len(valid)

    points = np.vstack(chunks)[:n]
    ux, uz = vb_displacement(points[:, 0], points[:, 1])
    return points, ux.reshape(-1, 1), uz.reshape(-1, 1)


def make_geometry():
    rect = dde.geometry.Rectangle([X_MIN, Z_MIN], [X_MAX, Z_MAX])
    tunnel = dde.geometry.Disk([0.0, H], R_TUNNEL)
    return dde.geometry.CSGDifference(rect, tunnel)


def set_lbfgs_options(maxiter):
    try:
        dde.optimizers.config.set_LBFGS_options(maxiter=maxiter, maxls=50, gtol=1e-8)
    except AttributeError:
        dde.optimizers.set_LBFGS_options(maxiter=maxiter, maxls=50, gtol=1e-8)


def build_and_train(args):
    global E_VAR, NU_VAR
    E_VAR = dde.Variable(args.e_init)
    NU_VAR = dde.Variable(args.nu_init)

    geom = make_geometry()
    train_points, ux_data, uz_data = sample_inverse_points(args.num_train, args.seed)

    bc_data_ux = dde.PointSetBC(train_points, ux_data, component=0)
    bc_data_uz = dde.PointSetBC(train_points, uz_data, component=1)
    bc_top_szz = dde.icbc.OperatorBC(geom, operator_sigma_zz, top_boundary)
    bc_top_sxz = dde.icbc.OperatorBC(geom, operator_sigma_xz, top_boundary)

    data = dde.data.PDE(
        geom,
        pde,
        [bc_data_ux, bc_data_uz, bc_top_szz, bc_top_sxz],
        num_domain=0,
        num_boundary=args.num_boundary,
        anchors=train_points,
        train_distribution="LHS",
        num_test=args.num_test,
    )

    net = dde.nn.FNN([2] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    loss_weights = [1, 1, 50, 50, 10, 10]
    external_vars = [E_VAR, NU_VAR]

    result_dir = project_root() / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    variable_log = result_dir / "inverse_variables.dat"
    variable_cb = dde.callbacks.VariableValue(
        external_vars, period=args.display_every, filename=str(variable_log)
    )

    model.compile(
        "adam",
        lr=args.lr,
        loss_weights=loss_weights,
        external_trainable_variables=external_vars,
    )
    losshistory, train_state = model.train(
        iterations=args.adam_iterations,
        display_every=args.display_every,
        callbacks=[variable_cb],
    )

    last_train_loss = np.asarray(losshistory.loss_train[-1], dtype=float)
    if np.isnan(last_train_loss).any():
        raise RuntimeError("Adam produced NaN loss; stop before L-BFGS.")

    e_after_adam, nu_after_adam = current_material_values()
    print_parameter_report("After Adam", e_after_adam, nu_after_adam)

    if not args.skip_lbfgs:
        set_lbfgs_options(args.lbfgs_maxiter)
        model.compile(
            "L-BFGS",
            loss_weights=loss_weights,
            external_trainable_variables=external_vars,
        )
        model.train(display_every=args.display_every, callbacks=[variable_cb])

    return model


def evaluate_model(model):
    nx, nz = 180, 110
    xv = np.linspace(X_MIN, X_MAX, nx)
    zv = np.linspace(Z_MIN, Z_MAX, nz)
    xx, zz = np.meshgrid(xv, zv)
    points = np.column_stack([xx.ravel(), zz.ravel()])
    mask = inside_domain(points)
    points_out = points[mask]

    pred = model.predict(points_out)
    uz_pred = pred[:, 1]
    uz_true = vb_displacement(points_out[:, 0], points_out[:, 1])[1]
    rel_l2 = np.linalg.norm(uz_pred - uz_true) / (np.linalg.norm(uz_true) + 1e-12)

    x_line = np.linspace(X_MIN, X_MAX, 500)
    z_line = np.zeros_like(x_line)
    line_points = np.column_stack([x_line, z_line])
    uz_line_pred = model.predict(line_points)[:, 1]
    uz_line_true = vb_displacement(x_line, z_line)[1]
    rmse = np.sqrt(np.mean((uz_line_pred - uz_line_true) ** 2))
    peak_err = abs(np.max(uz_line_pred) - np.max(uz_line_true)) / (
        abs(np.max(uz_line_true)) + 1e-12
    )

    metrics = {
        "rel_l2_uz": rel_l2,
        "surface_rmse": rmse,
        "surface_peak_rel_err": peak_err,
    }
    return metrics


def plot_inverse_results(model, output_path):
    nx, nz = 240, 150
    xv = np.linspace(X_MIN, X_MAX, nx)
    zv = np.linspace(Z_MIN, Z_MAX, nz)
    xx, zz = np.meshgrid(xv, zv)
    points = np.column_stack([xx.ravel(), zz.ravel()])
    mask = inside_domain(points)
    points_out = points[mask]

    uz_pred = model.predict(points_out)[:, 1]
    uz_true = vb_displacement(points_out[:, 0], points_out[:, 1])[1]

    field_pred = np.full(points.shape[0], np.nan)
    field_true = np.full(points.shape[0], np.nan)
    field_pred[mask] = uz_pred
    field_true[mask] = uz_true
    field_pred = field_pred.reshape(nz, nx)
    field_true = field_true.reshape(nz, nx)

    x_line = np.linspace(X_MIN, X_MAX, 500)
    z_line = np.zeros_like(x_line)
    uz_line_true = vb_displacement(x_line, z_line)[1]
    uz_line_pred = model.predict(np.column_stack([x_line, z_line]))[:, 1]

    vmin = np.nanmin([field_pred, field_true])
    vmax = np.nanmax([field_pred, field_true])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    c0 = axes[0].contourf(
        xx, zz, field_pred, levels=40, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("(a) PINN inversion of $u_z$")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_aspect("equal")
    fig.colorbar(c0, ax=axes[0], shrink=0.88)

    axes[1].plot(x_line, uz_line_true, "k-", lw=2.2, label="Exact")
    axes[1].plot(x_line, uz_line_pred, "r--", lw=2.0, label="PINN inverse")
    axes[1].set_title("(b) Surface settlement at $z=0$")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("$u_z$")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


def print_parameter_report(title, e_value, nu_value):
    e_err = abs(e_value - E_TRUE) / abs(E_TRUE) * 100.0
    nu_err = abs(nu_value - NU_TRUE) / abs(NU_TRUE) * 100.0
    print(f"\n[{title}]")
    print("parameter    inverse      exact       rel_error(%)")
    print(f"E            {e_value:10.6f}  {E_TRUE:10.6f}  {e_err:12.6f}")
    print(f"nu           {nu_value:10.6f}  {NU_TRUE:10.6f}  {nu_err:12.6f}")


def print_identifiability_warning(e_value):
    e_err = abs(e_value - E_TRUE) / abs(E_TRUE)
    if e_err > 0.05:
        print(
            "\n[diagnostic] E is more than 5% away from the exact value. "
            "For this displacement-only inverse setup, Young's modulus can be weakly "
            "identifiable because the Verruijt-Booker displacement field is mainly "
            "controlled by nu, ur, and ue while E scales the homogeneous stress residual."
        )


def project_root():
    return Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PINN inverse example for Verruijt-Booker tunnel settlement."
    )
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-boundary", type=int, default=400)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--adam-iterations", type=int, default=10000)
    parser.add_argument("--lbfgs-maxiter", type=int, default=25000)
    parser.add_argument("--skip-lbfgs", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--display-every", type=int, default=500)
    parser.add_argument("--e-init", type=float, default=8.0)
    parser.add_argument("--nu-init", type=float, default=0.25)
    args = parser.parse_args()

    if args.quick:
        args.num_train = min(args.num_train, 80)
        args.num_boundary = min(args.num_boundary, 80)
        args.num_test = min(args.num_test, 200)
        args.adam_iterations = min(args.adam_iterations, 50)
        args.lbfgs_maxiter = min(args.lbfgs_maxiter, 50)
        args.skip_lbfgs = True
        args.display_every = min(args.display_every, 25)

    return args


def main():
    args = parse_args()
    dde.config.set_random_seed(args.seed)

    model = build_and_train(args)

    e_value, nu_value = current_material_values()
    print_parameter_report("Final inverse result", e_value, nu_value)
    print_identifiability_warning(e_value)

    metrics = evaluate_model(model)
    print("\n[metrics]")
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")

    output_path = project_root() / "result" / "inverse_fig9.png"
    plot_inverse_results(model, output_path)
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
