import os

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# Matplotlib 中文显示配置（Windows 常用字体回退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 算例参数（论文算例1）
# -----------------------------
R_TUNNEL = 1.5  # 隧道半径 r (m)
E = 10.0  # 弹性模量 E (MPa)
NU = 0.2  # 泊松比 nu
H = 7.5  # 隧道中心埋深 h (m)

X_MIN, X_MAX = -15.0, 15.0
Y_MIN, Y_MAX = 0.0, 15.0  # y=0 为自由地表，y向下为正

UR = 1.0  # 围岩径向位移因子
UE = 2.0  # 围岩椭圆化位移因子

LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


def vb_displacement(x, y, r=R_TUNNEL, h=H, nu=NU, ur=UR, ue=UE):
    """Verruijt-Booker 位移解析解（全域对照“准确值”）"""
    x = np.asarray(x)
    y = np.asarray(y)

    k = nu / (1.0 + nu)
    m = 1.0 / (1.0 - 2.0 * nu)

    y1 = y - h
    y2 = y + h
    r1_sq = x**2 + y1**2
    r2_sq = x**2 + y2**2
    r1_4 = r1_sq**2
    r2_4 = r2_sq**2
    r2_6 = r2_sq**3

    ux_12 = (
        -ur * r**3 * x * (1.0 / r1_sq + 1.0 / r2_sq)
        + ue * r**3 * x * ((x**2 - k * y1**2) / r1_4 + (x**2 - k * y2**2) / r2_4)
    )
    uy_12 = (
        -ur * r**3 * (y1 / r1_sq + y2 / r2_sq)
        + ue
        * r**3
        * (y1 * (k * x**2 - y1**2) / r1_4 + y2 * (k * x**2 - y2**2) / r2_4)
    )

    ux_34 = (
        -2.0 * ur * r**3 * x / m * (1.0 / r2_sq - 2.0 * m * y * y2 / r2_4)
        - 4.0
        * ue
        * r**3
        * x
        * h
        / (1.0 + m)
        * (y2 / r2_4 + m * y * (x**2 - 3.0 * y2**2) / r2_6)
    )
    uy_34 = (
        2.0
        * ur
        * r**3
        / m
        * ((m + 1.0) * y2 / r2_sq - m * y * (x**2 - y2**2) / r2_4)
        - 2.0
        * ue
        * r**3
        * h
        * (
            (x**2 - y2**2) / r2_4
            + (m / (m + 1.0)) * (2.0 * y * y2 * (3.0 * x**2 - y2**2) / r2_6)
        )
    )

    return ux_12 + ux_34, uy_12 + uy_34


def pde(x, u):
    """二维线弹性平衡方程残差：div(sigma)=0"""
    ux_x = dde.grad.jacobian(u, x, i=0, j=0)
    ux_y = dde.grad.jacobian(u, x, i=0, j=1)
    uy_x = dde.grad.jacobian(u, x, i=1, j=0)
    uy_y = dde.grad.jacobian(u, x, i=1, j=1)

    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)

    tr_eps = eps_xx + eps_yy
    sigma_xx = LAMBDA * tr_eps + 2.0 * MU * eps_xx
    sigma_yy = LAMBDA * tr_eps + 2.0 * MU * eps_yy
    sigma_xy = 2.0 * MU * eps_xy

    dsxx_dx = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
    dsxy_dy = dde.grad.jacobian(sigma_xy, x, i=0, j=1)
    dsxy_dx = dde.grad.jacobian(sigma_xy, x, i=0, j=0)
    dsyy_dy = dde.grad.jacobian(sigma_yy, x, i=0, j=1)

    f1 = dsxx_dx + dsxy_dy
    f2 = dsxy_dx + dsyy_dy
    return [f1, f2]


def top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_MIN)


def operator_sigma_yy(x, u, _):
    ux_x = dde.grad.jacobian(u, x, i=0, j=0)
    uy_y = dde.grad.jacobian(u, x, i=1, j=1)
    tr_eps = ux_x + uy_y
    return LAMBDA * tr_eps + 2.0 * MU * uy_y


def operator_sigma_xy(x, u, _):
    ux_y = dde.grad.jacobian(u, x, i=0, j=1)
    uy_x = dde.grad.jacobian(u, x, i=1, j=0)
    eps_xy = 0.5 * (ux_y + uy_x)
    return 2.0 * MU * eps_xy


def sample_tunnel_boundary_points(n=360):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xb = R_TUNNEL * np.cos(theta)
    yb = H + R_TUNNEL * np.sin(theta)
    ux_b, uy_b = vb_displacement(xb, yb)
    pts = np.column_stack([xb, yb])
    return pts, ux_b.reshape(-1, 1), uy_b.reshape(-1, 1)


def sample_farfield_points_each_edge(n_each=120):
    # 左右边界 + 底部边界，用解析值作“软锚点”，抑制远场尾部漂移
    y_side = np.linspace(Y_MIN, Y_MAX, n_each)
    x_left = np.full_like(y_side, X_MIN)
    x_right = np.full_like(y_side, X_MAX)

    x_bottom = np.linspace(X_MIN, X_MAX, n_each)
    y_bottom = np.full_like(x_bottom, Y_MAX)

    x_all = np.concatenate([x_left, x_right, x_bottom])
    y_all = np.concatenate([y_side, y_side, y_bottom])
    ux_all, uy_all = vb_displacement(x_all, y_all)
    pts = np.column_stack([x_all, y_all])
    return pts, ux_all.reshape(-1, 1), uy_all.reshape(-1, 1)


def build_and_train():
    rect = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])
    hole = dde.geometry.Disk([0.0, H], R_TUNNEL)
    geom = dde.geometry.CSGDifference(rect, hole)

    # 洞壁位移边界（加密）
    pts_tunnel, ux_tunnel, uy_tunnel = sample_tunnel_boundary_points(360)
    bc_tunnel_ux = dde.PointSetBC(pts_tunnel, ux_tunnel, component=0)
    bc_tunnel_uy = dde.PointSetBC(pts_tunnel, uy_tunnel, component=1)

    # 地表自由边界
    bc_top_syy = dde.icbc.OperatorBC(geom, operator_sigma_yy, top_boundary)
    bc_top_sxy = dde.icbc.OperatorBC(geom, operator_sigma_xy, top_boundary)

    # 截断边界软锚点（全域建模下提升尾部精度）
    pts_far, ux_far, uy_far = sample_farfield_points_each_edge(120)
    bc_far_ux = dde.PointSetBC(pts_far, ux_far, component=0)
    bc_far_uy = dde.PointSetBC(pts_far, uy_far, component=1)

    data = dde.data.PDE(
        geom,
        pde,
        [bc_tunnel_ux, bc_tunnel_uy, bc_top_syy, bc_top_sxy, bc_far_ux, bc_far_uy],
        num_domain=4000,
        num_boundary=900,
        train_distribution="LHS",
        num_test=3000,
    )

    net = dde.nn.FNN([2] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    # L-BFGS 参数适当放宽，降低线搜索不稳定
    dde.optimizers.config.set_LBFGS_options(maxiter=25000, maxls=50, gtol=1e-8)

    # 2个PDE + 6个边界项
    loss_weights = [1, 1, 12, 12, 25, 25, 8, 8]

    model.compile("adam", lr=5e-4, loss_weights=loss_weights)
    losshistory, _ = model.train(iterations=10000, display_every=500)

    last_train_loss = np.asarray(losshistory.loss_train[-1], dtype=float)
    if np.isnan(last_train_loss).any():
        raise RuntimeError("Adam阶段出现NaN损失，已中止L-BFGS。")

    model.compile("L-BFGS", loss_weights=loss_weights)
    model.train()
    return model


def evaluate_metrics(model):
    nx, ny = 180, 110
    xv = np.linspace(X_MIN, X_MAX, nx)
    yv = np.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = np.meshgrid(xv, yv)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask_hole = (pts[:, 0] ** 2 + (pts[:, 1] - H) ** 2) >= R_TUNNEL**2
    pts_out = pts[mask_hole]

    pred = model.predict(pts_out)[:, 1]
    truth = vb_displacement(pts_out[:, 0], pts_out[:, 1])[1]
    rel_l2 = np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + 1e-12)

    x_line = np.linspace(X_MIN, X_MAX, 500)
    y_line = np.zeros_like(x_line)
    uy_true = vb_displacement(x_line, y_line)[1]
    uy_pred = model.predict(np.column_stack([x_line, y_line]))[:, 1]
    rmse = np.sqrt(np.mean((uy_pred - uy_true) ** 2))
    peak_err = abs(np.max(uy_pred) - np.max(uy_true)) / (abs(np.max(uy_true)) + 1e-12)

    print(f"[指标] 域内 uy 相对L2误差: {rel_l2:.4e}")
    print(f"[指标] 地表曲线 RMSE: {rmse:.4e}")
    print(f"[指标] 峰值相对误差: {peak_err:.4e}")


def plot_case1_results(model):
    nx, ny = 240, 150
    xv = np.linspace(X_MIN, X_MAX, nx)
    yv = np.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = np.meshgrid(xv, yv)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    mask_hole = (pts[:, 0] ** 2 + (pts[:, 1] - H) ** 2) >= R_TUNNEL**2
    pts_out = pts[mask_hole]

    pred = model.predict(pts_out)
    uy_pred = pred[:, 1]
    uy_true = vb_displacement(pts_out[:, 0], pts_out[:, 1])[1]

    field_pred = np.full(pts.shape[0], np.nan)
    field_true = np.full(pts.shape[0], np.nan)
    field_pred[mask_hole] = uy_pred
    field_true[mask_hole] = uy_true
    field_pred = field_pred.reshape(ny, nx)
    field_true = field_true.reshape(ny, nx)

    x_line = np.linspace(X_MIN, X_MAX, 500)
    y_line = np.zeros_like(x_line)
    uy_line_true = vb_displacement(x_line, y_line)[1]
    uy_line_pred = model.predict(np.column_stack([x_line, y_line]))[:, 1]

    vmin = np.nanmin([field_pred, field_true])
    vmax = np.nanmax([field_pred, field_true])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    c0 = axes[0].contourf(xx, yy, field_pred, levels=40, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("(a) PINN 预测值 $u_y$")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    fig.colorbar(c0, ax=axes[0], shrink=0.88)

    c1 = axes[1].contourf(xx, yy, field_true, levels=40, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("(b) Verruijt-Booker 准确值 $u_y$")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal")
    fig.colorbar(c1, ax=axes[1], shrink=0.88)

    axes[2].plot(x_line, uy_line_true, "k-", lw=2.2, label="准确值")
    axes[2].plot(x_line, uy_line_pred, "r--", lw=2.0, label="PINN预测值")
    axes[2].set_title("(c) 地表沉降对比 ($y=0$)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("地表沉降 (与位移单位一致)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("case1_fig7_reproduction_v2.png", dpi=320)
    plt.show()


if __name__ == "__main__":
    dde.config.set_random_seed(2026)
    model = build_and_train()
    evaluate_metrics(model)
    plot_case1_results(model)
    print("已输出图像: case1_fig7_reproduction_v2.png")
