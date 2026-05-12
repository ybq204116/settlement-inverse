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

UR = 1.0  # 围岩径向位移因子 (cm, 与论文图6保持一致的量级设定)
UE = 2.0  # 围岩椭圆化位移因子 (cm)

LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


def vb_displacement(x, y, r=R_TUNNEL, h=H, nu=NU, ur=UR, ue=UE):
    """Verruijt-Booker 位移解析解（地表半无限空间）.
    输入:
        x, y: 可为标量或 ndarray
    输出:
        ux, uy: 与输入同形状的位移分量
    """
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

    # 式(1)(2): 奇异解
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

    # 式(3)(4): 自由面修正项
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

    ux = ux_12 + ux_34
    uy = uy_12 + uy_34
    return ux, uy


def pde(x, u):
    """基于拉梅常数的二维线弹性平衡方程残差.
    u[:, 0]=ux, u[:, 1]=uy
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]

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

    f1 = dsxx_dx + dsxy_dy  # x方向平衡
    f2 = dsxy_dx + dsyy_dy  # y方向平衡
    return [f1, f2]


def top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_MIN)


def operator_sigma_yy(x, u, _):
    ux_x = dde.grad.jacobian(u, x, i=0, j=0)
    uy_y = dde.grad.jacobian(u, x, i=1, j=1)
    tr_eps = ux_x + uy_y
    sigma_yy = LAMBDA * tr_eps + 2.0 * MU * uy_y
    return sigma_yy


def operator_sigma_xy(x, u, _):
    ux_y = dde.grad.jacobian(u, x, i=0, j=1)
    uy_x = dde.grad.jacobian(u, x, i=1, j=0)
    eps_xy = 0.5 * (ux_y + uy_x)
    sigma_xy = 2.0 * MU * eps_xy
    return sigma_xy


def sample_tunnel_boundary_points(n=150):
    theta = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    xb = R_TUNNEL * np.cos(theta)
    yb = H + R_TUNNEL * np.sin(theta)
    ux_b, uy_b = vb_displacement(xb, yb)
    pts = np.column_stack([xb, yb])
    return pts, ux_b.reshape(-1, 1), uy_b.reshape(-1, 1)


def build_and_train():
    # 几何域：矩形减去隧道圆孔
    rect = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])
    hole = dde.geometry.Disk([0.0, H], R_TUNNEL)
    geom = dde.geometry.CSGDifference(rect, hole)

    # 洞壁位移样本（算例1中对应边界样本）
    pts_tunnel, ux_b, uy_b = sample_tunnel_boundary_points(150)
    bc_tunnel_ux = dde.PointSetBC(pts_tunnel, ux_b, component=0)
    bc_tunnel_uy = dde.PointSetBC(pts_tunnel, uy_b, component=1)

    # 自由地表: sigma_yy = 0, sigma_xy = 0
    bc_top_syy = dde.icbc.OperatorBC(geom, operator_sigma_yy, top_boundary)
    bc_top_sxy = dde.icbc.OperatorBC(geom, operator_sigma_xy, top_boundary)

    data = dde.data.PDE(
        geom,
        pde,
        [bc_tunnel_ux, bc_tunnel_uy, bc_top_syy, bc_top_sxy],
        num_domain=1500,
        num_boundary=400,
        train_distribution="LHS",
        num_test=2000,
    )

    net = dde.nn.FNN([2] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    # 两阶段优化：Adam -> L-BFGS
    model.compile(
        "adam",
        lr=1e-3,
        # 2个PDE残差 + 4个边界条件
        loss_weights=[1, 1, 10, 10, 10, 10],
    )
    losshistory, _ = model.train(iterations=8000, display_every=500)

    # 若Adam阶段仍出现NaN，先停止切换L-BFGS，避免触发线搜索崩溃
    last_train_loss = np.asarray(losshistory.loss_train[-1], dtype=float)
    if np.isnan(last_train_loss).any():
        raise RuntimeError(
            "Adam阶段出现NaN损失，已中止L-BFGS。请检查边界条件或降低学习率。"
        )

    model.compile("L-BFGS")
    model.train()
    return model


def plot_case1_results(model):
    # 网格对比云图
    nx, ny = 220, 140
    xv = np.linspace(X_MIN, X_MAX, nx)
    yv = np.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = np.meshgrid(xv, yv)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    # 去除洞内点
    mask_hole = (pts[:, 0] ** 2 + (pts[:, 1] - H) ** 2) >= R_TUNNEL**2
    pts_out = pts[mask_hole]

    pred = model.predict(pts_out)
    _, uy_true = vb_displacement(pts_out[:, 0], pts_out[:, 1])
    uy_pred = pred[:, 1]

    # 恢复到网格
    field_pred = np.full(pts.shape[0], np.nan)
    field_true = np.full(pts.shape[0], np.nan)
    field_pred[mask_hole] = uy_pred
    field_true[mask_hole] = uy_true
    field_pred = field_pred.reshape(ny, nx)
    field_true = field_true.reshape(ny, nx)

    # 地表沉降曲线
    x_line = np.linspace(X_MIN, X_MAX, 400)
    y_line = np.zeros_like(x_line)
    uy_line_true = vb_displacement(x_line, y_line)[1]
    uy_line_pred = model.predict(np.column_stack([x_line, y_line]))[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    c0 = axes[0].contourf(xx, yy, field_pred, levels=35, cmap="viridis")
    axes[0].set_title("(a) PINN 预测值 $u_y$")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    fig.colorbar(c0, ax=axes[0], shrink=0.88)

    c1 = axes[1].contourf(xx, yy, field_true, levels=35, cmap="viridis")
    axes[1].set_title("(b) Verruijt-Booker 准确值 $u_y$")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal")
    fig.colorbar(c1, ax=axes[1], shrink=0.88)

    axes[2].plot(x_line, uy_line_true, "k-", lw=2, label="准确值")
    axes[2].plot(x_line, uy_line_pred, "r--", lw=2, label="PINN预测值")
    axes[2].set_title("(c) 地表沉降对比 ($y=0$)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("地表沉降 (与输入位移单位一致)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("case1_fig7_reproduction.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dde.config.set_random_seed(2026)
    model = build_and_train()
    plot_case1_results(model)
    print("已输出图像: case1_fig7_reproduction.png")
