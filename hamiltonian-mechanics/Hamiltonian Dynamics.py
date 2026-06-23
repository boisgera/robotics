import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hamiltonian Dynamics with Pytorch
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray
    from scipy.integrate import solve_ivp
    import torch

    return NDArray, np, plt, solve_ivp, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hamiltonian:
    $$
    H(p, q) = K(p, q) + V(q) = \frac{1}{2} p^T M^{-1}(q) p + V(q)
    \qquad { \rm with } \qquad
    p = M(q) \dot{q}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hamilton equations:
    $$
    \begin{align}
    \dot{q} &= +\nabla_p H(p, q) \\
    \dot{p} &= -\nabla_q H(p, q) + f
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simple pendulum:

    - $q = \theta \in \mathbb{R}$,
    - $M(q) = m \in \mathbb{R}$,
    - $p = m \dot{q} \in \mathbb{R}$.



    $$
    K(q, p) = \frac{m\dot{q}^2}{2} = \frac{p^2}{2 m},
    \qquad
    V(q) = - m g \ell \cos q
    $$
    """)
    return


@app.cell
def _(torch):
    m = 1.0
    g = 1.0
    l = 1.0

    def K(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 / m * p * p

    def V(q: torch.Tensor) -> torch.Tensor:
        return - m * g * l * torch.cos(q)

    def H(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return K(q, p) + V(q)

    return (H,)


@app.cell
def _(NDArray, np, torch):
    def F(H):
        def grad_H(q: NDArray, p: NDArray) -> NDArray:
            q = torch.tensor(q, requires_grad=True, dtype=torch.float64)
            p = torch.tensor(p, requires_grad=True, dtype=torch.float64)
            Hpq = H(q, p)
            Hpq.backward()
            dq = q.grad
            dp = p.grad
            return np.concatenate((dq.numpy(), dp.numpy()), 0)
        def f(t: float, qp: NDArray) -> NDArray:
            qp = np.array(qp)
            n = len(qp) // 2
            q, p = qp[:n], qp[n:]
            grad_Hqp = grad_H(q, p)
            d_qp = np.concatenate((grad_Hqp[n:], -grad_Hqp[:n]), 0)
            return d_qp
        return f

    return (F,)


@app.cell
def _(F, H, np, solve_ivp):
    t_span = [0.0, 20.0]
    r = solve_ivp(fun=F(H), y0 = [np.pi/2, 0.0], t_span=t_span, dense_output=True)
    return r, t_span


@app.cell
def _(np, plt, r, t_span):
    t = np.linspace(t_span[0], t_span[1], 1000)
    sol = r.sol
    plt.plot(t, sol(t)[0], "C0", label=r"$q$")
    plt.plot(t, sol(t)[1], "C1", label=r"$p$")
    plt.legend()
    plt.grid(True)
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
