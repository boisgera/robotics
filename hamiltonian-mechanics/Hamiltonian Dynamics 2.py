import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    M = m = 1.0
    l = 1.0
    g = 1.0
    return M, g, l, m


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.linalg as la
    from scipy.integrate import solve_ivp

    return la, np, plt, solve_ivp


@app.cell
def _(M, l, m, np):
    def Mq(x_theta):
        x, theta = x_theta
        return np.array(
            [
                [M + m, + m * l * np.cos(theta)  ],
                [+ m * l * np.cos(theta), m * l * l],
            ]
        )

    return (Mq,)


@app.cell
def _(Mq, g, l, la, m, np):
    def fun(t, x_theta_dx_dtheta):
        x, theta, dx, dtheta = x_theta_dx_dtheta
        x_theta = (x, theta)
        d2x, d2theta = la.inv(Mq(x_theta)) @ np.array(
            [
                +m * l * np.sin(theta) * dtheta ** 2,
                +m * g * l * np.sin(theta),
            ]
        )
        return (dx, dtheta, d2x, d2theta)

    return (fun,)


@app.cell
def _(fun, np, solve_ivp):
    result = solve_ivp(
        fun=fun,
        t_span=[0.0, 10.0],
        y0=[0.0, np.pi/2, 0.0, 0.0],
        dense_output=True,
    )
    return (result,)


@app.cell
def _(np, plt, result):
    t = np.linspace(0.0, 10.0, 1000)
    x = result.sol(t)[0]
    theta = result.sol(t)[1]
    plt.plot(t, x)
    return t, theta


@app.cell
def _(plt, t, theta):
    plt.plot(t, theta)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
