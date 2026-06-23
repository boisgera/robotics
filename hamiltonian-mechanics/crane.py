import time

import pyxel
import torch

WIDTH = 400
HEIGHT = 400

# -1.0 <= x, y <= +1.0
def xy_to_ij(x, y):
    return (round(0.5 * (+x + 1.0) * WIDTH), round(0.5 * (-y + 1.0) * HEIGHT))


g = 10.0
m_x = 1.0
m = 1.0
l = 0.5
beta_x = 0.5 # 5.0  # 0.5
beta_theta = 0.5 # 0.5  # 0.5
q = torch.tensor([0.0, 0.25 * torch.pi], dtype=torch.float64)  # [x, theta]
p = torch.tensor([0.0, 0.0], dtype=torch.float64)


def M(q):
    x, theta = q
    return torch.tensor(
        [
            [m_x + m, m *l * (theta.cos())], 
            [m * l * (theta.cos()), m * l * l]
        ], dtype=torch.float64
    )


def K(q, p):
    return 0.5 * p @ torch.linalg.inv(M(q)) @ p


def V(q):
    x, theta = q
    eps = 0 # 0.1
    return - m * g * l * (theta.cos() - 1) + eps * x * x


def H(q, p):
    return K(q, p) + V(q)


def grad_H(q, p):
    q = q.detach().clone()
    q.requires_grad = True
    p = p.detach().clone()
    p.requires_grad = True
    h = H(q, p)
    #print("H:", h.item())
    h.backward()
    return q.grad.detach().clone(), p.grad.detach().clone()


# print(grad_H(q, p))

t = None


def update():
    global t
    global q, p

    if t is None:
        t = time.time()
    t_new = time.time()
    dt = t_new - t
    t = t_new

    f_x = 0.0
    if pyxel.btn(pyxel.KEY_RIGHT):
        f_x = +3.0
    elif pyxel.btn(pyxel.KEY_LEFT):
        f_x = -3.0
    c_theta = 0.0
    if pyxel.btn(pyxel.KEY_UP):
        c_theta = +3.0
    elif pyxel.btn(pyxel.KEY_DOWN):
        c_theta = -3.0

    grad_q_H, grad_p_H = grad_H(q, p)
    q += dt * grad_p_H
    v = torch.linalg.inv(M(q)) @ p
    # print(v)
    beta = torch.tensor([[beta_x, 0.0], [0.0, beta_theta]], dtype=torch.float64)
    f = torch.tensor([f_x, c_theta], dtype=torch.float64)
    #print("*", - dt * grad_q_H , - dt * beta @ v,  dt * c)
    #print("**", - (dt * (beta @ v) @ p).item())
    p += -dt * grad_q_H - dt * beta @ v + dt * f


def draw():
    x, theta = q
    pyxel.cls(7)
    ic, jc = xy_to_ij(x.item(), 0.0)
    ip, jp = xy_to_ij((x + l * (theta.sin())).item(), (-l * (theta.cos())).item())
    pyxel.line(ic, jc, ip, jp, 3)
    pyxel.circ(ip, jp, WIDTH / 100.0, 4)


pyxel.init(width=WIDTH, height=HEIGHT, title="Crane")
pyxel.run(update, draw)
