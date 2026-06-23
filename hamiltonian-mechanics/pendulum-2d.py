import time

import pyxel
import torch

WIDTH = 400
HEIGHT = 400

# -1.0 <= x, y <= +1.0
def xy_to_ij(x, y):
    return (
        round(0.5 * (+x + 1.0) * WIDTH),
        round(0.5 * (-y + 1.0) * HEIGHT)
    )

g = 10.0
m = 1.0
l = 0.5
beta_1 = 0.5
beta_2 = 10.0
q = torch.tensor([torch.pi/2, 0.0], dtype=torch.float64) # [theta, alpha]
p = torch.tensor([0.0, 0.0], dtype=torch.float64)

def M(q):
    theta = q[0]
    return m * l * l * torch.tensor([[1.0, 0.0], [0.0, theta.sin()**2]], dtype=torch.float64)

def K(q, p):
    eps = 1e-10
    return 0.5 * p @ torch.linalg.inv(M(q) + eps * torch.eye(2, dtype=torch.float64)) @ p

def V(q):
    theta = q[0]
    return - m * g * l * theta.cos()

def H(q, p):
    return K(q, p) + V(q)

def grad_H(q, p):
    q = q.detach().clone()
    q.requires_grad = True
    p = p.detach().clone()
    p.requires_grad = True
    h = H(q, p)
    # print("H:", h.item(), "q, p:", q.item(), p.item())
    h.backward()
    return q.grad.detach().clone(), p.grad.detach().clone()

#print(grad_H(q, p))

t = None

def update():
    global t
    global q, p

    if t is None:
        t = time.time()
    t_new = time.time()
    dt = t_new - t
    t = t_new

    c_1 = 0.0
    if pyxel.btn(pyxel.KEY_UP):
        c_1 = +3.0
    elif pyxel.btn(pyxel.KEY_DOWN):
        c_1 = -3.0

    c_2 = 0.0
    if pyxel.btn(pyxel.KEY_LEFT):
        c_2 = +3.0
    elif pyxel.btn(pyxel.KEY_RIGHT):
        c_2 = -3.0

    grad_q_H, grad_p_H = grad_H(q, p)
    q += dt * grad_p_H
    v = p / (m * l * l)
    c = torch.tensor([c_1, c_2], dtype=torch.float64)
    B = torch.tensor([[beta_1, 0.0], [0.0, beta_2]], dtype=torch.float64)
    p += - dt * grad_q_H - dt * B @ v + dt * c

def draw():
    pyxel.cls(7)
    ic, jc = xy_to_ij(0.0, 0.0)
    theta, alpha = q
    ip, jp = xy_to_ij((l * theta.sin() * alpha.cos()).item(), (-l * theta.cos()).item())
    pyxel.line(ic, jc, ip, jp, 3)
    pyxel.circ(ip, jp, WIDTH/100.0, 4)

pyxel.init(width=WIDTH, height=HEIGHT, title="Pendulum")
pyxel.run(update, draw)
