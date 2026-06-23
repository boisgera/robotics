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
beta = 0.5
q = torch.tensor(torch.pi/2, dtype=torch.float64)
p = torch.tensor(0.0, dtype=torch.float64)

def K(q, p):
    return 0.5 * p * p / (m * l * l)

def V(q):
    return - m * g * l * q.cos()

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

    c = 0.0
    if pyxel.btn(pyxel.KEY_UP):
        c = +3.0
    elif pyxel.btn(pyxel.KEY_DOWN):
        c = -3.0

    grad_q_H, grad_p_H = grad_H(q, p)
    q += dt * grad_p_H
    v = p / (m * l * l)
    p += - dt * grad_q_H  - dt * beta * v + dt * c

def draw():
    pyxel.cls(7)
    ic, jc = xy_to_ij(0.0, 0.0)
    ip, jp = xy_to_ij((l * q.sin()).item(), (-l * q.cos()).item())
    pyxel.line(ic, jc, ip, jp, 3)
    pyxel.circ(ip, jp, WIDTH/100.0, 4)

pyxel.init(width=WIDTH, height=HEIGHT, title="Pendulum")
pyxel.run(update, draw)
