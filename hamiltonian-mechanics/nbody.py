import time

import pyxel
import torch


WIDTH = 800
HEIGHT = 800

g = 1.0
m1 = 1.0
m2 = 1.0
q = torch.tensor([0.5, 0.0], dtype=torch.float64)
p = torch.tensor([0.0, 1.5], dtype=torch.float64)

def K(p):
    return 0.5 * p @ p / m2

def V(p, q):
    return - g * m1 * m2 / q.norm()

def H(p, q):
    return K(p) + V(p, q)

def grad_H(p, q):
    p = p.detach().clone()
    p.requires_grad = True
    q = q.detach().clone()
    q.requires_grad = True
    h = H(p, q)
    h.backward()
    return p.grad, q.grad

t = None

def update():
    global t
    global p, q
    if t is None:
        t = time.time()
    t_new = time.time()
    dt = t - t_new
    t = t_new
    grad_p_H, grad_q_H = grad_H(p, q)
    q += (dt * grad_p_H)
    q = q.clip(-1.0, 1.0)
    p -= (dt * grad_q_H)

def draw():
    pyxel.cls(7)
    pyxel.circ(400, 400, 10.0*m1, 0)
    pyxel.circ(400.0 + 400 * q[0], 400.0 + 400.0 * q[1], 10.0*m2, 0)


pyxel.init(width=800, height=800, title="N-body problem")
pyxel.run(update, draw)
