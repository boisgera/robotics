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
l_1 = 0.5
l_2 = 0.25
beta_1 = 10.0
beta_2 = 10.0
q = torch.tensor([torch.pi/2, torch.pi/4], dtype=torch.float64) # [theta_1, theta_2]
p = torch.tensor([0.0, 0.0], dtype=torch.float64)

def M(q):
    theta_1, theta_2 = q
    # P: d_theta_1, d_theta_2 -> d_theta_1, d(theta_1 + theta_2)
    P = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    Q = torch.linalg.inv(P)
    M_ = m * torch.tensor([
        [l_1*l_1, l_1*l_2* (theta_2.cos())], 
        [l_1*l_2* (theta_2.cos()), l_2*l_2 ]
    ])
    return Q * M_ * P

def K(q, p):
    return 0.5 * p @ torch.linalg.inv(M(q)) @ p

def V(q):
    theta_1, theta_2 = q
    return - m * g * ( l_1 * theta_1.cos() + l_2 * (theta_1 + theta_2).cos())

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
    print(h.item())
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
    v = torch.linalg.inv(M(q)) @ p
    beta = torch.tensor([beta_1, beta_2], dtype=torch.float64)
    c = torch.tensor([c_1, c_2], dtype=torch.float64)
    p += - dt * grad_q_H  - dt * beta @ v + dt * c

def draw():
    pyxel.cls(7)
    ic, jc = xy_to_ij(0.0, 0.0)
    theta_1, theta_2 = q
    ii, ji = xy_to_ij((
        l_1 * theta_1.sin()).item(), 
        -(l_1 * theta_1.cos()).item())
    ip, jp = xy_to_ij((
        l_1 * theta_1.sin() + l_2 * (theta_1 + theta_2).sin()).item(), 
        -(l_1 * theta_1.cos() + l_2 * (theta_1 + theta_2).cos()).item())
    pyxel.line(ic, jc, ii, ji, 3)
    pyxel.line(ii, ji, ip, jp, 3)
    pyxel.circ(ip, jp, WIDTH/100.0, 4)

pyxel.init(width=WIDTH, height=HEIGHT, title="Pendulum")
pyxel.run(update, draw)
