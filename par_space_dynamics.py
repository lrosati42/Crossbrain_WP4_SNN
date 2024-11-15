import numpy as np

def denorm_params(x0, CpES):
    x0_MIN = 2.
    x0_MAX = 4.5
    CpES_MIN = 0.1
    CpES_MAX = 1.
    x0_o = - ((x0 * (x0_MAX - x0_MIN)) + x0_MIN)
    CpES_o = (CpES * (CpES_MAX - CpES_MIN)) + CpES_MIN
    return x0_o, CpES_o

def direction_grid(ds):
    start = 0
    mid = int(0.6/ds)
    dx = int(0.1/ds)
    dy = int(0.1/ds)
    end = int(1./ds)
    idx = np.arange(end-start)
    P = np.ones(shape=(idx.size, idx.size), dtype=int)

    P[:dx,:dy] *= 2
    P[dx:mid-dx,:dy] *= 1
    P[mid-dx:mid+dx,:dy] *= 8
    P[mid+dx:end-dx,:dy] *= 1
    P[end-dx:,:dy] *= 8

    P[:dx,dy:mid-dy] *= 3
    P[dx:mid-dx,dy:mid-dy] *= 0 # I zone
    P[mid-dx:mid+dx,dy:mid-dy] *= 7
    P[mid+dx:end-dx,dy:mid-dy] *= 0 # IV zone
    P[end-dx:,dy:mid-dy] *= 7

    P[:dx,mid-dy:mid+dy] *= 2
    P[dx:mid-dx,mid-dy:mid+dy] *= 1
    # P[mid-dx:mid+dx,mid-dy:mid+dy] *= 0 # random zone
    P[mid+dx:end-dx,mid-dy:mid+dy] *= 5
    P[end-dx:,mid-dy:mid+dy] *= 6

    P[:dx,mid+dy:end-dy] *= 3
    P[dx:mid-dx,mid+dy:end-dy] *= 0 # II zone
    P[mid-dx:mid+dx,mid+dy:end-dy] *= 3
    P[mid+dx:end-dx,mid+dy:end-dy] *= 0 # III zone
    P[end-dx:,mid+dy:end-dy] *= 7

    P[:dx,end-dy:] *= 4
    P[dx:mid-dx,end-dy:] *= 5
    P[mid-dx:mid+dx,end-dy:] *= 4
    P[mid+dx:end-dx,end-dy:] *= 5
    P[end-dx:,end-dy:] *= 6

    P[mid-dx:mid,mid-dy:mid] *= 6
    P[mid:mid+dx,mid-dy:mid] *= 4
    P[mid-dx:mid,mid:mid+dy] *= 8
    P[mid:mid+dx,mid:mid+dy] *= 2

    return P

def direction_map(d):
    '''
    0: RANDOM
    1: ↑
    2: ↗
    3: →
    4: ↘
    5: ↓
    6: ↙
    7: ←
    8: ↖
    '''

    if d == 1: # ↑
        p = {
            'up': 0.3,
            'down': 0.2,
            'left': 0.25,
            'right': 0.25
        }
    elif d == 2: # ↗
        p = {
            'up': 0.3,
            'down': 0.2,
            'left': 0.2,
            'right': 0.3
        }
    elif d == 3: # →
        p = {
            'up': 0.25,
            'down': 0.25,
            'left': 0.2,
            'right': 0.3
        }
    elif d == 4: # ↘
        p = {
            'up': 0.3,
            'down': 0.2,
            'left': 0.2,
            'right': 0.3
        }
    elif d == 5: # ↓
        p = {
            'up': 0.2,
            'down': 0.3,
            'left': 0.25,
            'right': 0.25
        }
    elif d == 6: # ↙
        p = {
            'up': 0.2,
            'down': 0.3,
            'left': 0.3,
            'right': 0.2
        }
    elif d == 7: # ←
        p = {
            'up': 0.25,
            'down': 0.25,
            'left': 0.3,
            'right': 0.2
        }
    elif d == 8: # ↖
        p = {
            'up': 0.3,
            'down': 0.2,
            'left': 0.3,
            'right': 0.2
        }
    else:
        p = {
            'up': 0.25,
            'down': 0.25,
            'left': 0.25,
            'right': 0.25
        }

    return p

class Parameter_Walker():
    def __init__(self, ds = 0.01, x0_0 = 0.2, CpES_0 = 0.2):
        self.name = "2D random walker in the epileptor parameter space"
        self.ds = ds
        self.x0_0 = x0_0
        self.CpES_0 = CpES_0
        self.MIN = 0
        self.MAX = 1

        self.P = direction_grid(self.ds)
        # Inizializza le coordinate
        self.x, self.y = np.array([x0_0]), np.array([CpES_0])

        # Mappa le direzioni ai cambiamenti nelle coordinate
        self.directions = {
            'up': (0, self.ds),
            'down': (0, -self.ds),
            'left': (-self.ds, 0),
            'right': (self.ds, 0)
        }

    def __str__(self):
        return f"{self.name}."

    def walk(self, steps, return_traj = False, return_points = False):
        for _ in range(steps):
            idx = int(self.x[-1]/self.ds)
            idy = int(self.y[-1]/self.ds)
            arrow = self.P[idx, idy]
            probabilities = direction_map(arrow)
            direction = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
            dx, dy = self.directions[direction]
            # Calcola le nuove coordinate
            new_x, new_y = self.x[-1] + dx, self.y[-1] + dy
            # Controlla le condizioni al contorno e applica il rimbalzo
            if new_x < self.MIN or new_x >= self.MAX:
                dx = -dx
            if new_y < self.MIN or new_y >= self.MAX:
                dy = -dy
            self.x = np.append(self.x, self.x[-1] + dx)
            self.y = np.append(self.y, self.y[-1] + dy)

        if return_traj:
            return self.x, self.y  
        elif return_points:
            return self.x[-1], self.y[-1]