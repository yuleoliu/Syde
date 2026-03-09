import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from warnings import warn
import math


class TorchMiniSom:
    Y_HEX_CONV_FACTOR = (3.0 / 2.0) / math.sqrt(3)

    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function="asymptotic_decay",
                 neighborhood_function="gaussian", topology="rectangular",
                 activation_distance="euclidean", random_seed=None,
                 sigma_decay_function="asymptotic_decay",
                 device="cpu"):
        self.device = torch.device(device)

        if sigma > math.sqrt(x * x + y * y):
            warn("Warning: sigma might be too high for the dimension of the map.")

        if random_seed is not None:
            torch.manual_seed(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len

        self._weights = torch.rand((x, y, input_len), device=self.device) * 2 - 1
        self._weights /= torch.norm(self._weights, dim=-1, keepdim=True)

        self._activation_map = torch.zeros((x, y), device=self.device)
        self._neigx = torch.arange(x, device=self.device)
        self._neigy = torch.arange(y, device=self.device)

        if topology not in ["hexagonal", "rectangular"]:
            raise ValueError("Only hexagonal and rectangular topologies are supported")
        self.topology = topology

        self._xx, self._yy = torch.meshgrid(self._neigx, self._neigy, indexing="ij")
        self._xx = self._xx.to(torch.float32)
        self._yy = self._yy.to(torch.float32)
        if topology == "hexagonal":
            self._xx[::2] -= 0.5
            self._yy *= self.Y_HEX_CONV_FACTOR
            if neighborhood_function == "triangle":
                warn("triangle neighborhood does not account for hexagonal topology")

        self._lr_decay_functions = {
            "inverse_decay_to_zero": self._inverse_decay_to_zero,
            "linear_decay_to_zero": self._linear_decay_to_zero,
            "asymptotic_decay": self._asymptotic_decay,
        }
        if isinstance(decay_function, str):
            if decay_function not in self._lr_decay_functions:
                raise ValueError(f"{decay_function} not supported")
            self._learning_rate_decay_function = self._lr_decay_functions[decay_function]
        else:
            self._learning_rate_decay_function = decay_function

        self._sigma_decay_functions = {
            "inverse_decay_to_one": self._inverse_decay_to_one,
            "linear_decay_to_one": self._linear_decay_to_one,
            "asymptotic_decay": self._asymptotic_decay,
        }
        if sigma_decay_function not in self._sigma_decay_functions:
            raise ValueError(f"{sigma_decay_function} not supported")
        self._sigma_decay_function = self._sigma_decay_functions[sigma_decay_function]

        self._neigh_functions = {
            "gaussian": self._gaussian,
            "mexican_hat": self._mexican_hat,
            "bubble": self._bubble,
            "triangle": self._triangle,
        }
        if neighborhood_function not in self._neigh_functions:
            raise ValueError(f"{neighborhood_function} not supported")
        self.neighborhood = self._neigh_functions[neighborhood_function]

        self._distance_functions = {
            "euclidean": self._euclidean_distance,
            "cosine": self._cosine_distance,
            "manhattan": self._manhattan_distance,
            "chebyshev": self._chebyshev_distance,
        }
        if isinstance(activation_distance, str):
            if activation_distance not in self._distance_functions:
                raise ValueError(f"{activation_distance} not supported")
            self._activation_distance = self._distance_functions[activation_distance]
        else:
            self._activation_distance = activation_distance

    def get_weights(self):
        return self._weights.detach().cpu().numpy()


    def get_euclidean_coordinates(self):
        return self._xx.T.cpu().numpy(), self._yy.T.cpu().numpy()

    def convert_map_to_euclidean(self, xy):
        return self._xx.T[xy], self._yy.T[xy]

    def _activate(self, x):
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        self._activate(x)
        return self._activation_map

    def _inverse_decay_to_zero(self, lr, t, max_iter):
        C = max_iter / 100.0
        return lr * C / (C + t)

    def _linear_decay_to_zero(self, lr, t, max_iter):
        return lr * (1 - t / max_iter)

    def _inverse_decay_to_one(self, sigma, t, max_iter):
        C = (sigma - 1) / max_iter
        return sigma / (1 + (t * C))

    def _linear_decay_to_one(self, sigma, t, max_iter):
        return sigma + (t * (1 - sigma) / max_iter)

    def _asymptotic_decay(self, param, t, max_iter):
        return param / (1 + t / (max_iter / 2))

    def _gaussian(self, c, sigma):
        d = 2 * sigma * sigma
        ax = torch.exp(-((self._xx - self._xx[c]) ** 2) / d)
        ay = torch.exp(-((self._yy - self._yy[c]) ** 2) / d)
        return (ax * ay).T.contiguous()

    def _mexican_hat(self, c, sigma):
        p = (self._xx - self._xx[c]) ** 2 + (self._yy - self._yy[c]) ** 2
        d = 2 * sigma * sigma
        return (torch.exp(-p / d) * (1 - 2 * p / d)).T

    def _bubble(self, c, sigma):
        ax = (self._neigx > c[0] - sigma) & (self._neigx < c[0] + sigma)
        ay = (self._neigy > c[1] - sigma) & (self._neigy < c[1] + sigma)
        return torch.outer(ax.float(), ay.float())

    def _triangle(self, c, sigma):
        triangle_x = (-torch.abs(c[0] - self._neigx)) + sigma
        triangle_y = (-torch.abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.0
        triangle_y[triangle_y < 0] = 0.0
        return torch.outer(triangle_x, triangle_y)

    def _euclidean_distance(self, x, w):
        return torch.norm(x - w, dim=-1)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(dim=2)
        denom = torch.norm(w, dim=2) * torch.norm(x)
        return 1 - num / (denom + 1e-8)

    def _manhattan_distance(self, x, w):
        return torch.norm(x - w, p=1, dim=-1)

    def _chebyshev_distance(self, x, w):
        return torch.max(torch.abs(x - w), dim=-1).values

    def winner(self, x):
        self._activate(x)
        idx = torch.argmin(self._activation_map)
        return torch.unravel_index(idx, self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        eta = self._learning_rate_decay_function(self._learning_rate, t, max_iteration)
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        g = self.neighborhood(win, sig) * eta
        self._weights += torch.einsum("ij,ijk->ijk", g, (x - self._weights))

    def train(self, data, num_iteration, random_order=False, use_epochs=False, fixed_points=None):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if use_epochs:
            max_iter = num_iteration
            total_iter = num_iteration * len(data)
        else:
            max_iter = num_iteration
            total_iter = num_iteration

        for t in range(total_iter):
            if use_epochs:
                epoch = t // len(data)
                idx = t % len(data)
            else:
                idx = torch.randint(len(data), (1,)).item() if random_order else t % len(data)
                epoch = t
            win = fixed_points.get(idx, self.winner(data[idx])) if fixed_points else self.winner(data[idx])
            self.update(data[idx], win, epoch, max_iter)

    def train_random(self, data, num_iteration):
        self.train(data, num_iteration, random_order=True)

    def train_batch(self, data, num_iteration):
        self.train(data, num_iteration, random_order=False)
    def batch_winner(self, data):
        # data: (N, feat_dim)
        # weights: (x, y, feat_dim)
        w_flat = self._weights.view(-1, self._input_len)   # (x*y, feat_dim)
        dists = torch.cdist(data, w_flat)                  # (N, x*y)
        bmu_indices = torch.argmin(dists, dim=1)           # (N,)
        return torch.unravel_index(bmu_indices, (self._weights.shape[0], self._weights.shape[1]))
    def quantization(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        winners = []
        for x in data:
            winners.append(self._weights[self.winner(x)])
        return torch.stack(winners)

    def quantization_error(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        q = self.quantization(data)
        return torch.norm(data - q, dim=1).mean().item()

    def distortion_measure(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        distortion = 0
        for d in data:
            distortion += (self.neighborhood(self.winner(d), self._sigma) *
                           torch.norm(d - self._weights, dim=2)).sum()
        return distortion.item()

    def topographic_error(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        total = 0
        error = 0
        for d in data:
            distances = torch.norm(self._weights - d, dim=2).flatten()
            bmu = torch.argsort(distances)[:2]
            coords = [torch.unravel_index(idx, self._weights.shape[:2]) for idx in bmu]
            dx, dy = abs(coords[0][0]-coords[1][0]), abs(coords[0][1]-coords[1][1])
            if dx+dy > 1:  # not neighbors
                error += 1
            total += 1
        return error / total

    def distance_map(self):
        um = torch.zeros((self._weights.shape[0], self._weights.shape[1]), device=self.device)
        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                w = self._weights[x, y]
                neighbors = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if (0 <= x+i < self._weights.shape[0]) and (0 <= y+j < self._weights.shape[1]) and (i != 0 or j != 0):
                            neighbors.append(torch.norm(w - self._weights[x+i, y+j]).item())
                if neighbors:
                    um[x, y] = sum(neighbors) / len(neighbors)
        return (um / um.max()).cpu().numpy()

    def activation_response(self, data, as_numpy=False):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        bmu_x, bmu_y = self.batch_winner(data)
        a = torch.zeros((self._weights.shape[0], self._weights.shape[1]), device=self.device)
        a.index_put_((bmu_x, bmu_y), torch.ones_like(bmu_x, dtype=torch.float32), accumulate=True)
        return a.detach().cpu().numpy() if as_numpy else a

    def win_map(self, data, return_indices=False, as_numpy=False):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        winmap = defaultdict(list)
        for i, x in enumerate(data):
            if return_indices:
                winmap[self.winner(x)].append(i)
            else:
                winmap[self.winner(x)].append(x.detach())
        if as_numpy:
            return {k: [v.cpu().numpy() for v in vs] for k, vs in winmap.items()}
        return winmap


    def labels_map(self, data, labels):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same length.")
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for pos in winmap:
            winmap[pos] = Counter(winmap[pos])
        return winmap
