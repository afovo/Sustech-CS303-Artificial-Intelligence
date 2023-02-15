import torch
import torch.nn as nn
import os.path as osp
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj
import time


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self)
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=160, bias=True),
            nn.ReLU())
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=160, out_features=80, bias=True),
            nn.ReLU())
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=80, out_features=10, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.hidden3(fc2)
        return fc1, fc2, output


def evaluate1(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor
) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hit = (d < RADIUS)
    d[hit] = 1
    d[~hit] = 0.1 / 1000 ** d[~hit]
    value = torch.sum(d * target_scores, dim=-1)
    return value


def get_rand_sol() -> torch.Tensor:
    return torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])


class Agent:

    def __init__(self) -> None:
        self.classifier = FCNN()
        model_path = osp.join(osp.dirname(__file__), "classifier.pth")
        self.classifier.load_state_dict(torch.load(model_path))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> torch.Tensor:
        """Compute the parameters required to fire a projectile.

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets.
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)
        start = time.time()
        ctps_inter = get_rand_sol()

        _, _, output = self.classifier(target_features)
        _, target_classes = torch.max(output, 1)
        target_scores = class_scores[target_classes]
        maximum = [evaluate(compute_traj(ctps_inter), target_pos, target_scores, RADIUS), ctps_inter.data]
        while True:
            ctps_inter = get_rand_sol()
            real_score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, RADIUS)
            if real_score > maximum[0]:
                maximum[0] = real_score
                maximum[1] = ctps_inter.data
            if time.time() - start > 0.1:
                break
        # print('random: ',maximum[0])
        ctps_inter.requires_grad = True
        while True:
            gra_score = evaluate1(compute_traj(ctps_inter), target_pos, target_scores)
            real_score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, RADIUS)
            # print(real_score)
            if real_score > maximum[0]:
                maximum[0] = real_score
                maximum[1] = ctps_inter.data
            t = time.time() - start
            if t >= 0.298:
                break
            t *= 11
            lr = 1 / (t ** 2)
            gra_score.backward()
            ctps_inter.data = ctps_inter.data + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
        ctps_inter.data = maximum[1]
        print('max: ', maximum[0])
        return ctps_inter
