from typing import Callable, Generic, List, TypeVar

from torch import Tensor, nn
from torch.distributions import Categorical, MultivariateNormal

DistrType = TypeVar("DistrType", MultivariateNormal, Categorical)
DistrFactory = Callable[[Tensor], DistrType]


class Actor(nn.Module, Generic[DistrType]):
    def forward(self, inputs: Tensor) -> DistrType:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> DistrType:
        return self.forward(inputs)


class Critic(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)


class FFActor(Actor[DistrType]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hiddens: List[int],
        distr_factory: DistrFactory,
    ):
        super().__init__()
        layers = [nn.Linear(in_features, hiddens[0]), nn.Tanh()]
        h1 = hiddens[0]
        for h0, h1 in zip(hiddens[:-1], hiddens[1:]):
            layers.extend(
                [
                    nn.BatchNorm1d(h0),
                    nn.Linear(h0, h1),
                    nn.Tanh(),
                ]
            )
        layers.extend(
            [
                # nn.BatchNorm1d(h1),
                nn.Linear(h1, out_features),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.distr_factory = distr_factory

    def forward(self, inputs: Tensor) -> DistrType:
        policy_output = self.layers(inputs)
        distr = self.distr_factory(policy_output)
        return distr


class FFCritic(Critic):
    def __init__(self, in_features: int, hiddens: List[int]):
        super().__init__()
        layers = [nn.Linear(in_features, hiddens[0]), nn.Tanh()]
        h1 = hiddens[0]
        for h0, h1 in zip(hiddens[:-1], hiddens[1:]):
            layers.extend([nn.BatchNorm1d(h0), nn.Linear(h0, h1), nn.Tanh()])
        layers.extend([nn.Linear(h1, 1)])
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        critic_values = self.layers(inputs)
        return critic_values
