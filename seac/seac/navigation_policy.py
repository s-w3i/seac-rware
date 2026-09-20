"""Simulator-independent local policy, message preprocessing, and CTDE critic."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

SCHEMA = 'ego-navigation-v1'
STATE_SCHEMA = 'pooled-warehouse-v1'
LOCAL_SIZE = 7 * 25 + 18
PACKET_SIZE = 20
ROBOT_STATE_SIZE = LOCAL_SIZE + 6
HEADINGS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
SPEC = dict(schema=SCHEMA, grid_shape=[7, 5, 5], scalar_size=18,
            grid_channels=['in_bounds', 'shelf', 'robot', 'heading_forward', 'heading_back', 'heading_left', 'heading_right'],
            scalars=['carrying', 'phase[6]', 'goal_bearing[right,forward]', 'goal_distance/(1+distance)',
                     'previous_action[4]', 'movement_success', 'movement_denied', 'stationary_age/64', 'cycle_age/1000'],
            packet_size=PACKET_SIZE, actions=['wait', 'forward', 'left', 'right'],
            recurrent_size=128, absolute_coordinates=False, critic_required=False,
            frame='right,forward; raster row 0 is two cells forward; col 0 is two cells left',
            distance_units='grid cells; Euclidean goal distance',
            age_clipping={'stationary': 64, 'cycle': 1000},
            packet_fields=['relative_position[right,forward]/4', 'relative_heading[right,forward]',
                           'goal_bearing[right,forward]', 'goal_distance/(1+distance)', 'phase[6]',
                           'carrying', 'previous_action[4]', 'clipped_cycle_age/1000', 'packet_age/2'],
            channel={'range': 2, 'range_metric': 'Chebyshev', 'delay': 1, 'expiry': 2,
                     'training_recipient_loss': .1})


def relative(vector, heading):
    forward = HEADINGS[heading]
    right = np.array([-forward[1], forward[0]])
    return np.array([np.dot(vector, right), np.dot(vector, forward)], dtype=np.float32)


def goal_features(delta):
    distance = np.linalg.norm(delta)
    return np.r_[delta / distance if distance else np.zeros(2), distance / (1 + distance)]


def packet_features(packet, receiver_position, receiver_heading, now):
    """Only sender-supplied status + receiver's own pose; no world queries."""
    age = now - packet['step']
    if age < 1 or age > 2:
        return None
    delta = relative(np.asarray(packet['position']) - receiver_position, receiver_heading)
    direction = relative(HEADINGS[packet['heading']], receiver_heading)
    goal = relative(np.asarray(packet['goal']) - receiver_position, receiver_heading)
    result = np.r_[delta / 4, direction, goal_features(goal), np.eye(6)[packet['phase']],
                   packet['carrying'], np.eye(4)[packet['previous_action']],
                   min(packet['cycle_age'], 1000) / 1000, age / 2].astype(np.float32)
    assert result.shape == (PACKET_SIZE,)
    return result


def pack_observation(local, inbox=()):
    local = np.asarray(local, dtype=np.float32)
    if local.shape != (LOCAL_SIZE,) or not np.isfinite(local).all():
        raise ValueError('Invalid local observation')
    messages = np.asarray(inbox, dtype=np.float32).reshape(-1, PACKET_SIZE)
    if not np.isfinite(messages).all():
        raise ValueError('Nonfinite packet')
    return np.r_[local, np.c_[np.ones(len(messages)), messages].ravel()].astype(np.float32)


class NavigationActor(nn.Module):
    recurrent = True
    hidden_size = 128

    def __init__(self, communication=False):
        super().__init__()
        self.communication = communication
        self.grid = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(32 * 25, 96), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(18, 32), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(PACKET_SIZE, 32), nn.ReLU())
        self.query = nn.Linear(128, 32, bias=False)
        self.key = nn.Linear(32, 32, bias=False)
        self.value = nn.Linear(32, 32, bias=False)
        self.gru = nn.GRUCell(160, 128)
        self.head = nn.Linear(128, 4)
        self.rngs = None

    def initial_state(self, shape, device):
        return torch.zeros(*shape, 128, device=device)

    def seed_streams(self, seed, count):
        self.rngs = [np.random.default_rng(np.random.SeedSequence([seed, i, 0xAC70])) for i in range(count)]

    def forward(self, obs, state=None, reset=None):
        if obs.shape[-1] < LOCAL_SIZE or (obs.shape[-1] - LOCAL_SIZE) % (PACKET_SIZE + 1):
            raise ValueError('Incompatible navigation observation')
        shape = obs.shape[:-1]
        x = obs.reshape(-1, obs.shape[-1])
        own = torch.cat([self.grid(x[:, :175].reshape(-1, 7, 5, 5)), self.scalars(x[:, 175:LOCAL_SIZE])], -1)
        context = own.new_zeros(len(x), 32)
        if self.communication and x.shape[-1] > LOCAL_SIZE:
            packets = x[:, LOCAL_SIZE:].reshape(len(x), -1, PACKET_SIZE + 1)
            valid = packets[..., 0] > 0
            encoded = self.message(packets[..., 1:])
            scores = (self.query(own)[:, None] * self.key(encoded)).sum(-1) / np.sqrt(32)
            # A finite sentinel and explicit renormalization make empty inboxes exactly zero.
            weights = torch.softmax(scores.masked_fill(~valid, -1e9), -1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            context = (weights[..., None] * self.value(encoded)).sum(1)
        if state is None:
            state = self.initial_state(shape, obs.device)
        if reset is not None:
            state = state * (~reset).unsqueeze(-1)
        state = self.gru(torch.cat([own, context], -1), state.reshape(-1, 128)).reshape(*shape, 128)
        return self.head(state), state

    def act(self, obs, state=None, reset=None, deterministic=False):
        logits, state = self(obs, state, reset)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(-1)
        else:
            probabilities = dist.probs.reshape(-1, 4)
            if self.rngs is None or len(self.rngs) != len(probabilities):
                raise ValueError('Initialize independent robot action RNG streams before sampling')
            uniform = torch.tensor([r.random() for r in self.rngs], device=logits.device, dtype=logits.dtype)
            action = (uniform[:, None] > probabilities.cumsum(-1)).sum(-1).clamp_max(3).reshape(logits.shape[:-1])
        return action, dist.log_prob(action), state


class NavigationCritic(nn.Module):
    recurrent = False
    hidden_size = 1

    def __init__(self, map_shape):
        super().__init__()
        self.map_shape = tuple(map_shape)
        self.conv1, self.conv2 = nn.Conv2d(4, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
        self.robot = nn.Sequential(nn.Linear(ROBOT_STATE_SIZE, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(320, 256), nn.ReLU(), nn.Linear(256, 1))

    def initial_state(self, shape, device):
        return torch.zeros(*shape, 1, device=device)

    def forward(self, state, hidden=None, reset=None):
        shape = state.shape[:-1]
        x = state.reshape(-1, state.shape[-1])
        size = 4 * np.prod(self.map_shape)
        maps = x[:, :size].reshape(-1, 4, *self.map_shape)
        mask = maps[:, 3:4]
        image = torch.relu(self.conv1(maps * mask)) * mask
        image = torch.relu(self.conv2(image)) * mask
        mean = image.sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        maximum = image.masked_fill(mask == 0, -1e9).amax((2, 3))
        tokens = x[:, size:].reshape(len(x), -1, ROBOT_STATE_SIZE + 1)
        valid = tokens[..., :1]
        nodes = self.robot(tokens[..., 1:])
        pooled = (nodes * valid).sum(1) / valid.sum(1).clamp_min(1)
        peak = nodes.masked_fill(valid == 0, -1e9).amax(1)
        return self.head(torch.cat([mean, maximum, pooled, peak], -1)).reshape(*shape, 1), hidden


class PolicyRuntime:
    """One copy per AGV. Requires only numpy/torch and this module."""

    def __init__(self, package, device='cpu'):
        saved = torch.load(package, map_location=device, weights_only=False)
        if saved.get('observation_spec') != SPEC or saved.get('format') != 'navigation-actor-v1':
            raise ValueError('Incompatible actor package')
        self.actor = NavigationActor(saved['communication']).to(device).eval()
        self.actor.load_state_dict(saved['actor'])
        self.device = device

    @torch.no_grad()
    def act(self, local_observation, inbox, recurrent_state, rng):
        x = torch.as_tensor(pack_observation(local_observation, inbox), device=self.device).unsqueeze(0)
        state = None if recurrent_state is None else torch.as_tensor(recurrent_state, dtype=torch.float32, device=self.device).reshape(1, 128)
        self.actor.rngs = [rng]
        action, _, next_state = self.actor.act(x, state)
        return int(action.item()), next_state[0].cpu().numpy()


def export_actor(actor, path):
    torch.save(dict(format='navigation-actor-v1', observation_spec=SPEC, communication=actor.communication,
                    actor={k: v.detach().cpu() for k, v in actor.state_dict().items()}), Path(path))
