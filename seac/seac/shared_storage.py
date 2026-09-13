"""GAE and contiguous recurrent minibatches, including pre-rollout burn-in."""
import torch


def gae(rewards, values, bootstrap_values, terminated, truncated, gamma=.99, lam=.95):
    while rewards.ndim < values.ndim:
        rewards = rewards.unsqueeze(-1)
        terminated = terminated.unsqueeze(-1)
        truncated = truncated.unsqueeze(-1)
    delta = rewards + gamma * (~terminated) * bootstrap_values - values
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values[0])
    for t in reversed(range(len(values))):
        running = delta[t] + gamma * lam * (~(terminated[t] | truncated[t])) * running
        advantages[t] = running
    return advantages, advantages + values


def batches(inputs, states, resets, recurrent, num_minibatches, sequence_length,
            burn_in, history=None, shuffle=True):
    """Yield (flat rollout indices, replay inputs). Padding indices are -1.

    Leading dimensions are [time, stream]. Each real sample occurs once.
    History contains observations, BEFORE-observation states and reset flags.
    """
    T, B = inputs.shape[:2]
    device = inputs.device
    if not recurrent:
        order = torch.randperm(T * B, device=device) if shuffle else torch.arange(T * B, device=device)
        for indices in torch.tensor_split(order, min(num_minibatches, T * B)):
            yield indices, (inputs.flatten(0, 1)[indices], None, None, None, 0)
        return
    h = 0 if history is None else len(history['inputs'])
    all_inputs = inputs if h == 0 else torch.cat([history['inputs'], inputs])
    all_states = states if h == 0 else torch.cat([history['states'], states])
    all_resets = resets if h == 0 else torch.cat([history['resets'], resets])
    descriptors = [(start, stream) for start in range(0, T, sequence_length) for stream in range(B)]
    order = torch.randperm(len(descriptors)).tolist() if shuffle else list(range(len(descriptors)))
    for group in torch.tensor_split(torch.tensor(order), min(num_minibatches, len(order))):
        selected = [descriptors[i] for i in group.tolist()]
        K = len(selected)
        x = inputs.new_zeros(burn_in + sequence_length, K, inputs.shape[-1])
        reset = torch.ones(burn_in + sequence_length, K, dtype=torch.bool, device=device)
        valid = torch.zeros_like(reset)
        initial = states.new_zeros(K, states.shape[-1])
        indices = torch.full((sequence_length, K), -1, dtype=torch.long, device=device)
        for k, (start, stream) in enumerate(selected):
            begin = max(0, h + start - burn_in)
            prefix = h + start - begin
            length = min(sequence_length, T - start)
            offset = burn_in - prefix
            x[offset:burn_in + length, k] = all_inputs[begin:h + start + length, stream]
            reset[offset:burn_in + length, k] = all_resets[begin:h + start + length, stream]
            valid[offset:burn_in + length, k] = True
            initial[k] = all_states[begin, stream]
            indices[:length, k] = torch.arange(start, start + length, device=device) * B + stream
        yield indices.flatten(), (x, initial, reset, valid, burn_in)


def replay(model, batch):
    x, state, reset, valid, burn = batch
    if not model.recurrent:
        return model(x)[0]
    with torch.no_grad():
        for t in range(burn):
            _, next_state = model(x[t], state, reset[t])
            state = torch.where(valid[t, :, None], next_state, state)
    outputs = []
    for t in range(burn, len(x)):
        output, next_state = model(x[t], state, reset[t])
        state = torch.where(valid[t, :, None], next_state, state)
        outputs.append(output)
    return torch.stack(outputs).flatten(0, 1)


def retain_history(inputs, states, resets, old, burn_in):
    if not burn_in:
        return None
    current = dict(inputs=inputs, states=states, resets=resets)
    return {key: (value if old is None else torch.cat([old[key], value]))[-burn_in:].detach().clone()
            for key, value in current.items()}
