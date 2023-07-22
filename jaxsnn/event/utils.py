import jax.numpy as np
from jax import random

from typing import List, Optional
from jaxsnn.base.types import Spike, WeightInput, WeightRecurrent


def filter_spikes_batch(
    spikes: Spike, layer_start: int, layer_end: Optional[int] = None
):
    """Only return spikes of neurons after layer start

    Other spikes are encoded with time=np.inf and index=-1
    """
    filtered_time = np.where(spikes.idx >= layer_start, spikes.time, np.inf)
    filtered_idx = np.where(spikes.idx >= layer_start, spikes.idx, -1)

    if layer_end is not None:
        filtered_time = np.where(filtered_idx < layer_end, filtered_time, np.inf)
        filtered_idx = np.where(filtered_idx < layer_end, filtered_idx, -1)

    return sort_batch(Spike(filtered_time, filtered_idx))


def filter_spikes(spikes: Spike, layer_start: int, layer_end: Optional[int] = None):
    """Only return spikes of neurons after layer start

    Other spikes are encoded with time=np.inf and index=-1
    """
    filtered_time = np.where(spikes.idx >= layer_start, spikes.time, np.inf)
    filtered_idx = np.where(spikes.idx >= layer_start, spikes.idx, -1)

    if layer_end is not None:
        filtered_time = np.where(filtered_idx < layer_end, filtered_time, np.inf)
        filtered_idx = np.where(filtered_idx < layer_end, filtered_idx, -1)

    sort_idx = np.argsort(filtered_time, axis=-1)

    idx = filtered_idx[sort_idx]
    time = filtered_time[sort_idx]
    return Spike(time, idx)


def cut_spikes(spikes: Spike, count):
    return Spike(spikes.time[:count], spikes.idx[:count])


def cut_spikes_batch(spikes: Spike, count):
    return Spike(spikes.time[:, :count], spikes.idx[:, :count])


# sort spikes
def sort_batch(spikes: Spike) -> Spike:
    sort_idx = np.argsort(spikes.time, axis=-1)
    n_spikes = spikes.time.shape[0]
    time = spikes.time[np.arange(n_spikes)[:, None], sort_idx]
    idx = spikes.idx[np.arange(n_spikes)[:, None], sort_idx]
    return Spike(time=time, idx=idx)


def add_noise_batch(spikes: Spike, rng: random.PRNGKey, std: float = 5e-7) -> Spike:
    noise = random.normal(rng, spikes.time.shape) * std
    spikes_with_noise = Spike(time=spikes.time + noise, idx=spikes.idx)
    return sort_batch(spikes_with_noise)


def bump_weights(
    params: List[WeightInput], recording: List[Spike]
) -> List[WeightInput]:
    min_avg_spike = (0.3, 0.0)
    scalar_bump = 5e-3
    batch_size = recording[0].idx.shape[0]
    for i, (layer_recording, layer_params) in enumerate(zip(recording, params)):
        layer_size = layer_params.input.shape[1]
        spike_count = np.array(
            [
                np.sum(layer_recording.idx == neuron_ix) / batch_size
                for neuron_ix in range(layer_size)
            ]
        )
        bump = (spike_count < min_avg_spike[i]) * scalar_bump
        params[i] = WeightInput(layer_params.input + bump)
    return params


def clip_gradient(grad: List[WeightInput]) -> List[WeightInput]:
    for i in range(len(grad)):
        grad[i] = WeightInput(np.where(np.isnan(grad[i].input), 0.0, grad[i].input))
    return grad


def save_params(params: List[WeightInput], filenames: List[str]):
    # TODO this needs to work with pytrees
    for p, filename in zip(params, filenames):
        np.save(filename, p.input, allow_pickle=True)


def save_params_recurrent(params: WeightRecurrent, folder: str):
    np.save(f"{folder}/weights_input.npy", params.input, allow_pickle=True)
    np.save(f"{folder}/weights_recurrent.npy", params.recurrent, allow_pickle=True)


def load_params_recurrent(folder: str):
    return WeightRecurrent(
        input=np.load(
            f"{folder}/weights_input.npy",
        ),
        recurrent=np.load(
            f"{folder}/weights_recurrent.npy",
        ),
    )


def load_params(filenames) -> List[WeightInput]:
    return [WeightInput(np.load(f)) for f in filenames]


def get_index_trainset(trainset, idx):
    return (Spike(trainset[0].time[idx], trainset[0].idx[idx]), trainset[1][idx])
