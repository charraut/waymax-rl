import flax
import jax
import jax.numpy as jnp
from jax import flatten_util

from waymax_rl.utils import PRNGKey, Transition


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jax.Array
    insert_position: int
    sample_position: int
    key: PRNGKey


class ReplayBuffer:
    """Replay buffer."""

    def __init__(self, buffer_size: int, batch_size: int, dummy_data_sample: Transition):
        """Initialize the replay buffer."""
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        data_size = len(dummy_flatten)

        self._data_shape = (buffer_size, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._batch_size = batch_size
        self._size = buffer_size

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            insert_position=0,
            sample_position=0,
            key=key,
        )

    def insert(self, buffer_state: ReplayBufferState, samples: Transition, mask: jax.Array) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
            buffer_state: Buffer state
            samples: Transition to insert with a leading batch size.
            mask: Mask of the samples to insert.

        Returns:
            New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})",
            )

        # # Flatten the samples
        # _samples = self._flatten_fn(samples)

        # # Padded indices of the mask elements
        # samples_size = jnp.sum(mask)
        # mask_indices = jnp.where(mask, size=len(mask), fill_value=len(mask))

        # # Current buffer state
        # data = buffer_state.data
        # insert_idx = buffer_state.insert_position
        # sample_idx = buffer_state.sample_position

        # # Create a copy of the buffer with samples inserted at insert_idx
        # data_indices = (insert_idx + jnp.arange(len(mask))) % self._size
        # update_mask = jnp.arange(len(mask))[:, None] < samples_size

        # # Update the buffer state
        # data = data.at[data_indices].set(jnp.where(update_mask, _samples[mask_indices], data[data_indices]))
        # insert_idx = (insert_idx + samples_size) % self._size
        # sample_idx = jnp.minimum(sample_idx + samples_size, self._size)

        # Flatten the samples
        new_samples = self._flatten_fn(samples)
        samples_size = len(new_samples)

        # Current buffer state
        data = buffer_state.data
        insert_idx = buffer_state.insert_position

        # Update the buffer and the control numbers
        data = jax.lax.dynamic_update_slice_in_dim(data, new_samples, insert_idx, axis=0)
        insert_idx = (insert_idx + samples_size) % self._size
        sample_idx = jnp.minimum(buffer_state.sample_position + samples_size, self._size)

        return buffer_state.replace(
            data=data,
            insert_position=insert_idx,
            sample_position=sample_idx,
        )

    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
        """Sample a batch of data from the replay buffer.

        Args:
            buffer_state: Buffer state

        Returns:
            New buffer state and a random batch of data.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})",
            )

        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._batch_size,),
            minval=0,
            maxval=buffer_state.sample_position,
        )

        batch = jnp.take(buffer_state.data, idx, axis=0, unique_indices=True)

        return buffer_state.replace(key=key), self._unflatten_fn(batch)
