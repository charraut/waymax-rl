# Code source: https://github.com/google/brax/blob/main/brax/training/replay_buffers.py


import math
from collections.abc import Sequence

import flax
import jax
import jax.numpy as jnp
from jax import flatten_util
from jax.experimental import pjit

from waymax_rl.utils import PRNGKey, Transition


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
    insert_position: int
    sample_position: int
    key: PRNGKey


class ReplayBuffer:
    """Contains replay buffer methods."""

    def init(self, key: PRNGKey) -> ReplayBufferState:
        """Init the replay buffer."""
        raise NotImplementedError("This method should be implemented by derived classes.")

    def insert(self, buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
        """Insert data into the replay buffer."""
        raise NotImplementedError("This method should be implemented by derived classes.")

    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
        """Transition a batch of data."""
        raise NotImplementedError("This method should be implemented by derived classes.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        """Total amount of elements that are sampleable."""
        raise NotImplementedError("This method should be implemented by derived classes.")


class UniformSamplingQueue(ReplayBuffer):
    """Base class for limited-size FIFO reply buffers."""

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        dummy_data_sample: Transition,
    ):
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        data_size = len(dummy_flatten)

        # _buffer_size = int(buffer_size + buffer_size % num_envs)

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

    def insert(self, buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Transition to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})",
            )

        update = self._flatten_fn(samples)

        data = buffer_state.data
        insert_idx = buffer_state.insert_position

        # Update the buffer and the control numbers
        data = jax.lax.dynamic_update_slice_in_dim(data, update, insert_idx, axis=0)
        insert_idx = (insert_idx + len(update)) % self._size
        sample_idx = jnp.minimum(buffer_state.sample_position + len(update), self._size)

        return buffer_state.replace(
            data=data,
            insert_position=insert_idx,
            sample_position=sample_idx,
        )

    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
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

        batch = jnp.take(buffer_state.data, idx, axis=0, mode="wrap")

        return buffer_state.replace(key=key), self._unflatten_fn(batch)

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.sample_position


class PmapWrapper(ReplayBuffer):
    """Wrapper to distribute the buffer on multiple devices.

    Each device stores a replay buffer 'buffer' such that no data moves from one
    device to another.
    The total capacity of this replay buffer is the number of devices multiplied
    by the size of the wrapped buffer.
    The sample size is also the number of devices multiplied by the size of the
    wrapped buffer.
    This should not be used inside a pmapped function:
    You should just use the regular replay buffer in that case.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        local_device_count: int | None = None,
    ):
        self._buffer = buffer
        self._num_devices = local_device_count or jax.local_device_count()

    def init(self, key: PRNGKey) -> ReplayBufferState:
        key = jax.random.fold_in(key, jax.process_index())
        keys = jax.random.split(key, self._num_devices)

        return jax.pmap(self._buffer.init)(keys)

    # NB: In multi-hosts setups, every host is expected to give a different batch
    def insert(self, buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
        self._buffer.check_can_insert(buffer_state, samples, self._num_devices)
        samples = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, self._num_devices) + x.shape[1:]), samples)

        # This is to enforce we're gonna iterate on the start of the batch before the end of the batch
        samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)

        return jax.pmap(self._buffer.insert_internal)(buffer_state, samples)

    # NB: In multi-hosts setups, every host will get a different batch
    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
        self._buffer.check_can_sample(buffer_state, self._num_devices)
        buffer_state, samples = jax.pmap(self._buffer.sample_internal)(buffer_state)

        samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
        samples = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), samples)

        return buffer_state, samples

    def size(self, buffer_state: ReplayBufferState) -> int:
        axis_name = "x"

        def psize(buffer_state):
            return jax.lax.psum(self._buffer.size(buffer_state), axis_name=axis_name)

        return jax.pmap(psize, axis_name=axis_name)(buffer_state)[0]


class PjitWrapper(ReplayBuffer):
    """Wrapper to distribute the buffer on multiple devices with pjit.

    Each device stores a part of the replay buffer depending on its index on axis
    'axis_name'.
    The total capacity of this replay buffer is the size of the mesh multiplied
    by the size of the wrapped buffer.
    The sample size is also the size of the mesh multiplied by the size of the
    sample in the wrapped buffer. Transition batches from each shard are concatenated
    (i.e. for random sampling, each shard will sample from the data they can see).
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        mesh: jax.sharding.Mesh,
        axis_names: Sequence[str],
    ):
        """Constructor.

        Args:
          buffer: The buffer to replicate.
          mesh: Device mesh for pjitting context.
          axis_names: The axes along which the replay buffer data should be
            partitionned.
        """
        self._buffer = buffer
        self._mesh = mesh
        self._num_devices = math.prod(mesh.shape[name] for name in axis_names)

        def init(key: PRNGKey) -> ReplayBufferState:
            keys = jax.random.split(key, self._num_devices)

            return jax.vmap(self._buffer.init)(keys)

        def insert(buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
            samples = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self._num_devices) + x.shape[1:]),
                samples,
            )
            # This is to enforce we're gonna iterate on the start of the batch before the end of the batch
            samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)

            return jax.vmap(self._buffer.insert_internal)(buffer_state, samples)

        def sample(buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
            buffer_state, samples = jax.vmap(self._buffer.sample_internal)(buffer_state)
            samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
            samples = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), samples)

            return buffer_state, samples

        def size(buffer_state: ReplayBufferState) -> int:
            return jnp.sum(jax.vmap(self._buffer.size)(buffer_state))

        partition_spec = jax.sharding.PartitionSpec(
            (axis_names),
        )
        self._partitioned_init = pjit.pjit(init, out_shardings=partition_spec)
        self._partitioned_insert = pjit.pjit(
            insert,
            out_shardings=partition_spec,
        )
        self._partitioned_sample = pjit.pjit(
            sample,
            out_shardings=partition_spec,
        )
        # This will return the TOTAL size across all devices.
        self._partitioned_size = pjit.pjit(size, out_shardings=None)

    def init(self, key: PRNGKey) -> ReplayBufferState:
        """See base class."""
        with self._mesh:
            return self._partitioned_init(key)

    def insert(self, buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
        """See base class."""
        self._buffer.check_can_insert(buffer_state, samples, self._num_devices)
        with self._mesh:
            return self._partitioned_insert(buffer_state, samples)

    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, Transition]:
        """See base class."""
        self._buffer.check_can_sample(buffer_state, self._num_devices)
        with self._mesh:
            return self._partitioned_sample(buffer_state)

    def size(self, buffer_state: ReplayBufferState) -> int:
        """See base class. The total size (sum of all partitions) is returned."""
        with self._mesh:
            return self._partitioned_size(buffer_state)
