import jax
from waymax import datatypes
from waymax.agents.actor_core import ActorState, Params, WaymaxActorCore, WaymaxActorOutput


class Agent(WaymaxActorCore):
    def __init__(self, env, config, model, optimizer, logger, rng):
        super().__init__(env, config, model, optimizer, logger, rng)

    def init(self, params: Params, rng: jax.Array) -> ActorState:
        return super().init(params, rng)

    def select_action(
        self,
        params: Params,
        state: datatypes.SimulatorState,
        actor_state: ActorState,
        rng: jax.Array,
    ) -> WaymaxActorOutput:
        return super().select_action(params, state, actor_state, rng)

    def learn(self, env_state, action, reward, next_env_state, done):
        pass
