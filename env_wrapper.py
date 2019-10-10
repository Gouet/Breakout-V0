import gym

class EnvWrapper:
    def __init__(self, gym_env, actors, update_obs=None):
        self.envs = []
        self.update_obs = update_obs
        for _ in range(actors):
            self.envs.append(gym.make(gym_env))

    def step(self, actions):
        batch_states = []
        batch_rewards = []
        batch_dones = []
        for i, action in enumerate(actions):
            states, rewards, done_, _ = self.envs[i].step(action)
            if self.update_obs is not None:
                states = self.update_obs(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_dones.append(done_)
        self.dones = batch_dones
        return batch_states, batch_rewards, batch_dones

    def render(self, id):
        self.envs[id].render()

    def done(self):
        return all(self.dones)

    def reset(self):
        batch_states = []
        self.dones = []
        print('RESET')
        for env in self.envs:
            obs = env.reset()
            self.dones.append(False)
            if self.update_obs is not None:
                obs = self.update_obs(obs)
            batch_states.append(obs)
        return batch_states