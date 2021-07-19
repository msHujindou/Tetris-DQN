from os import stat
import numpy as np

cost_for_each_step = 1
stop_condition = -1
not_permit_reward = 0


class game:
    def __init__(self) -> None:
        self.game_state = np.array([[0, 1, 2], [-1, 0, -100], [5, 0, 2]])
        self.current_reward = 1
        self.current_position = (0, 0)
        self.current_state = 0

    def reset(self):
        self.game_state = np.array([[0, 1, 2], [-1, 0, -100], [5, 0, 2]])
        self.current_reward = 1
        self.current_state = 0
        self.current_position = (0, 0)
        return self.current_state

    def select_random_step(self):
        action_index = int(np.random.randint(0, 4))
        return action_index

    def step(self, action):
        # 0 -> left
        # 1 -> top
        # 2 -> right
        # 3 -> bottom
        if action == 0:
            if self.current_position[1] <= 0:
                return self.current_state, not_permit_reward, False, "Left bound reach"
            self.current_position = (
                self.current_position[0],
                self.current_position[1] - 1,
            )
            self.current_reward -= cost_for_each_step
            self.current_state = 3 * self.current_position[0] + self.current_position[1]
            reward = self.game_state[self.current_position]
            self.current_reward += reward
            # self.game_state[self.current_position] = 0
            if self.current_reward < stop_condition:
                return (
                    self.current_state,
                    reward,
                    True,
                    f"self.current_reward is {self.current_reward}",
                )
            return self.current_state, reward, False, None
        elif action == 1:
            if self.current_position[0] <= 0:
                return self.current_state, not_permit_reward, False, "Top bound reach"
            self.current_position = (
                self.current_position[0] - 1,
                self.current_position[1],
            )
            self.current_reward -= cost_for_each_step
            self.current_state = 3 * self.current_position[0] + self.current_position[1]
            reward = self.game_state[self.current_position]
            self.current_reward += reward
            # self.game_state[self.current_position] = 0
            if self.current_reward < stop_condition:
                return (
                    self.current_state,
                    reward,
                    True,
                    f"self.current_reward is {self.current_reward}",
                )
            return self.current_state, reward, False, None
        elif action == 2:
            if self.current_position[1] >= 2:
                return self.current_state, not_permit_reward, False, "Right bound reach"
            self.current_position = (
                self.current_position[0],
                self.current_position[1] + 1,
            )
            self.current_reward -= cost_for_each_step
            self.current_state = 3 * self.current_position[0] + self.current_position[1]
            reward = self.game_state[self.current_position]
            self.current_reward += reward
            # self.game_state[self.current_position] = 0
            if self.current_reward < stop_condition:
                return (
                    self.current_state,
                    reward,
                    True,
                    f"self.current_reward is {self.current_reward}",
                )
            return self.current_state, reward, False, None
        else:
            if self.current_position[0] >= 2:
                return (
                    self.current_state,
                    not_permit_reward,
                    False,
                    "Bottom bound reach",
                )
            self.current_position = (
                self.current_position[0] + 1,
                self.current_position[1],
            )
            self.current_reward -= cost_for_each_step
            self.current_state = 3 * self.current_position[0] + self.current_position[1]
            reward = self.game_state[self.current_position]
            self.current_reward += reward
            # self.game_state[self.current_position] = 0
            if self.current_reward < stop_condition:
                return (
                    self.current_state,
                    reward,
                    True,
                    f"self.current_reward is {self.current_reward}",
                )
            return self.current_state, reward, False, None


# env = gym.make("FrozenLake-v0", is_slippery=False)
# env.reset()
# env.render()
# new_state, reward, done, _ = env.step(2)
# print(f"[{new_state}],[{reward}],[{done}],[{_}]")
# new_state, reward, done, _ = env.step(3)
# print(f"[{new_state}],[{reward}],[{done}],[{_}]")
# new_state, reward, done, _ = env.step(3)
# print(f"[{new_state}],[{reward}],[{done}],[{_}]")
# env.render()
# env.close()

total_episode = 80000


class FrozenLake:
    """
    Q-Learning with Frozen Lake
    Parameters:
    -----------
    env_name: str
        gym environment name
    lr: float
        Learning rate for agent
    min_eps: float
        Minimum exploration probability
    max_eps: float
        Max exploration probability
    gamma: float
        Rate for discounting future rewards
    decay_rate: float
        Decay rate for exploration probability
    """

    def __init__(
        self,
        env_name="FrozenLake-v0",
        lr=0.8,
        min_eps=0.001,
        max_eps=1.0,
        gamma=0.95,
        decay_rate=0.005,
    ):
        self.lr = lr
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.decay_rate = -np.log((0.5 - min_eps) / (max_eps - min_eps)) / total_episode

        self.init_env(env_name)

    def init_env(self, env):
        """
        Initializes the gym environment
        """
        self.env = game()

        self.action_space = 4
        self.state_size = 9

        self.init_qtable()

    def init_qtable(self):
        """
        Creates the Q table to hold the states, and actions
        """
        self.qtable = np.zeros((self.state_size, self.action_space))

    def train(self, episodes=total_episode, steps=99, epsilon=1.0):
        """
        Selects actions for the agent to take in given
        states and updates the Q table
        Parameters
        ----------
        episodes: int
            Total number of episodes
        steps: int
            Number of steps
        epsilon: float, default=1.0
            Exploration probability
        """

        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(steps):
                explore_exploit_tradeoff = np.random.uniform()

                if explore_exploit_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state, :])
                    # print("qtable selected")
                else:
                    action = self.env.select_random_step()
                    # print("random selected")

                new_state, reward, done, _ = self.env.step(action)
                # print(f"[{action}:{state}->{new_state}],{reward},{done},[{_}]")

                # if action == 0 and state == 0 and reward > 0:
                #     print(reward, new_state, bf, self.env.current_position)

                if done:
                    break

                # if new_state > 5:
                #     print(f"[{new_state}],[{reward}],[{done}],[{_}]")

                # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # if new_state != state:
                self.qtable[state, action] = self.qtable[state, action] + self.lr * (
                    reward
                    + self.gamma * np.amax(self.qtable[new_state, :])
                    - self.qtable[state, action]
                )

                total_reward += reward
                state = new_state

            epsilon = self.min_eps + (self.max_eps - self.min_eps) * np.exp(
                -self.decay_rate * episode
            )
            rewards.append(total_reward)

        print("\nScore: ", np.sum(rewards) / episodes)

    def play(self, episodes=10, steps=99):
        """
        Plays FrozenLake with the trained agent
        """
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            disp = "\n\n===========================" + f"{episode}"

            print(disp)

            for step in range(steps):
                action = np.argmax(self.qtable[state, :])

                new_state, reward, done, _ = self.env.step(action)

                if done:
                    rewards.append(reward)
                    print(f"step: {step}\t", f"reward: {total_reward}")
                    break

                total_reward += reward

                state = new_state

        print("Reward", np.sum(rewards) / episodes)


fl = FrozenLake()
fl.train()
print(fl.qtable)
