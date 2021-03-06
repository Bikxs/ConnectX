import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_agents.specs  as array_spec
from kaggle_environments import make
from tf_agents import networks
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

AGENTS_DIR = 'agents'

A = 1000000.0
B = 100.0
C = 1.0
D = -100.0
E = -1000.0
F = -1000000.0


def get_heuristic(grid, mark, config):
    num_twos = count_windows(grid, 2, mark, config)
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
    num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
    score = A * num_fours + B * num_threes + C * num_twos + D * num_twos_opp + E * num_threes_opp + num_fours_opp * F
    return num_fours > 0, score


# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)


# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows


# Calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)
    score = get_heuristic(next_grid, col, mark, config)
    return score


# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


def alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the minimax algorithm.
    Args:
        mark (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """
    moves = game.moves_available()  # game.get_player_moves(active_player)
    if not moves:
        return None, 0
        # raise Exception("No possible move left")
    random.shuffle(moves)
    new_depth = depth - 1
    best_move = random.choice(moves)
    if my_turn:
        value = -float('inf')
        for column_index in moves:
            won, _score, new_game = game.forecast_move(column_index)
            if won:
                return column_index, _score
            else:
                if new_depth == 0:
                    score = _score
                else:
                    _, score = alphabeta(new_game, new_depth, my_turn=False)
                if score > value:
                    best_move = column_index
                    value = score
                # beta pruning
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return best_move, value
    else:
        # opponent's turn
        value = +float('inf')
        for column_index in moves:
            won, _score, new_game = game.forecast_move(column_index)
            if won:
                return column_index, -_score
            else:
                if new_depth == 0:
                    score = -_score
                else:
                    _, score = alphabeta(new_game, new_depth, my_turn=True)
                if score < value:
                    best_move = column_index
                    value = score
                # alpha prunning
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return best_move, value
    return best_move, val


class Game:
    def __init__(self, board, mark, config):
        self.board = board
        self.config = config
        self.mark = mark

    def moves_available(self):
        top_row = self.board[0, :]
        return [column_index for column_index, value in enumerate(top_row) if value == 0]

    def forecast_move(self, col):
        board = drop_piece(self.board, col=col, mark=self.mark, config=self.config)
        if not col in self.moves_available():
            return True, 2 * F, Game(board, config=self.config, mark=self.mark)

        won, score = get_heuristic(board, self.mark, config=self.config)
        mark = 1 if self.mark == 2 else 2
        return won, score, Game(board, config=self.config, mark=mark)

    def score(self, mark):
        won, score = get_heuristic(self.board, mark=mark, config=self.config)
        return score

    @classmethod
    def fromEnv(cls, observation, configuration):
        num_columns = configuration['columns']
        num_rows = configuration['rows']
        mark = observation['mark']
        board = np.array(observation['board'])
        board = np.array(board).reshape((num_rows, num_columns))
        return Game(board=board, config=configuration, mark=mark)

    def is_won(self, mark):
        won, score = get_heuristic(self.board, mark=mark, config=self.config)
        return won


class ConnectX(py_environment.PyEnvironment):
    def __init__(self, switch_prob=0.5, agent2="negamax"):
        self.env = make('connectx', debug=False)
        self.pair = [None, agent2]
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob

        # Define required gym fields (examples):
        self.config = self.env.configuration

        self.columns = self.config.columns
        self.rows = self.config.rows
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=int(self.columns - 1), name='play')
        self._observation_spec = {
            'board': array_spec.BoundedArraySpec(shape=(self.rows, self.columns, 1), dtype=np.float32, minimum=-1,
                                                 maximum=1, name='board'),
            'mark': array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=-1, maximum=1, name='mark')}

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def extract_obs(self, obs):
        return {'board': np.array(obs['board'], dtype=np.float32).reshape((self.rows, self.columns, 1)),
                'mark': np.array(obs['mark'], dtype=np.float32)}

    def _step(self, action):
        action = int(action)

        _obs, old_reward, done, _ = self.trainer.step(action)
        game = Game.fromEnv(_obs, self.config)

        # best_move, value = alphabeta(game, depth=5, my_turn=False)
        # reward = value
        _, reward = get_heuristic(np.array(_obs['board']).copy().reshape((self.rows, self.columns)), _obs['mark'],
                                  self.config)
        
        self.obs = self.extract_obs(_obs)
        if done:
            if old_reward is None:
                # make an illegal move
                reward = 2 * F

            self.trainer.reset()
            return ts.termination(self.obs, reward=reward)
        else:
            return ts.transition(self.obs, reward=reward, discount=1.0)

    def _reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        obs = self.trainer.reset()
        self.obs = self.extract_obs(obs)
        return ts.restart(self.obs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)


def compute_stats(environment, policy, num_episodes=100):
    returns = []
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        # print(time_step)
        returns.append(time_step.reward)
    return np.mean(returns), np.std(returns)


def train(agent_name, vs_agent, description, checkpoint_to_load=None, num_iterations=20000, learning_rate=1e-4):
    agent_folder = f'{AGENTS_DIR}/{agent_name}'
    if os.path.exists(agent_folder):
        print(f"Agent's: {agent_name} folder exists - skipping to train it")
        return

    print(f'Print tensorflow version:{tf.version.VERSION}')
    print(f'\nTraining {agent_name} vs {vs_agent}')

    def tf_env():
        return tf_py_environment.TFPyEnvironment(ConnectX(agent2=vs_agent))

    # Hyper parameters

    # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    # @param {type:"number"}
    log_interval = 100  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    # validate_py_environment(ConnectX(), episodes=5)
    # spec
    train_env = tf_env()
    eval_env = tf_env()

    preprocessing_layers = {
        # 'board': tf.keras.layers.Flatten(),
        'board': tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, 4), tf.keras.layers.Flatten()]),
        'mark': tf.keras.layers.Flatten()
    }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    fc_layer_params = [1024, 1024, 1024, 512]
    dropout_layer_params = [None, .2, .15, .1]
    q_net = networks.q_network.QNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=None,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=None, #dropout_layer_params,
        activation_fn=tf.keras.activations.relu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',distribution='truncated_normal'),
        batch_squash=True,
        dtype=tf.float32,
        name='QNetwork'
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(time_step_spec=
                               train_env.time_step_spec(),
                               action_spec=train_env.action_spec(),
                               q_network=q_net,
                               optimizer=optimizer,
                               td_errors_loss_fn=common.element_wise_squared_loss,
                               train_step_counter=train_step_counter)

    agent.initialize()
    print("Neural network:")
    print(q_net.summary())
    print()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    def collect_data(environment, policy, buffer, steps):
        def collect_step():
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            buffer.add_batch(traj)

        for _ in range(steps):
            collect_step()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    # print(f'dataset: {dataset}')
    iterator = iter(dataset)

    # Reset the train step
    agent.train_step_counter.assign(0)

    def load_checkpoint_fn():
        load_agent_folder = f'{AGENTS_DIR}/{checkpoint_to_load}'
        load_checkpoint_dir = f'{load_agent_folder}/checkpoint'
        load_checkpointer = common.Checkpointer(
            ckpt_dir=load_checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter
        )
        load_checkpointer.initialize_or_restore()
        print(f'Loaded checkpoint {checkpoint_to_load}')

    def save_checkpoint_and_policy():

        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter
        )
        train_checkpointer.save(step)

    if checkpoint_to_load is not None:
        load_checkpoint_fn()
        step = tf.compat.v1.train.get_global_step()

    rewards = []
    start_time = datetime.now()
    start_step = agent.train_step_counter.numpy()

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0 or step == 1:
            print(f'step#{step} loss: {train_loss:,.5f}')

        if step % eval_interval == 0:
            avg_return, std_return = compute_stats(eval_env, agent.policy, num_eval_episodes)
            print(f'\n**********************************************************************************************')
            print(f'** step#{step} AVG Reward: {avg_return:,.5f}, Std Reward: {std_return:,.5f}')
            print(f'**********************************************************************************************\n')
            rewards.append({'step': int(step), 'avg_return': avg_return, 'std_return': std_return})

    os.makedirs(agent_folder)
    policy_dir = f'{agent_folder}/policy'
    checkpoint_dir = f'{agent_folder}/checkpoint'
    end_step = agent.train_step_counter.numpy()
    save_checkpoint_and_policy()
    info_file = open(f"{agent_folder}/info.txt", "w+")
    info_file.write(f"name: {agent_name}\n")
    info_file.write(f"description: {description}\n")
    info_file.write(f"loaded from checkpoint: {'None' if checkpoint_to_load is None else checkpoint_to_load}\n")
    info_file.write(f"training started: {start_time}\n")
    info_file.write(f"training completed: {datetime.now()}\n")
    info_file.write(f"start step number: {start_step}\n")
    info_file.write(f"end step number: {end_step}\n")

    info_file.close()

    df = pd.DataFrame(rewards)
    df.to_csv(f"{agent_folder}/rewards.csv")


def load_pocicy(policy_name):
    policy_dir = os.path.join(AGENTS_DIR, f'{policy_name}_{int(datetime.now().timestamp())}.policy')
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
    return saved_policy


def agent_from_policy(policy_name):
    # make agent and return
    pass


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    devices = device_lib.list_local_devices()
    print(devices)

    train("dqn_base_v1",
          vs_agent="random",
          num_iterations=50000,
          checkpoint_to_load=None,
          description="DQN agent trained against a random agent from scratch")
    train("dqn_base_v2",
          vs_agent="random",
          num_iterations=50000,
          learning_rate=1e-5,
          checkpoint_to_load="dqn_base_v1",
          description="DQN agent trained against a random agent from initialized from dqn_base_v1")
    train("dqn_v_negamax_v1",
          vs_agent="negamax",
          num_iterations=100000,
          learning_rate=1e-5,
          checkpoint_to_load="dqn_base_v2",
          description="DQN agent trained against negamax agent initialized initialized from dqn_base_v2")
    train("dqn_v_negamax_v2",
          vs_agent="negamax",
          num_iterations=100000,
          learning_rate=1e-5,
          checkpoint_to_load="dqn_v_negamax_v1",
          description="DQN agent trained against negamax agent initialized initialized from dqn_v_negamax_v2")
    train("dqn_v_negamax_v3",
          vs_agent="negamax",
          num_iterations=300000,
          learning_rate=1e-6,
          checkpoint_to_load="dqn_v_negamax_v2",
          description="DQN agent trained against negamax agent initialized initialized from dqn_v_negamax_v2")
