import multiprocessing as mp
import time

from joblib import Parallel, delayed
from kaggle_environments import evaluate

import submission_minimax


def get_win_percentages(agent_dict_1, agent_dict_2, n_rounds=16):
    agent1, agent2 = agent_dict_1['agent'], agent_dict_2['agent']
    agent1_name, agent2_name = agent_dict_1['name'], agent_dict_2['name']
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    cores = mp.cpu_count()
    half_rounds_per_core = n_rounds // cores // 2

    def evaluate_per_core(half_rounds):
        outcomes = evaluate("connectx", [agent1, agent2], config, [], half_rounds)
        outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1], config, [], half_rounds)]
        return outcomes

    start_time = time.time()
    results = Parallel(n_jobs=mp.cpu_count())(delayed(evaluate_per_core)(half_rounds_per_core) for i in range(cores))
    total_time = time.time() - start_time

    outcomes = [result for results_per_job in results for result in results_per_job]
    print()
    print("******************************")
    print("In total, {} episodes have been evaluated using {} CPU's cores.".format(len(outcomes), cores))
    print("Total time: {:.2f} minutes ({:.2f} seconds per match on average)".format(total_time / 60,
                                                                                    total_time / n_rounds))
    print(f"{agent1_name} Won: {outcomes.count([1, -1]) / len(outcomes):.2%}")
    print(f"{agent2_name} Won: {outcomes.count([-1, 1]) / len(outcomes):.2%}")
    print(f"Ties:        {outcomes.count([0, 0]) / len(outcomes):.2%}")
    print(f"Invalid Plays by {agent1_name}:", outcomes.count([0, None]))
    print(f"Invalid Plays by {agent2_name}:", outcomes.count([None, 0]))


if __name__ == '__main__':
    agents = [{'name': 'AlphaBetaWithTimeout', 'agent': submission_minimax.agent_alpha_beta_timeout},
              {'name': 'Random', 'agent': submission_minimax.agent_random},
              {'name': 'AlphaBetaWithDepth5', 'agent': submission_minimax.agent_alpha_beta_depth_5}]
    for agent in agents:
        get_win_percentages(agent, {'name': 'random', 'agent': 'random'})
        get_win_percentages(agent, {'name': 'negmax', 'agent': 'negmax'})
        print("----------------------------------------------------")
        print()
