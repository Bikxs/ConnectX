from strategy_minimax import agent_alpha_beta_depth_5


def act(observation, configuration):
    # return agent_alpha_beta_timeout(observation=observation, configuration=configuration)
    print(observation)
    return agent_alpha_beta_depth_5(observation=observation, configuration=configuration)