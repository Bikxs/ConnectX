from kaggle_environments import make

import submission

if __name__ == '__main__':
    env = make("connectx", debug=True)
    env.render()
    my_agent = submission.act
    env.reset()
    # Play as the first agent against default "random" agent.
    # env.run([my_agent, "random"])
    env.run([my_agent, my_agent])
    env.render(mode="ipython", width=500, height=450)
