import click
import gym
import numpy as np
import time
import os
import pandas as pd

from pdqn import PDQNAgent
from pdqn_multipass import MultiPassPDQNAgent
from common import ClickPythonLiteralOption
from common.pess_domain import PESSFlattenedActionWrapper
# from common.wrappers import TimestepWrapper
from wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper, TimestepWrapper

from deg_nn.deg_api import MLPregression, zscores, model

from rl_plotter.logger import Logger

import torch.nn as nn
import torch.nn.functional as F

import gym_pess

# import __main__
# setattr(__main__, "MLPregression", MLPregression())



def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32) for i in range(93)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, save_dir, episodes=1000, save_states=False):
    returns = []
    states = []
    is_evaluate = True
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        total_reward = 0.
        while not terminal:
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
            if save_states:
                states.append((state[0]))
        returns.append(total_reward)

    if save_states:
        states = np.array(states)
        np.savetxt(save_dir + "\\" + "nodes_final_results.txt", states)
    return np.array(returns)


@click.command()
@click.option('--seed', default=2, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=100000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=50000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.0001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--learning-rate-actor-param', default=0.000001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=False, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--layers', default='[256,128,64]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-dir', default="results/pess", help='Output directory.', type=str)
@click.option('--title', default="PDDQN_day81_tuning", help="Prefix of output files", type=str)
@click.option('--save-states', default=True, help="Whether save the state when evaluating", type=bool)
@click.option('--is-evaluate', default=False, help="Whether is evaluating", type=bool)

def run_pess_mpdqn(seed, episodes, batch_size, learning_rate_actor, learning_rate_actor_param,
                   epsilon_steps, gamma, tau_actor, tau_actor_param, clip_grad, indexed, weighted,
                   average, random_weighted, initial_memory_threshold, use_ornstein_noise, replay_memory_size,
                   epsilon_final, inverting_gradients, layers, scale_actions, action_input_layer, zero_index_gradients,
                   initialise_params, save_dir, title, evaluation_episodes, save_states, is_evaluate):

    time_str = time.strftime("%Y%m%d%H%M", time.localtime())

    if save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)), time_str)
        os.makedirs(save_dir, exist_ok=True)

    episode_number = 10
    env = gym.make('PESS-v0')

    # env = ScaledStateWrapper(env)
    env = PESSFlattenedActionWrapper(env, is_evaluate, episode_number)

    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    # env = TimestepWrapper(env)

    dir = os.path.join(save_dir,title)

    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    agent_class = MultiPassPDQNAgent

    agent = agent_class(
        env.observation_space.spaces[0], env.action_space,
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,
        indexed=indexed,
        weighted=weighted,
        average=average,
        random_weighted=random_weighted,
        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        zero_index_gradients=zero_index_gradients,
        seed=seed)

    if initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)

        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

    print(agent)
    MAX_TIME_STEPS = 96 * 7
    total_reward = 0.
    returns = []
    start_time = time.time()
    returns_r100 = []

    logger = Logger(exp_name="your_exp_name", env_name='PESS-v0')

    for i in range(episodes):

        if save_dir and i % 100 == 0:
            agent.save_models(os.path.join(save_dir, str(i)))

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()


        for j in range(MAX_TIME_STEPS):

            # env.episode_number = i
            env.set_episode_value(i)

            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret

            if terminal:
                break

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)

            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            # print("state: ", state)

            episode_reward += reward

            # logger.update(score=[episode_reward], total_steps=i)



        agent.end_episode()

        returns.append(episode_reward)

        # return_rewards = pd.DataFrame(data=returns,columns=['rewards'])
        # return_rewards.to_csv(save_dir + '/' + 'all_rewards.csv', index=True)
        np.savetxt(save_dir + '/' + "all_rewards.txt", np.array(returns))


        total_reward += episode_reward
        if i % 100 == 0:
            print("state: ", state[0], state[3])
            print("reward: ", '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1),
                                                                     np.array(returns[-100:]).mean()))
            returns_r100.append(np.array(returns[-100:]).mean())
            np.savetxt(save_dir + '/' + str(learning_rate_actor) +"_" + str(learning_rate_actor_param) + "all_rewards_r100.txt", returns_r100)
            print("episode reward: ", '{0:.4f}'.format(episode_reward))

        env.reset_episode_step()

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()

    # returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.load_models(os.path.join(save_dir + "\\"))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, save_dir, evaluation_episodes, save_states)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        # np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_pess_mpdqn()

