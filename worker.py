import numpy as np
import torch as T
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve
from wrappers import make_env


def worker(name, input_shape, n_actions, global_agent,
           optimizer, env_id, n_threads, global_idx, global_icm,
           icm_optimizer, icm):
    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)
    # local_agent.load_state_dict(global_agent.state_dict())
    if icm:
        local_icm = ICM(input_shape, n_actions)
        # local_icm.load_state_dict(global_icm.state_dict())
    else:
        local_icm = None
        intrinsic_reward = None

    memory = Memory()

    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape=frame_buffer)

    episode, max_steps, t_steps, scores = 0, 1e3, 0, []

    while episode < max_steps:
        obs,_ = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            if name == '0':
                env.render()
            state = T.tensor(obs[np.newaxis,:], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, terminated,truncated, info = env.step(action)
            done = terminated or truncated
            memory.remember(obs, action, obs_, reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1
            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_cost(obs, hx, done, rewards,
                                             values, log_probs,
                                             intrinsic_reward)
                optimizer.zero_grad()
                hx = hx.detach()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(
                                        local_agent.parameters(),
                                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                                        local_icm.parameters(),
                                        global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())
                
                if icm:
                    intrinsic_reward = intrinsic_reward.detach().numpy()
                    intrinsic_reward = intrinsic_reward[0]

                memory.clear_memory()
        episode += 1
        # with global_idx.get_lock():
        #    global_idx.value += 1
        if name == '0':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_score_5000 = np.mean(scores[max(0, episode-5000): episode+1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100) (5000) {:.2f} {:.2f} intrinsic reward {:.2f}'.format(
                                                episode, name, n_threads,
                                                t_steps/1e6, score,
                                                avg_score, avg_score_5000,intrinsic_reward))
        # if episode % 100 == 0:
            # T.save(local_agent.state_dict(),'checkpoints/saved_a3c_model.pth')
            # if icm:
            #     T.save(local_icm.state_dict(),'checkpoints/saved_icm_model.pth')
    if name == '0':
        x = [z for z in range(episode)]
        plot_learning_curve(x, scores, 'ICM_hallway_final.png')
