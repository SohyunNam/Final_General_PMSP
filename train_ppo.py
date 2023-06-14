import os
import vessl
import torch
import json

from cfg import get_cfg
from agent.ppo import *
from environment.env import PMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.use_vessl:
        vessl.init(organization="sun-eng-dgx", project="Final-General-PMSP", hp=cfg)

    num_job = cfg.num_job
    num_m = cfg.num_machine
    weight_tard = cfg.weight_tard
    weight_setup = cfg.weight_setup

    num_episode = cfg.n_episode

    model_dir = './output/model/' if not cfg.use_vessl else '/output/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    simulation_dir = './output/simulation/' if not cfg.use_vessl else '/output/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    log_dir = './output/log/' if not cfg.use_vessl else '/output/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup])
    agent = PPO(cfg, env.state_dim, env.action_dim).to(device)

    if cfg.load_model:
        checkpoint = torch.load('./trained_model/episode-30000.pt')
        start_episode = checkpoint['episode'] + 1
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, reward_tard, reward_setup, tardiness, setup, makespan\n')

    for episode in range(start_episode, start_episode + num_episode):
        state = env.reset()
        r_epi = 0.0
        done = False

        while not done:
            for t in range(cfg.T_horizon):
                logit = agent.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done = env.step(action)

                agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    tardiness = env.monitor.tardiness / env.num_job
                    setup = env.monitor.setup / env.num_job
                    makespan = env.sink.makespan
                    print("episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f" % (
                        episode, r_epi, setup, tardiness, makespan))
                    with open(log_dir + "train_log.csv", 'a') as f:
                        f.write('%d,%.2f,%.2f,%.2f, %.4f, %.4f,%.4f \n' % (
                        episode, r_epi, env.reward_tard, env.reward_setup, tardiness, setup, makespan))
                    break
            agent.train_net()

        if episode % 100 == 0 or episode == 1:
            _ = env.get_logs(simulation_dir + "log_{0}.csv".format(episode))
            agent.save(episode, model_dir)
            if cfg.use_vessl:
                tardiness = env.monitor.tardiness / env.num_job
                setup = env.monitor.setup / env.num_job
                makespan = env.sink.makespan
                vessl.log(step=episode, payload={'reward': r_epi})
                vessl.log(step=episode, payload={'reward_setup': env.reward_setup})
                vessl.log(step=episode, payload={'reward_tard': env.reward_tard})
                vessl.log(step=episode, payload={'Train_Tardiness': tardiness})
                vessl.log(step=episode, payload={'Train_Setup': setup})
                vessl.log(step=episode, payload={'Train_Makespan': makespan})