import torch
import torch.nn.functional as F
import torch.optim as optim
import psutil
import logging
from collections import deque

from utils import *
from model import ActorCritic
from envs import *


def train(pid, rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    filepath = "./train_model_" + str(rank)
    env = gym.wrappers.Monitor(env, filepath, force=True)
    env.seed(args.seed + rank)

    # Setup logger for this thread
    log = {}
    setup_logger(f'{args.env_name}_train_log_{rank}', args.log_dir)
    logger = logging.getLogger(f'{args.env_name}_train_log_{rank}')
    
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    model.train()
    
    # Tracking variables
    episode_count = 0
    episode_rewards = []
    recent_scores = deque(maxlen=100)  # Track last 100 episodes
    best_score = 0
    
    obs = env.reset()
    state = torch.from_numpy(obs)
    done = True
    
    while True:
        # Process status check
        pps = psutil.Process(pid=pid)
        try:
            if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                break
        except psutil.NoSuchProcess:
            break
            
        # Sync with shared model
        model.load_state_dict(shared_model.state_dict())
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        if done:
            cx = torch.zeros(1, 512)
            hx = torch.zeros(1, 512)
            current_episode_reward = 0
        else:
            cx = cx.detach()
            hx = hx.detach()

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            
            obs, reward, done, _ = env.step(action.item())
            current_episode_reward += reward
            
            with lock:
                counter.value += 1
                
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            
            state = torch.from_numpy(obs)
            
            if done:
                episode_count += 1
                episode_rewards.append(current_episode_reward)
                recent_scores.append(current_episode_reward)
                best_score = max(best_score, current_episode_reward)
                
                # Calculate metrics
                avg_score = np.mean(recent_scores) if recent_scores else 0
                avgmaxp = np.mean([max(episode_rewards[i:i+100]) for i in range(0, len(episode_rewards), 100)]) if episode_rewards else 0
                
                # Log metrics every episode
                logger.info(
                    f"Thread {rank} | "
                    f"Episode {episode_count} | "
                    f"Steps {counter.value} | "
                    f"Score {current_episode_reward:.1f} | "
                    f"AvgScore {avg_score:.1f} | "
                    f"AvgMaxP {avgmaxp:.1f} | "
                    f"Best {best_score:.1f}"
                )
                
                obs = env.reset()
                state = torch.from_numpy(obs)
                break

        # Rest of the training code remains the same...
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()
            
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + args.gamma * R
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            td_error = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + td_error
            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()