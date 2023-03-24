import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F



def choose_concept(vocab, hidden_l,world, opts,listener, concept_type):
    '''
    Used by the listener to predict the concept from the vocabulary
    
    Documentation:
    vocab: string
    hidden_l: hidden state of listener
    world: world object
    opts: options object
    listener: listener obje+ct

    Returns:
    concept: string
    output_l: output of listener
    action_s: action of speaker

    '''
    # concept space is ordered as follows: [sector[0-7], segment[8-11], color[12-15]]

    vocab_encoding = world.get_vocab_tensor(vocab)
    # pass the vocabulary to listener
    output_l, hidden_l = listener(vocab_encoding, hidden_l)
    # depending on the concept type we decide the possible actions
    # predict the concept
    if concept_type == 'sector':
        # we need distribution over index 0-7
        # normalise over sector index of output_l
        # s = output_l.detach().numpy()[:8].sum()
        # p = output_l.detach().numpy()[:8]/s
        vec = output_l.detach()[:8]
        # print(f'vec = {vec}')

        p = F.softmax(vec, dim=0)
        p = p/p.sum()
        # print(f'p = {p.sum()}')
        # print(f'p = {p}')
        concept_i = np.random.choice(np.arange(0,8), p =p.numpy())
        # pi_c_given_v 
        pi_given_v = p[concept_i]
        
    if concept_type == 'segment':
        # we need distribution over index 8-11
        # print(f'output_l = {output_l.detach().numpy()[8:12]}')
        vec = output_l.detach()[8:12]
        # print(f'vec = {v+ec}')

        p = F.softmax(vec)
        # print(f'p = {p.sum()}')
        concept_i = np.random.choice(np.arange(8,12), p =p.numpy())
        # pi_c_given_v
        pi_given_v = p[concept_i-8]
    
    if concept_type == 'color':
        # we need distribution over index 12-15
        # s = output_l.detach().numpy()[12:16].sum()
        # p = output_l.detach().numpy()[12:16]/s
        vec = output_l.detach()[12:16]
        p = F.softmax(vec)
        p = p/p.sum()
        # print(f'p = {p}')
        p = p/p.sum()
        # print(f'p = {p.sum()}')
        concept_i = np.random.choice(np.arange(12,16), p = p.numpy())
        # pi_c_given_v 
        pi_given_v = p[concept_i-12]

    concept = world.all_concepts[concept_i]
    action_l = concept_i
    return concept, output_l, hidden_l, action_l, pi_given_v

def choose_vocab(concept, hidden_s, world, opts,speaker, ):
    '''
    Used by the speaker to choose the vocabulary from the concept
    
    Documentation:
    concept: string
    hidden_s: hidden state of speaker
    world: world object
    opts: options object
    speaker: speaker obje+ct
    
    Returns:
    vocab: string
    output_s: output of speaker
    action_l: action of listener
    '''

    # get concept encoding
    concept_encoding = world.get_concept_tensor(concept)
    # get word for this concept
    output_s, hidden_s = speaker(concept_encoding, hidden_s)
    # get vocabulary from output
    vocab_i = np.random.choice(np.arange(opts. n_vocab), p = output_s.detach().numpy())
    action_s = vocab_i
    vocab = world.vocabularies[vocab_i]
    print(f'output = {output_s}')
    return vocab, output_s, hidden_s, action_s

def get_reward2(pred_concepts, concepts, chosen_order):
    '''
        Returns the reward for the speaker and listener
    '''
    first = chosen_order[0]
    if first == 'sector':
        if pred_concepts[0].equal(concepts[0]) :
            # Case1 :only one vertex is there or color is unique:
            # Case 2: target vertex has unique color
            return 1
        else:
            # go for next pred_concept
            second = chosen_order[1]
            if second == 'segment':
                if pred_concepts[1].equal(concepts[1]):
                    # Case 3: only target vertex is present in the intersection region
                    # or
                    # Case 4: target vertex is present in the intersection region and has unique color
                    return 1
                else:
                    return 0
    if first == 'segment':
        if pred_concepts[0].equal(concepts[0]):
            # Case 5: only one vertex is there
            # Case 6: target vertex has unique color
            return 1
        else:
            # go for next pred_concept
            second = chosen_order[1]
            if second == 'sector':
                if pred_concepts[1].equal(concepts[1]):
                    # Case 7: only target vertex is present in the intersection region
                    # or
                    # Case 8: target vertex is present in the intersection region and has unique color
                    return 1
                else:
                    return 0
    if first == 'color':
            return 0

def get_reward(concepts, pred_concepts, chosen_order):
    '''
        Returns the reward for the speaker and listener
    '''
    pos_r = 1
    neg_r = -10
    rewards = []
    if concepts[0].equal(pred_concepts[0]):
        rewards.append(pos_r)
    else:
        rewards.append(neg_r)
    if concepts[1].equal(pred_concepts[1]):
            rewards.append(pos_r)
    else:
        rewards.append(neg_r)
    if concepts[2].equal(pred_concepts[2]):
            rewards.append(pos_r)
    else: 
        rewards.append(neg_r)
    return rewards
def run_episode(env, obv, speaker, listener, opts, world):
    '''
    Run one episode of the game

    Returns:
    ----------------
    speaker_reward: list of rewards for speaker
    listener_reward: list of rewards for listener
    speaker_action: list of actions for speaker
    listener_action: list of actions for listener
    speaker_action_probs: list of action probabilities for speaker
    listener_action_probs: list of action probabilities for listener
    '''
    speaker_reward = []
    listener_reward = []
    # pass this observation and get the concept order 
    # order_probs = speaker.get_concept_order(obv)
    # get the concepts 
    # chosen_order = np.random.choice(np.arange(opts.order_vec_size), p = order_probs.detach().numpy())
    # chosen_order = opts.order_vec[chosen_order]
    # get concepts from order
    chosen_order = ['segment', 'sector', 'color']
    octant, segment, quadrant, color = env.world.get_concepts(env.target_index, env.source_index)
    # generate vocabulary using RNN
    sec_e = world.get_concept_tensor(octant)
    seg_e = world.get_concept_tensor(segment)
    col_e = world.get_concept_tensor(color)
    concepts = {'sector': sec_e, 'segment': seg_e, 'color': col_e}
    print(f'concepts = {concepts}')
    hidden_s = speaker.initHidden()
    msg, s_log_prob, s_digits,s_log_probs_i, s_entropy = speaker(chosen_order,concepts,hidden_s)
    agi = torch.argmax(msg, dim = 1)
    print(f'vocab 1 = {world.vocabularies[agi[0]]}, vocab 2 = {world.vocabularies[agi[1]]}, vocab 3 = {world.vocabularies[agi[2]]}')
    # print(f'vocab chosen = {world.vocabularies[np.where(msg.detach().numpy() == 1)]}')
    # reward = [1]
    # return reward, log_prob
    # print(f'pi_m_given_concept = {type(pi_m_given_concept)}')
    # passing vocabularies to listener
    hidden_l = listener.initHidden()
    pred_concepts, l_log_prob, l_digits,l_log_probs_i, l_entropy = listener(msg, hidden_l, concept_order = chosen_order)
    concepts_vec = [concepts[chosen_order[0]], concepts[chosen_order[1]], concepts[chosen_order[2]]]
    # calculate reward
    rewards = get_reward(concepts_vec, pred_concepts, chosen_order)
    return rewards, s_log_prob, l_log_prob,s_log_probs_i,l_log_probs_i, s_entropy, l_entropy



def vanilla_policy_gradient(env, obv, speaker, listener, opts, world, s_optim, l_optim, s_reward, l_reward, s_action, l_action, s_action_prob, l_action_prob):
    
    '''
    ================== Vanilla Policy Gradient ==================
    '''
    # speaker part 
    discount_factor = 1
    s_optim.zero_grad()
    r = np.full(len(s_reward), discount_factor) ** np.arange(len(s_reward)) * np.array(s_reward)
    r = r[::-1].cumsum()[::-1]
    discounted_rewards = torch.tensor(r - r.mean())
    s_action_probs = torch.stack(s_action_probs)
    s_action = torch.tensor(s_action, dtype=torch.int64).reshape(len(s_action), -1 )
    k = torch.gather(s_action_probs, 1, s_action)
    log_prob = torch.log(s_action_prob)
    selected_log_probs = discounted_rewards.view(1,-1)*log_prob.view(1,-1)
    s_loss = -selected_log_probs.sum()
    s_loss.backward()
    s_optim.step()

    
    # listener part
    l_optim.zero_grad()
    r = np.full(len(l_reward), discount_factor) ** np.arange(len(l_reward)) * np.array(l_reward)
    r = r[::-1].cumsum()[::-1]
    discounted_rewards = torch.tensor(r - r.mean())
    # print(f'discounted_rewards {discounted_rewards}')
    # l_action_probs = torch.stack(l_action_probs)
    # l_action = torch.tensor(l_action, dtype=torch.int64).reshape(len(l_action), -1 )
    # k = torch.gather(l_action_probs, 1, l_action)
    log_prob = torch.log(l_action_prob)
    
    selected_log_probs = discounted_rewards.view(1, -1)*log_prob.view(1,-1)
    # print(f'selected_log_probs {selected_log_probs}')
    l_loss = -selected_log_probs.sum()
    l_loss.backward()
    l_optim.step()

    return s_loss, l_loss

def vpg(rewards, log_probs_i,entropy, optimiser):
    
    gamma = 1
    rewards = torch.tensor(rewards).reshape(1,-1)
    optimiser.zero_grad()
    log_probs_i = torch.stack(log_probs_i)
    # log_probs_i = torch.stack(l_log_probs_i)
    # s_log_prob = s_log_probs_i*rewards.sum()
    # rewards = torch.cumsum(rewards, dim = 0)
    # log_probs_i = torch.cumsum(log_probs_i, dim = 0)
    log_prob = log_probs_i*rewards
    log_prob = torch.cumsum(log_prob, dim = 0)
    # r = np.full(len(rewards), gamma) ** np.arange(len(rewards)) * np.array(rewards)
    # r = r[::-1].cumsum()[::-1]
    # discounted_rewards = torch.tensor(r - r.mean())
    # discounted_rewards = torch.tensor(rewards)
    # selected_log_probs = discounted_rewards*log_probs
    # loss = -selected_log_probs.sum()
    loss = (log_prob.sum() + 0.5*entropy.sum())
    loss.backward()
    optimiser.step()
    return loss

def train(env, speaker, listener, opts, world, n_episodes = 1, epochs = 100000 ):
    # step 1 : get goal from env
    optimizer1 = torch.optim.Adam(speaker.parameters(), lr = 0.001)
    optimizer2 = torch.optim.Adam(listener.parameters(), lr=0.001)
    for _ in range(epochs):
        print(f'epoch {_}')
        share_reward, l_rewards,s_probs, l_probs = [], [], [], []
        for i in range(n_episodes):
            # reset the environment
            obv = env.reset()
            rewards, s_log_prob, l_log_prob,s_log_prob_i,l_log_prob_i, s_entropy, l_entropy = run_episode(env, obv, speaker, listener, opts, world)
            print(f'rewards {rewards}')
            share_reward.append(rewards)
            
        # train the speaker and listener
        # s_loss, l_loss = vanilla_policy_gradient(env, obv, speaker, listener, opts, world, optimizer1, optimizer2, \
        # s_rewards, l_rewards, s_actions, l_actions, s_probs, l_probs)
        s_loss = vpg(rewards, s_log_prob_i, s_entropy ,optimiser=optimizer1)
        l_loss = vpg(rewards, l_log_prob_i, l_entropy, optimiser=optimizer2)

        print(f"s_loss {s_loss}")
        print(f'l_loss {l_loss}')
        print('='*50) 
        print('\n')  

    
def update_listener(listener):
    pass
def run_centralised_episode():
    pass
