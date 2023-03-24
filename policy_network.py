# Script for the policy architecture of agent

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
def cat_softmax(probs):
    cat_distr = OneHotCategorical(probs = probs)
    return cat_distr.sample(), cat_distr.entropy()

class SNet(nn.Module):
    '''
    
    '''
    def __init__(self, opts) -> None:
        super(SNet, self).__init__()
        self.opts = opts
        self.output_size = opts.n_vocab
        # self.obv_vec_size = opts.obv_vec_size
        # self.order_size = opts.order_vec_size
        self.rnn_input_size = opts.n_concepts
        # print(f'obv_vec_size = {self.obv_vec_size} , order_size = {self.order_size}')
        # sequential model for generating tuple order for
        
        # self.order = nn.Sequential(
        #     nn.Linear(self.obv_vec_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.order_size),
        #     nn.Softmax(dim=0)
        # )
        # Taking order and generating messages using RNN
        self.hidden_size = self.opts.rnn_hidden_size
        self.i2h = nn.Linear(self.rnn_input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.rnn_input_size + self.hidden_size, self.output_size)
        self.o2o = nn.Linear(self.opts.rnn_hidden_size + self.output_size, self.output_size)
        self.softmax = nn.Softmax(dim=0)
    
    def get_concept_order(self,input_obv):
        t = torch.cat((input_obv['agent_observation'], input_obv['target_embedding'], input_obv['avg_color']))
        concept_order = self.order(t)
        return concept_order
    
    def rnn_step(self, input, hidden):
        input_combined = torch.cat((input, hidden))
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output))
        output = self.o2o(output_combined)
        # output = self.softmax(output)
        return output, hidden
    
    def generate_message(self, concept_order, concepts, hidden):
        '''
        Returns the message and log probs of those actions
        '''
        digits = []
        message = []
        log_probs = 0.
        entropy = 0.
        log_probs_i = []
        for c in concept_order:
            if c == 'sector':
                # encode sector here 
                output_s, hidden = self.rnn_step(concepts['sector'], hidden)
            elif c ==  'segment':
                output_s,hidden = self.rnn_step(concepts['segment'], hidden)
            elif c == 'color':
                output_s, hidden = self.rnn_step(concepts['color'], hidden)
            
            digits.append(output_s)
            probs = F.softmax(output_s, dim=0)
            print(f'speaker probs {c} = {probs}')

            predict, entropy = cat_softmax(probs)
            log_probs += torch.log((predict * probs).sum(dim=0))
            log_probs_i.append(torch.log((predict * probs).sum(dim=0)))
            # now generate message and probability
            message.append(predict)
        
        message = torch.stack(message)
        digits = torch.stack(digits)

        return message, log_probs, digits, log_probs_i, entropy
    

        # getting 3 messages from the speaker
    def forward(self, concept_order, concepts, hidden):
        msg, log_probs, digits,log_probs_i,entropy = self.generate_message(concept_order, concepts,hidden)
        return msg, log_probs, digits,log_probs_i,entropy
    
    def initHidden(self):
        return torch.zeros(self.hidden_size)

    


class LNet(nn.Module):
    def __init__(self, opts) -> None:
        super(LNet, self).__init__()
        self.opts = opts
        self.output_size = opts.n_concepts
        self.input_size = opts.n_vocab
        self.hidden_size = opts.rnn_hidden_size

        # rnn for listening to messages
        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.o2o = nn.Linear(self.hidden_size + self.output_size, self.output_size)
        self.softmax = nn.Softmax()
    def pred_concept(self, input, hidden, concept):
        # vocab is the message
        input_combined = torch.cat((input, hidden))
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output))
        output = self.o2o(output_combined)
        # output = self.softmax(output)
        return output, hidden  
    def forward(self, msg, hidden, concept_order):
        pred_concepts = []
        digits = []
        log_probs_i = []
        log_probs = 0.
        entropy = 0.
        for i in range(3):
            output, hidden = self.pred_concept(msg[i], hidden, concept_order[i])
            if concept_order[i] == 'sector':
                output[4:] = -10000.0   
            elif concept_order[i] == 'segment':
                output[:4] = -10000.0
                output[8:]=-10000.0
            elif concept_order[i] == 'color':
                output[:8] = -10000.0
            digits.append(output)
            probs = F.softmax(output, dim=0)
            print(f'listener probs {i} = {probs}')
            predict, entropy = cat_softmax(probs)
            log_probs += torch.log((predict * probs).sum(dim=0))
            log_probs_i.append(torch.log((predict * probs).sum(dim=0)))
            pred_concepts.append(predict)
                    
        pred_concepts = torch.stack(pred_concepts)
        digits = torch.stack(digits)

        return pred_concepts, log_probs, digits,log_probs_i, entropy

        

    def initHidden(self):
        return torch.zeros(self.hidden_size)
    


# Centralised policy for speaker and listener

class Policy(nn.Module):
    def __init__(self, opts) -> None:
        super(Policy, self).__init__()
        self.opts = opts
        self.snet = SNet(self.opts)
        self.lnet = LNet(self.opts)
        self.speaker_hidden = self.snet.initHidden()
        self.listener_hidden = self.lnet.initHidden()
    
    def order(self, input_obv):
        self.concept_order = self.snet.get_concept_order(input_obv)
    
    def speak(self, input_concept, hidden):
        output, hidden = self.snet(input_concept, hidden)
        return output, hidden
    
    def listen(self, input, hidden):
        output, hidden = self.lnet(input, hidden)
        return output, hidden
        