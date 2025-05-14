import numpy as np
import copy
import mcts
from pre import predict_cycle , onehot_to_sequence , sequence_to_onehot , get_availables , mutate_seq

count = 0
count2 = 20000
sequence_scores = {'if_dist_peptide':[], 'if_dist_receptor':[],'plddt':[], 'reward':[],'sequence':[]}   #playout
sequence_scores1 = {'if_dist_peptide':[], 'if_dist_receptor':[],'plddt':[], 'reward':[],'sequence':[]}   #init
sequence_scores2 = {'if_dist_peptide':[], 'if_dist_receptor':[],'plddt':[], 'reward':[],'sequence':[]}   #move
playout_dict = {}
init_dict = {}
move_dict = {}

class Seqenv():
    """board for the game"""

    def __init__(self, init_seq, aatypes, pocket, receptor_seq, output_dir, receptor_name, onlyplddt):
        self.start_seq = init_seq
        self.aatypes = aatypes
        # self.states = []
        self.peptide_len = len(self.start_seq)
        # self.availables = list(range(self.peptide_len * len(self.aatypes)))
        self.pocket = pocket
        self.receptor_seq = receptor_seq
        self.output_dir = output_dir
        self.receptor_name = receptor_name
        self.onlyplddt = onlyplddt
        self.repeated = False
        self.init_state = sequence_to_onehot(self.start_seq, self.aatypes)
        self.previous_init_state = sequence_to_onehot(self.start_seq, self.aatypes)
        self.init_state_count = 0

        receptor_if_residues = self.pocket.split(',')
        seq_list = [int(num) for num in receptor_if_residues]
        self.receptor_if_residues = np.array(seq_list)
        self.index = 10000
        self.reward = 0.0
        self.seqs = []

    def init_seq_state(self):
        self.repeated = False
        self.previous_reward = -float("inf")
        self.unuseful_move = 0
        self._state = copy.deepcopy(self.init_state)
        
        self.availables = get_availables(self._state, self.aatypes)

        combo = onehot_to_sequence(self._state, self.aatypes)
        self.init_combo = combo
        self.seqs.append(combo)

        if combo not in init_dict.keys():
            if_dist_peptide, if_dist_receptor, plddt = predict_cycle(self.receptor_seq, combo, self.output_dir, self.receptor_if_residues, self.receptor_name, self.index)
            self.index = self.index + 1
            if self.onlyplddt==True:
                reward =  plddt/100
            else:
                reward = (2*plddt)/(if_dist_peptide+if_dist_receptor)
                reward = reward/10
            self.reward = reward

            init_dict[combo] = [if_dist_peptide, if_dist_receptor, plddt, reward, self.index-1]

        else:
            if_dist_peptide = init_dict[combo][0]
            if_dist_receptor = init_dict[combo][1]
            plddt = init_dict[combo][2]
            reward = init_dict[combo][3]
            self.reward = reward


        sequence_scores1['plddt'].append(plddt)
        sequence_scores1['reward'].append(reward)
        sequence_scores1['sequence'].append(combo)

        
        np.save(self.output_dir+'1_plddt.npy', np.array(sequence_scores1['plddt']))
        np.save(self.output_dir+'1_reward.npy', np.array(sequence_scores1['reward']))
        np.save(self.output_dir+'1_sequence.npy', np.array(sequence_scores1['sequence']))

        print("_____________________________init_seq_state")
        print("start_seq", combo)


        self.previous_init_state = copy.deepcopy(self._state)

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = self._state
        return square_state


    def do_move(self, move, playout = 0):
        """make the move for current player"""
        self.previous_reward = self.reward
        one_dim = move // len(self.aatypes)
        two_dim = move % len(self.aatypes)


        if self._state[one_dim, two_dim] == 1:
            self.unuseful_move = 1
            self.reward = 0.0
        else:
            self._state[one_dim,:] = 0
            self._state[one_dim, two_dim] = 1

            self.availables = get_availables(self._state, self.aatypes)

            peptide_sequence = onehot_to_sequence(self._state, self.aatypes)        
            if playout == 0:
                if peptide_sequence not in move_dict.keys():
                    global count2
                    if_dist_peptide, if_dist_receptor, plddt = predict_cycle(self.receptor_seq, peptide_sequence, self.output_dir, self.receptor_if_residues, self.receptor_name, count2)
                    count2 = count2 + 1
                    if self.onlyplddt==True:
                        reward =  plddt/100
                    else:
                        reward = (2*plddt)/(if_dist_peptide+if_dist_receptor)
                        reward = reward/10
                    self.reward = reward

                    move_dict[peptide_sequence] = [if_dist_peptide, if_dist_receptor, plddt, reward, count2-1]

                else:
                    if_dist_peptide = move_dict[peptide_sequence][0]
                    if_dist_receptor = move_dict[peptide_sequence][1]
                    plddt = move_dict[peptide_sequence][2]
                    reward = move_dict[peptide_sequence][3]
                    self.reward = reward

                sequence_scores2['plddt'].append(plddt)
                sequence_scores2['reward'].append(reward)
                sequence_scores2['sequence'].append(peptide_sequence)

 
                np.save(self.output_dir+'2_plddt.npy', np.array(sequence_scores2['plddt']))
                np.save(self.output_dir+'2_reward.npy', np.array(sequence_scores2['reward']))
                np.save(self.output_dir+'2_sequence.npy', np.array(sequence_scores2['sequence']))
            else:
                if peptide_sequence not in playout_dict.keys():
                    global count
                    if_dist_peptide, if_dist_receptor, plddt = predict_cycle(self.receptor_seq, 
                                                                             peptide_sequence, 
                                                                             self.output_dir, 
                                                                             self.receptor_if_residues, 
                                                                             self.receptor_name, 
                                                                             count)
                    count = count + 1
                    self.if_dist_peptide = if_dist_peptide
                    self.if_dist_receptor = if_dist_receptor
                    self.plddt = plddt
                    if self.onlyplddt==True:
                        reward =  plddt/100
                    else:
                        reward = (2*plddt)/(if_dist_peptide+if_dist_receptor)
                        reward = reward/10
                    self.reward = reward

                    playout_dict[peptide_sequence] = [if_dist_peptide, if_dist_receptor, plddt, reward, count-1]
                    
                else:
                    self.if_dist_peptide = playout_dict[peptide_sequence][0]
                    self.if_dist_receptor = playout_dict[peptide_sequence][1]
                    self.plddt = playout_dict[peptide_sequence][2]
                    self.reward = playout_dict[peptide_sequence][3]
                    

                sequence_scores['plddt'].append(self.plddt)
                sequence_scores['reward'].append(self.reward)
                sequence_scores['sequence'].append(peptide_sequence)

                np.save(self.output_dir+'_plddt.npy', np.array(sequence_scores['plddt']))
                np.save(self.output_dir+'_reward.npy', np.array(sequence_scores['reward']))
                np.save(self.output_dir+'_sequence.npy', np.array(sequence_scores['sequence']))


        current_seq = onehot_to_sequence(self._state, self.aatypes)
        if current_seq in self.seqs:
            self.repeated = True
            self._state_fitness = 0.0
        else:
            self.seqs.append(current_seq)
        
        if self.reward > self.previous_reward:  
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0


    def game_end(self):
        """Check whether the game is ended or not"""
        if self.reward < self.previous_reward:
            # print(onehot_to_sequence(self.current_state(), self.aatypes))
            return True
        if self.unuseful_move == 1:
            return True
        if self.repeated:
            return True
        return False


class Mutate():
    """game server"""

    def __init__(self, seqenv):
        self.seqenv = seqenv


    def start_mutating(self, player, temp=1e-3, jumpout = 50):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        if (self.seqenv.previous_init_state == self.seqenv.init_state).all():
            self.seqenv.init_state_count += 1
        if self.seqenv.init_state_count >= jumpout:
            print("Random start replacement****")
            current_start_seq = onehot_to_sequence(self.seqenv.init_state, self.seqenv.aatypes)
            occurred_seqs = list(sequence_scores1['sequence'])
            new_start_seq = mutate_seq(current_start_seq , occurred_seqs)
            self.seqenv.init_state = sequence_to_onehot(new_start_seq, self.seqenv.aatypes)
            self.seqenv.init_state_count = 0


        states, mcts_probs, rewards = [], [], []
        self.seqenv.init_seq_state()
        while True:
            move, move_probs = player.get_action(self.seqenv,temp=temp,return_prob=True)
            states.append(self.seqenv.current_state())
            mcts_probs.append(move_probs)
            rewards.append(self.seqenv.reward)
            # perform a move
            self.seqenv.do_move(move)

            end = self.seqenv.game_end()
            if end:
            # reset MCTS root node
                #rewards = [y_mcts.reward] * len(states)
                player.reset_player()
                return zip(states, mcts_probs, rewards)
    