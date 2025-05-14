import random
import numpy as np
from collections import deque
from mcts import MCTSPlayer
from policyvaluenet import PolicyValueNet
from mutate import Mutate, Seqenv, sequence_scores, sequence_scores1, sequence_scores2, playout_dict, init_dict, move_dict
import os
from pre import msa
import pandas as pd
import csv

aatypes=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
init_pep = []


def filter_dict(before_dict, filtered_dict):
    for peptide_sequence, values in before_dict.items():
        if values[2] > 70:  # 检查plddt值
            if peptide_sequence not in filtered_dict:  # 确保peptide_sequence不重复
                filtered_dict[peptide_sequence] = values


class TrainPipeline():
    def __init__(self, init_seq, receptor_seq, pocket, output_dir, num_iterations, plDDT_only, receptor_name, jumpout_num, init_model=None):
        # peptide and pocket params
        self.init_seq = init_seq
        global init_pep
        init_pep = init_seq
        self.receptor_seq = receptor_seq
        self.pocket = pocket
        self.peptide_length = len(init_seq)
        self.output_dir = output_dir
        self.onlyplddt = plDDT_only
        self.receptor_name = receptor_name
        self.jumpout_num = jumpout_num
        self.aatypes=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        # self.n_playout = 16  # num of simulations for each move
        self.n_playout = len(init_seq)
        self.c_puct = 0.5  
        self.batch_size = 8  # mini-batch size for training
        self.data_buffer = deque(maxlen=1000)
        self.play_batch_size = 1
        self.game_batch_num = num_iterations
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        #
        self.seq_env = Seqenv(self.init_seq, self.aatypes, self.pocket, self.receptor_seq, self.output_dir, self.receptor_name, self.onlyplddt)
        self.mutate = Mutate(self.seq_env)
        self.playout_dict = {}
        


        if init_model: # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(len(self.aatypes),self.peptide_length,model_file=init_model,use_gpu=True)
        else: # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(len(self.aatypes),self.peptide_length,use_gpu=True)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)


    def collect_selfplay_data(self, n_games):
        """collect self-play data for training"""
        for _ in range(n_games):
            play_data = self.mutate.start_mutating(self.mcts_player,temp=self.temp,jumpout=self.jumpout_num)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
            np.save(self.output_dir+'/data_buffer.npy',np.array(self.data_buffer, dtype=object))

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1-np.var(np.array(winner_batch)-old_v.flatten())/np.var(np.array(winner_batch)))
        explained_var_new = (1-np.var(np.array(winner_batch)-new_v.flatten())/np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        parent_path = os.path.dirname(self.output_dir)
        msa(parent_path, self.receptor_seq)
        for iter_num in range(self.game_batch_num):
            print("start i:{}".format(iter_num+1))
            self.collect_selfplay_data(self.play_batch_size)
            print("batch i:{}, episode_len:{}".format(iter_num+1, self.episode_len))

            with open(f"{self.output_dir}/playout_dict.csv", 'w') as f:
                f.write('sequence, plddt, if_dist_peptide, if_dist_receptor, reward, file_name\n')
                writer = csv.writer(f)
                for key, value in playout_dict.items():
                    writer.writerow([key, value[2], value[0], value[1], value[3], f"unrelaxed_{value[4]}.pdb"])

            with open(f"{self.output_dir}/init_dict.csv", 'w') as f:
                f.write('sequence, plddt, if_dist_peptide, if_dist_receptor, reward, file_name\n')
                writer = csv.writer(f)
                for key, value in init_dict.items():
                    writer.writerow([key, value[2], value[0], value[1], value[3], f"unrelaxed_{value[4]}.pdb"])

            with open(f"{self.output_dir}/move_dict.csv", 'w') as f:
                f.write('sequence, plddt, if_dist_peptide, if_dist_receptor, reward, file_name\n')
                writer = csv.writer(f)
                for key, value in move_dict.items():
                    writer.writerow([key, value[2], value[0], value[1], value[3], f"unrelaxed_{value[4]}.pdb"])

            high_dict = {}
            filter_dict(init_dict, high_dict)
            filter_dict(move_dict, high_dict)
            filter_dict(playout_dict, high_dict)
            with open(f"{self.output_dir}/high.csv", 'w') as f:
                f.write('sequence, plddt, if_dist_peptide, if_dist_receptor, reward, file_name\n')
                writer = csv.writer(f)
                for key, value in high_dict.items():
                    writer.writerow([key, value[2], value[0], value[1], value[3], f"unrelaxed_{value[4]}.pdb"])
            # p_dict = pd.DataFrame(playout_dict)
            # p_dict.to_csv(f"{self.output_dir}/playout_dict.csv", index=False)

            # i_dict = pd.DataFrame(init_dict)
            # i_dict.to_csv(f"{self.output_dir}/init_dict.csv", index=False)

            # m_dict = pd.DataFrame(move_dict)
            # m_dict.to_csv(f"{self.output_dir}/move_dict.csv", index=False)

            p_seqs = np.array(list(sequence_scores['sequence']))
            p_plddts = np.array(list(sequence_scores['plddt']))
            df_playout = pd.DataFrame(
                    {"peptide": p_seqs, "plddt": p_plddts})
            df_playout.to_csv(f"{self.output_dir}/playout.csv", index=False)

            i_seqs = np.array(list(sequence_scores1['sequence']))
            i_plddts = np.array(list(sequence_scores1['plddt']))
            df_init = pd.DataFrame(
                    {"peptide": i_seqs, "plddt": i_plddts})
            df_init.to_csv(f"{self.output_dir}/init.csv", index=False)

            m_seqs = np.array(list(sequence_scores2['sequence']))
            m_plddts = np.array(list(sequence_scores2['plddt']))
            df_move = pd.DataFrame(
                    {"peptide": m_seqs, "plddt": m_plddts})
            df_move.to_csv(f"{self.output_dir}/move.csv", index=False)

            if len(self.data_buffer) > self.batch_size:
                print("start training...")
                loss, entropy = self.policy_update()
                self.policy_value_net.save_model(self.output_dir+'_current_policy.pt')