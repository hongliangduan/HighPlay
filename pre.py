import numpy as np
import os
import subprocess
import json
import copy
import shutil
import requests
import tarfile

host_url="https://api.colabfold.com"
headers = {}
use_pairing=False
submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"


def CC_index(peptide):
    indexes_of_c = []
    index = -1
    while True:
        index = peptide.find('C', index + 1)
        if index == -1:
            break
        indexes_of_c.append(index)
    C1 = indexes_of_c[0]
    C2 = indexes_of_c[-1]
    return C2, C1

def CC_distance(peptide):
    C2, C1 = CC_index(peptide)
    return C2 - C1


def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
        query += f">{n}\n{seq}\n"
        n += 1

    while True:
        error_count = 0
        try:
            res = requests.post(f'{host_url}/{submission_endpoint}', data={ 'q': query, 'mode': mode }, timeout=6.02, headers=headers)
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            error_count += 1
            if error_count > 5:
                raise
            continue
        break

    try:
        out = res.json()
    except ValueError:
        out = {"status":"ERROR"}
    return out

def download(ID, path):
    error_count = 0
    while True:
        try:
            res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02, headers=headers)
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            error_count += 1
            if error_count > 5:
                raise
            continue
        break
    with open(path,"wb") as out: out.write(res.content)

def msa(path, seq):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    x = seq
    tar_gz_file = f'{path}/out.tar.gz'
    mode = "pairgreedy"
    N = 101
    seqs_unique = []
    seqs = [x] if isinstance(x, str) else x
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    out = submit(seqs_unique, mode, N)
    ID, _ = out["id"],0
    download(ID, tar_gz_file)

    # 解压
    with tarfile.open(tar_gz_file) as tar_gz:
        tar_gz.extractall(path)

    # 删除其他文件，保留a3m
    for file_name in os.listdir(path):
        if file_name.endswith('.sh'):
            os.remove(path + '/' + file_name)
        if file_name.endswith('.gz'):
            os.remove(path + '/' + file_name)
        if file_name.endswith('.m8'):
            os.remove(path + '/' + file_name)
  
    # 删除a3m的最后一行
    with open(f'{path}/uniref.a3m', 'r') as file:
        lines = file.readlines()
    if lines:
        lines.pop()
    with open(f'{path}/uniref.a3m', 'w') as file:
        file.writelines(lines)



def read_pdb_coordinates(file_path):
    protein_resno = []
    protein_atoms = []
    protein_atom_coords = []
    
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                resno = int(line[23:30])
                protein_resno.append(resno)
                atoms = line[12:16].strip()
                protein_atoms.append(atoms)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                protein_atom_coords.append([x, y, z])
    
    return np.array(protein_resno), np.array(protein_atoms), np.array(protein_atom_coords)


def find_peptide_index(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return i

        
def initialize_weights(peptide_length):
    '''Initialize sequence probabilities
    '''

    weights = np.random.gumbel(0,1,(peptide_length,20))
    weights = np.array([np.exp(weights[i])/np.sum(np.exp(weights[i])) for i in range(len(weights))])

    #Get the peptide sequence
    #Residue types
    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    peptide_sequence = ''.join(restypes[[x for x in np.argmax(weights,axis=1)]])

    while peptide_sequence.count('C')!=2:
        weights = np.random.gumbel(0,1,(peptide_length,20))
        weights = np.array([np.exp(weights[i])/np.sum(np.exp(weights[i])) for i in range(len(weights))])
        peptide_sequence = ''.join(restypes[[x for x in np.argmax(weights,axis=1)]])
    else:
        if CC_distance(peptide_sequence)/len(peptide_sequence)>2/3:
            return weights, peptide_sequence
        else:
            return initialize_weights(peptide_length)        




def predict_cycle(receptor_seq, peptide_sequence, output_dir_base, receptor_if_residues, receptor_name, num_iter):
    # with open('tmp1.fasta', 'w') as f:
    #     f.writelines('>'+receptor_name+'\n')
    #     f.writelines(receptor_seq+':'+'\n')
    #     f.writelines(peptide_sequence)
    parent_path = os.path.dirname(output_dir_base)
    with open(f'{parent_path}/uniref.a3m', 'r') as a3m_file:
        with open('tmp1.a3m', 'w') as f:
            f.writelines('#'+str(len(receptor_seq))+','+str(len(peptide_sequence))+'\t'+'1,1'+'\n')
            f.writelines('>101'+'\t'+'102'+'\n')
            f.writelines(receptor_seq+peptide_sequence+'\n')
            for line in a3m_file:
                if line.startswith('>'):
                    f.writelines(line)
                else:
                    f.writelines(line.strip()+'-'*len(peptide_sequence)+'\n')
            f.writelines('>102'+'\n')
            f.writelines('-'*len(receptor_seq)+peptide_sequence)


    from y_train import init_pep
    C2, C1 = CC_index(init_pep)
    command = 'CUDA_VISIBLE_DEVICES=1 colabfold_batch --model-type alphafold2_multimer_v3 tmp1.a3m tmp1 --num-models 1 --flag-cyclic 0 1 --index-ss 1 '+str(C1)+' '+str(C2) 
    subprocess.call(command, shell=True)

    file_list = os.listdir('./tmp1')
    for file_name in file_list:
        if "rank_001" in file_name and file_name.endswith(".json"):
            file_path = os.path.join('./tmp1', file_name)
            with open(file_path,'r')as fp:
                json_data = json.load(fp)
                plddt = sum(json_data['plddt'][-len(peptide_sequence):])/len(json_data['plddt'][-len(peptide_sequence):])

        elif "rank_001" in file_name and file_name.endswith(".pdb"):
            file_path = os.path.join('./tmp1', file_name)
            protein_resno, protein_atoms, protein_atom_coords = read_pdb_coordinates(file_path)
            peptide_length = len(peptide_sequence)
            peptide_index = find_peptide_index(protein_resno)

            receptor_coords = protein_atom_coords[:peptide_index]
            peptide_coords = protein_atom_coords[peptide_index:]
            receptor_resno = protein_resno[:peptide_index]
            peptide_resno = protein_resno[peptide_index:]

            receptor_if_pos = []
            for ifr in receptor_if_residues:
                receptor_if_pos.extend([*np.argwhere(receptor_resno==ifr)])
            receptor_if_pos = np.array(receptor_if_pos)[:,0]

            mat = np.append(peptide_coords,receptor_coords[receptor_if_pos],axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            l1 = len(peptide_coords)
            contact_dists = dists[:l1,l1:]
            closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]
            closest_dists_receptor = contact_dists[np.argmin(contact_dists,axis=0),np.arange(contact_dists.shape[1])]

            new_file_path = os.path.join('./tmp1', f'unrelaxed_{num_iter}.pdb')
            os.rename(file_path, new_file_path)

    new_file = os.path.join(output_dir_base, f'unrelaxed_{num_iter}.pdb')
    os.replace(f'./tmp1/unrelaxed_{num_iter}.pdb', new_file)
    shutil.rmtree('./tmp1')

    return closest_dists_peptide.mean(), closest_dists_receptor.mean(), plddt




def sequence_to_onehot(sequence, aatypes):
    restype_order = {restype: i for i, restype in enumerate(aatypes)}
    num_entries = max(restype_order.values()) + 1
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
    for aa_index, aa_type in enumerate(sequence):
        aa_id = restype_order[aa_type]
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr


def onehot_to_sequence(onehot, aatypes):
    mapping = {restype: i for i, restype in enumerate(aatypes)}
    inverse_mapping = {index: aa_type for aa_type, index in mapping.items()}
    non_zero_indices = np.argmax(onehot, axis=1)
    sequence = [inverse_mapping[index] for index in non_zero_indices]
    sequence_str = ''.join(sequence)
    return sequence_str



def get_availables(state, aatypes):
    '''环肽固定C1和C2,其余位置可变'''
    availables = np.flatnonzero(state==0)
    sequence = onehot_to_sequence(state, aatypes)
    from y_train import init_pep
    C2, C1 = CC_index(init_pep)
    # C2, C1 = CC_index(sequence)
    availables = list(availables)
    availables.append(20*C1 + aatypes.index("C"))
    availables.append(20*C2 + aatypes.index("C"))
    for i in range(20*C1, 20*C1+20):
        availables.remove(i)
    for i in range(20*C2, 20*C2+20):
        availables.remove(i)
    availables = np.array(availables)
    return availables




def mutate_seq(peptide_sequence, ex_list):
    '''环肽固定C随机突变三个位置
    '''
    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])
    seqlen = len(peptide_sequence)
    #Mutate seq
    seeds = peptide_sequence
    #Go through a shuffled version of the positions and aas
    from y_train import init_pep
    C2, C1 = CC_index(init_pep)
    # C2, C1 = CC_index(peptide_sequence)
    local = list(range(seqlen))
    local.remove(C1)
    local.remove(C2)
    local = np.array(local)
    while True:
        seeds = peptide_sequence
        pi_s = np.random.choice(local, 3, replace=False) # True,mutate<=3, False=3
        for pi in pi_s:
            aa = np.random.choice(restypes, replace=False)
            #new_s
            new_seq = seeds[:pi] + aa + seeds[pi + 1:]
            seeds = new_seq
        if new_seq not in ex_list:
            break
    return new_seq