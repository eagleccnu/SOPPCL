import os
import torch
import esm
import dill
import fileinput
import numpy as np
from Bio import SeqIO


# ─────────────────────────────────────────────
# 配置区：根据实际环境修改以下路径
# ─────────────────────────────────────────────
HHSUITE_BUILD_DIR = '/home/amax/hh-suite/build'
UNIPROT_DB = '/home/amax/Project/database/uniprot20_2013_03/uniprot20_2013_03'
TEMP_DIR = '/home/amax/Project/database/tmp/hhm_tmp'
# ─────────────────────────────────────────────


def read_fasta(file_path):
    for record in SeqIO.parse(file_path, 'fasta'):
        id = record.id
        sequence = str(record.seq)
    return id, sequence


def generate_hhm(fasta_path, pdb_id):
    os.makedirs(TEMP_DIR, exist_ok=True)
    a3m_path = os.path.join(TEMP_DIR, pdb_id + '.a3m')
    hhm_path = os.path.join(TEMP_DIR, pdb_id + '.hhm')
    os.chdir(HHSUITE_BUILD_DIR)
    cmd1 = 'hhblits -i ' + fasta_path + \
           ' -d ' + UNIPROT_DB + ' -n 2 -mact 0.01 ' \
           '-oa3m ' + a3m_path
    cmd2 = 'hhmake -i ' + a3m_path + ' -o ' + hhm_path
    os.system('export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH" && %s && %s' % (cmd1, cmd2))
    return hhm_path


def find_line(file):
    f = open(file, 'r')
    for line, strin in enumerate(f):
        if strin.split() == ['#']:
            fline = line
    f.close()
    return fline


def hhm(file):
    h = []
    hh = []
    hhm_feat = []
    finput = fileinput.input(file)
    for line, strin in enumerate(finput):
        if line > find_line(file) + 4:
            str_ve = strin.split()[0:22]
            if len(str_ve) > 1:
                h.append(list(str_ve))
    for i in range(0, len(h), 2):
        hh.append(h[i]+h[i+1])
    for item in hh:
        item = item[2:]
        for i in range(len(item)):
            if item[i] == '0':
                item[i] = 1
            elif item[i] == '*':
                item[i] = 0
            else:
                item[i] = round(2**(int(item[i])/(-1000)), 2)
        hhm_feat.append(item)
    finput.close()
    return hhm_feat


def process_fasta(fasta_path, output_path):
    # 读取序列
    pdbid, sequence = read_fasta(fasta_path)
    data = [(pdbid, sequence)]

    # ESM-2 提取嵌入
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1:tokens_len - 1])

    array_data = sequence_representations[0].numpy()
    esm_emb = array_data.tolist()

    # 生成 HHM 文件
    hhm_path = generate_hhm(fasta_path, pdbid)

    # 解析 HHM 特征并与 ESM 拼接保存
    hmm = hhm(hhm_path)

    esm_hhm = [x + y for x, y in zip(esm_emb, hmm)]
    # esm_hhm = np.array(esm_hhm)

    with open(output_path, 'wb') as f:
        dill.dump(esm_hhm, f)

    print('输出文件：', output_path)
    print('特征shape：', np.array(esm_hhm).shape)


if __name__ == '__main__':
    fasta_path = '/home/amax/Project_torch/pytorch_learning/1pd7.fasta'   # 修改为输入 FASTA 文件路径
    output_path = '/home/amax/Project_torch/pytorch_learning/1pd7.dat'   # 修改为输出 DAT 文件路径
    process_fasta(fasta_path, output_path)



