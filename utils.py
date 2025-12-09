from Bio.PDB import PDBParser, is_aa
import pandas as pd
import numpy as np
import pynmrstar
import re
import csv
import os
import fnmatch
from difflib import SequenceMatcher
import json


# this function is to reduce 3-letter abbreviation to 1-letter abbreviation
def abbr_reduce(long_abbr):
    dict_abbr_reduce = {}
    dict_abbr_reduce['ALA'] = 'A'
    dict_abbr_reduce['ARG'] = 'R'
    dict_abbr_reduce['ASN'] = 'N'
    dict_abbr_reduce['ASP'] = 'D'
    dict_abbr_reduce['CYS'] = 'C'
    dict_abbr_reduce['GLU'] = 'E'
    dict_abbr_reduce['GLN'] = 'Q'
    dict_abbr_reduce['GLY'] = 'G'
    dict_abbr_reduce['HIS'] = 'H'
    dict_abbr_reduce['ILE'] = 'I'
    dict_abbr_reduce['LEU'] = 'L'
    dict_abbr_reduce['LYS'] = 'K'
    dict_abbr_reduce['MET'] = 'M'
    dict_abbr_reduce['PHE'] = 'F'
    dict_abbr_reduce['PRO'] = 'P'
    dict_abbr_reduce['SER'] = 'S'
    dict_abbr_reduce['THR'] = 'T'
    dict_abbr_reduce['TRP'] = 'W'
    dict_abbr_reduce['TYR'] = 'Y'
    dict_abbr_reduce['VAL'] = 'V'

    list_long_abbr = list(dict_abbr_reduce.keys())

    if long_abbr in list_long_abbr:
        return (dict_abbr_reduce[long_abbr])
    else:
        return ('X')


# given a pdb filename, return the pdb id
def read_pdbid_from_filename(file_name):
    pdb_id = re.findall('([a-z0-9A-Z]+).pdb$', file_name)
    if len(pdb_id) == 0:
        return 1
    else:
        pdb_id = pdb_id[0]
        return pdb_id


# read pdb sequence from pdb file
def read_seq_from_pdb(file_name_pdb):
    parser = PDBParser(PERMISSIVE=1)
    pdb_id = file_name_pdb[-8:-4]
    structure = parser.get_structure(pdb_id, file_name_pdb)
    models = structure.get_list()
    model_0 = models[0]
    chains = model_0.get_list()
    chain_0 = chains[0]
    residues = chain_0.get_list()
    pri_seq = []
    for residue in residues:
        if is_aa(residue):
            resname = residue.get_resname()
            resname = abbr_reduce(resname)
            pri_seq.append(resname)
    return pri_seq


# read a pdb-bmrb dict from csv file
def read_pdb_bmrb_dict_from_csv(csv_file):
    dict_pdb_bmrb = {}
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for (i, row) in enumerate(reader):
            if i > 0:
                dict_pdb_bmrb[row[0]] = row[1]
    return dict_pdb_bmrb


# find bmrb file name by its ID
def search_file_with_bmrb(bmrb_id, dir_nmrstar):
    if not os.path.exists(dir_nmrstar):
        print('where is dir of nmrstar files?')
        return 1
    list_all_files = os.listdir(dir_nmrstar)
    file_name = fnmatch.filter(list_all_files, '*{}*'.format(bmrb_id))
    if len(file_name) != 1:
        return 1
    else:
        file_name = file_name[0]
    return file_name


# read bmrb sequence from bmrb file
def read_seq_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star)
    seq_one_letter_code = entry.get_tag('Entity.Polymer_seq_one_letter_code')
    if len(seq_one_letter_code) == 0:
        return None
    else:
        aa_seq = seq_one_letter_code[0]
        aa_seq = aa_seq.replace('\n', '')
        aa_seq = aa_seq.replace('\r', '')
        aa_seq = list(aa_seq)
    return aa_seq


# read order parameter set from bmrb file
def read_s2_into_pd_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star, convert_data_types=True)
    s2_loops = entry.get_loops_by_category("Order_param")
    s2_loop = s2_loops[0]
    s2_set = s2_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Order_param_val'])
    pd_s2_set = pd.DataFrame.from_records(s2_set, columns=['Comp_index_ID', 'Comp_ID', 'Order_param_val'])
    return pd_s2_set


one_hot_code = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'R': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'C': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'E': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Q': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                'F': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


# residue class for s2 prediction
class residue_s2:

    def __init__(self, resname):

        self.name = resname

        self.s2 = -1.0

        self.one_hot = ''

    def set_s2(self, s2):
        self.s2 = s2

    def set_one_hot(self, code):
        self.one_hot = code


# protein class for s2 training and test (s2 has experimental value)
class protein_s2:

    def __init__(self, pdb_id='', bmrb_id='', index_start_pdb=0, index_start_bmrb=0, length_eff=0):
        self.pdb_id = pdb_id
        self.bmrb_id = bmrb_id
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff
        self.pdb_seq = []  # amino acid sequence from pdb file
        self.bmrb_seq = []  # sequence from bmrb file
        self.matched_seq = []  # overlapped part of pdb_seq and bmrb_seq

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq_train(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        seq_matcher = SequenceMatcher(None, pdb_seq, bmrb_seq)
        (index_start_pdb, index_start_bmrb, length_eff) = seq_matcher.find_longest_match(0, len(pdb_seq), 0,
                                                                                         len(bmrb_seq))
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

        for resname in bmrb_seq:
            residue_temp = residue_s2(resname)
            self.bmrb_seq.append(residue_temp)

        for index in range(index_start_pdb, (index_start_pdb + length_eff)):
            residue_temp = residue_s2(pdb_seq[index])
            self.matched_seq.append(residue_temp)

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        seq_matcher = SequenceMatcher(None, pdb_seq, bmrb_seq)
        (index_start_pdb, index_start_bmrb, length_eff) = seq_matcher.find_longest_match(0, len(pdb_seq), 0,
                                                                                         len(bmrb_seq))
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

        for resname in bmrb_seq:
            residue_temp = residue_s2(resname)
            self.bmrb_seq.append(residue_temp)

        for index in range(index_start_pdb, (index_start_pdb + length_eff)):
            residue_temp = residue_s2(pdb_seq[index])
            self.matched_seq.append(residue_temp)

    # read order parameter s2 from nmrstar file
    def read_s2_from_star(self, bmrb_file):
        pd_s2 = read_s2_into_pd_from_star(bmrb_file)
        for _, row in pd_s2.iterrows():
            index = row['Comp_index_ID'] - 1
            self.bmrb_seq[index].set_s2(float(row['Order_param_val']))

    def merge(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        s = SequenceMatcher(None, bmrb_seq, pdb_seq)
        seq_blocks = s.get_matching_blocks()
        for block in seq_blocks:
            (begin_seq_bmrb, begin_seq_pdb, size) = block
            if size != 0:
                for k in range(size):
                    self.pdb_seq[begin_seq_pdb + k].set_s2(self.bmrb_seq[begin_seq_bmrb + k].s2)

    # generate the matched sequence
    def merge_seq(self):
        for index in range(self.length_eff):

            self.matched_seq[index].one_hot = self.pdb_seq[index + self.index_start_pdb].one_hot

            self.matched_seq[index].s2 = self.bmrb_seq[index + self.index_start_bmrb].s2

    # to generate a 2D numpy array
    def to_numpy(self):
        list_data = []
        seq = []
        indexStart_length = [self.index_start_pdb, self.length_eff]
        complete_s2 = []
        for residue in self.matched_seq:
            complete_s2.append(residue.s2)
            if residue.s2 > -0.5:
                res_data = one_hot_code[residue.name] + [residue.s2]
                list_data.append(res_data)
                seq.append(residue.name)
        array_data = np.array(list_data)
        # seq_str = ''.join(seq)
        # with open('/home/amax/myprojects/end2end/complete_fasta_26/' + self.pdb_id + '.fasta', 'w') as f:
        #     f.writelines(['>'+self.pdb_id+'\n', seq_str])
        with open('/home/amax/myprojects/end2end/complete_s2/' + self.pdb_id + '.json', 'w') as f:
            json.dump(complete_s2, f)
        return array_data, indexStart_length


# protein class for s2 prediction (s2 has no experimental value)
class protein_s2_pre:

    def __init__(self, pdb_id='', length_eff=0):
        self.pdb_id = pdb_id
        self.length_eff = length_eff
        self.pdb_seq = []  # amino acid sequence from pdb file

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq(self, pdb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        length_eff = len(pdb_seq)
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

    # to generate a 2D numpy array
    def to_numpy(self, pdb_id, label_file_pseudo):
        # with open(label_file_pseudo + pdb_id + '.json') as obj:
        #     pseudo_label = json.load(obj)
        list_data = []
        for i, res in enumerate(self.pdb_seq):
            # if i < 3 or i > self.length_eff - 4:
            #     flag = 1
            # else:
            #     flag = 0
            feature = one_hot_code[res.name]
            # feature = one_hot_code[res.name] + [pseudo_label[i]]
            list_data.append(feature)
        array_data = np.array(list_data)
        return array_data



