import numpy as np
import dill
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import pandas as pd

import utils
from modules import BiLSTM


torch.cuda.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_regression_model(
    weight_path: str,
    input_dim: int = 1310,
    hidden_dim: int = 1024,
    num_layers: int = 2,
) -> BiLSTM:
    model = BiLSTM(input_dim, hidden_dim, num_layers).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_protein_s2(model: BiLSTM, pdb_id: str,
                       csv_file: str, dir_nmrstar: str,
                       embedding_dir: str):
    pdb_file = f"./pdb_test/{pdb_id}.pdb"

    dict_pdb_bmrb = utils.read_pdb_bmrb_dict_from_csv(csv_file)
    bmrb_id = dict_pdb_bmrb[pdb_id.upper()]
    bmrb_file = utils.search_file_with_bmrb(bmrb_id, dir_nmrstar)
    bmrb_file_full_path = dir_nmrstar + bmrb_file

    protein_temp = utils.protein_s2(pdb_id, bmrb_id)
    protein_temp.read_seq(pdb_file, bmrb_file_full_path)
    protein_temp.read_s2_from_star(bmrb_file_full_path)
    protein_temp.merge(pdb_file, bmrb_file_full_path)

    with open(f"./{embedding_dir}/{pdb_id.lower()}.dat", "rb") as f:
        seq_embedding = dill.load(f)

    s2_pred, s2_label = [], []
    with torch.no_grad():
        for i, res in enumerate(protein_temp.pdb_seq):
            if res.s2 < -0.5:
                continue

            s2_label.append(res.s2)

            feature = np.array(seq_embedding[i], dtype=np.float32)[np.newaxis, :]
            feat_tensor = torch.from_numpy(feature)
            feat_tensor = feat_tensor.unsqueeze(1).to(device)

            pred = model(feat_tensor)[0]
            s2_pred.append(pred.cpu())

    s2_pred = [t.item() for t in s2_pred]
    if len(s2_pred) > 2:
        s2_pred_smooth = [0.0] * len(s2_pred)
        for i in range(1, len(s2_pred) - 1):
            s2_pred_smooth[i] = (
                s2_pred[i - 1] + s2_pred[i] + s2_pred[i + 1]
            ) / 3.0
        s2_pred_smooth[0] = s2_pred[0]
        s2_pred_smooth[-1] = s2_pred[-1]
    else:
        s2_pred_smooth = s2_pred

    s2_label_np = np.array(s2_label, dtype=np.float32)
    s2_pred_np = np.array(s2_pred_smooth, dtype=np.float32)

    return s2_label_np, s2_pred_np


def evaluate_on_testset():
    csv_file = "pdb_bmrb_pair.csv"
    dir_nmrstar = "./nmrstar/"
    embedding_dir = "./esm_hmm_10/"

    pdb_ids = ['1pd7', '1wrs', '1wrt', '1z9b', '2jwt',
               '2l6b', '2luo', '2m3o', '2xdi', '4aai']

    model = load_regression_model("regression_model.pth")

    pdb_list = []
    pre_list, real_list = [], []
    pcc_list, scc_list, mae_list, rmse_list = [], [], [], []

    for pdb_id in pdb_ids:
        print(f"Predicting S² for {pdb_id.upper()} ...")
        s2_label, s2_pred = predict_protein_s2(
            model, pdb_id, csv_file, dir_nmrstar, embedding_dir
        )

        rmse = np.sqrt(np.mean((s2_label - s2_pred) ** 2))
        mae = mean_absolute_error(s2_label, s2_pred)
        pcc = pearsonr(s2_label, s2_pred)[0]
        scc = spearmanr(s2_label, s2_pred)[0]

        print(f"PDB_ID={pdb_id.upper()}, "
              f"PCC={pcc:.3f}, SCC={scc:.3f}, "
              f"MAE={mae:.3f}, RMSE={rmse:.3f}")

        pdb_list.append(pdb_id.upper())
        pre_list.append(s2_pred.tolist())
        real_list.append(s2_label.tolist())
        pcc_list.append(pcc)
        scc_list.append(scc)
        mae_list.append(mae)
        rmse_list.append(rmse)

    results_save_path = "results.csv"
    df = pd.DataFrame({
        'PDB': pdb_list,
        'pre': pre_list,
        'real': real_list,
        'PCC': pcc_list,
        'SCC': scc_list,
        'MAE': mae_list,
        'RMSE': rmse_list
    })
    df.to_csv(results_save_path, index=False)
    print(f"Results saved to: {results_save_path}")

    print("meanPCC={:.3f}, meanSCC={:.3f}, meanMAE={:.3f}, meanRMSE={:.3f}".format(
        np.mean(pcc_list), np.mean(scc_list),
        np.mean(mae_list), np.mean(rmse_list)
    ))


if __name__ == "__main__":
    evaluate_on_testset()
