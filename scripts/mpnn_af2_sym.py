"""
Run ColabDesign on symmetrical proteins
"""
import argparse
import fcntl
import logging
import math
import os
import time
from string import ascii_uppercase, ascii_lowercase

import numpy as np
import pandas as pd

from colabdesign.af import mk_af_model
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.rf.utils import fix_pdb


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sym_colabdesign.py")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("pdb", type=str, help="input pdb file path")
parser.add_argument("output_folder", type=str, help="output folder path")
parser.add_argument("contigs", type=str, help="RFdiffusion contigs")
parser.add_argument(
    "--num_seqs",
    type=int,
    default=80,
    help="number of sequences to generate (default: 80)",
)
parser.add_argument(
    "--sampling_temp",
    type=float,
    default=0.1,
    help="mpnn sampling temperature (default: 0.1)",
)
parser.add_argument(
    "--num_recycles",
    type=int,
    default=12,
    help="number of repacking cycles (default: 12)",
)
parser.add_argument(
    "--rm_aa", type=str, default="C", help="residue to remove from the design"
)
parser.add_argument(
    "--copies", type=int, default=1, help="number of repeating copies (default: 1)"
)
parser.add_argument(
    "--mpnn_batch", type=int, default=8, help="mpnn batch size (default: 8)"
)
parser.add_argument("--results_dataframe", type=str, help="save results")
parser.add_argument(
    "--save_best_only", action="store_true", help="save only the best structures"
)
parser.add_argument(
    "--initial_guess", action="store_true", help="use initial guess for alphafold2 validation"
)
parser.add_argument(
    "--use_multimer", action="store_true", help="use multimer weights for alphafold2 validation"
)
parser.add_argument(
    "--use_soluble", action="store_true", help="use soluble weights for mpnn"
)
args = parser.parse_args()

pdb = args.pdb
output_folder = args.output_folder
rf_contigs = args.contigs
num_seqs = args.num_seqs
sampling_temp = args.sampling_temp
mpnn_batch = args.mpnn_batch
num_recycles = args.num_recycles
rm_aa = args.rm_aa
copies = args.copies
results_dataframe = args.results_dataframe
save_best_only = args.save_best_only
initial_guess = args.initial_guess
use_multimer = args.use_multimer
use_soluble = args.use_soluble


### Functions ###
def write_df_to_csv(df, output_path):
    """Save/append dataframe to CSV."""
    mode, header = ("a", False) if os.path.isfile(output_path) else ("w", True)
    with open(output_path, mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        time.sleep(np.random.uniform(0, 0.05))
        df.to_csv(f, header=header, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)


def fix_pdb_file(file_path, contigs, outdir, ext="_fx"):
    """Fix pdb file based on contigs."""
    with open(file_path, "r") as f:
        pdb = fix_pdb(f.read(), contigs)
    out_file_path = os.path.join(outdir, os.path.basename(file_path).replace(".pdb", f"{ext}.pdb"))
    with open(out_file_path, "w") as f:
        f.write(pdb)
    return out_file_path


def get_info(contig):
    F = []
    free_chain = False
    fixed_chain = False
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n, (a, b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b) - int(a[1:]) + 1
            F += [1] * L
            fixed_chain = True
        else:
            L = int(b)
            F += [0] * L
            free_chain = True
    return F, [fixed_chain, free_chain]


def run_mpnn_sampling(pdb, mpnn_model, num_samples, batch_size, temperature, target_len):
    """Run MPNN sampling to generate protein sequences."""
    mpnn_model.prep_inputs(  # Only design binder chains
        pdb_filename=pdb,
        chain=",".join(alphabet_list[:copies]),
        homooligomer=copies > 1,
        rm_aa=rm_aa,
        verbose=True,
    )
    # Fix target position
    fxpos = np.arange(target_len)
    mpnn_model._inputs["fix_pos"] = fxpos
    mpnn_model._inputs["bias"][fxpos] = (
        1e7 * np.eye(21)[mpnn_model._inputs["S"]][fxpos, :20]
    )
    # Sample sequences
    mpnn_out = mpnn_model.sample(
        num=num_samples // batch_size,
        batch=batch_size,
        temperature=temperature,
    )
    for k in af_terms:
        mpnn_out[k] = []
    for term in other_terms:
        if term not in mpnn_out:
            mpnn_out[term] = []
    return mpnn_out


def run_af2_predictions(mpnn_out, af_model, num_seqs, num_recycles, af_terms, output_path, pdb_name, pdb, binder_len):
    """Run AF2 predictions on MPNN sampled sequences."""
    for n in range(num_seqs):
        seq = mpnn_out["seq"][n][-binder_len:]*copies
        logger.info(f"Running AF2 predictions for sequence {n+1}/{num_seqs}...")
        logger.debug(f"Predicting sequence: {seq}, with length {len(seq)}")
        af_model.predict(seq=seq, num_recycles=num_recycles, num_models=1, verbose=False)

        for t in af_terms:
            mpnn_out[t].append(af_model.aux["log"][t])
        if "i_pae" in mpnn_out:
            mpnn_out["i_pae"][-1] *= 31
        if "pae" in mpnn_out:
            mpnn_out["pae"][-1] *= 31

        current_model_path = f"{output_path}/{pdb_name}_{n}.pdb"
        mpnn_out["model_path"].append(current_model_path)
        mpnn_out["input_pdb"].append(pdb)

        if not save_best_only or (mpnn_out["plddt"][n] > 0.7 and mpnn_out["rmsd"][n] < 3):
            af_model.save_current_pdb(current_model_path)

        af_model._save_results(save_best=save_best_only, verbose=False)
        af_model._k += 1
    return mpnn_out


### Checks ###
if num_seqs < mpnn_batch:
    mpnn_batch = num_seqs
    logger.warning(f"num_seqs must be greater than or equal to mpnn_batch. Setting mpnn_batch to {mpnn_batch}")
elif num_seqs % mpnn_batch != 0:
    mpnn_batch = math.gcd(num_seqs, mpnn_batch)
    logger.warning(f"num_seqs must be divisible by mpnn_batch. Setting mpnn_batch to {mpnn_batch}")

if rm_aa == "":
    rm_aa = None

### Parse contigs
contigs = []
for contig_str in rf_contigs.replace(" ",":").replace(",",":").split(":"):
    if len(contig_str) > 0:
        contig = []
        for x in contig_str.split("/"):
            if x != "0":
                contig.append(x)
        contigs.append("/".join(contig))

alphabet_list = list(ascii_uppercase + ascii_lowercase)
chains = alphabet_list[:len(contigs)]
info = [get_info(x) for x in contigs]
fixed_pos = []
fixed_chains = []
free_chains = []
both_chains = []
for pos,(fixed_chain,free_chain) in info:
    fixed_pos += pos
    fixed_chains += [fixed_chain and not free_chain]
    free_chains += [free_chain and not fixed_chain]
    both_chains += [fixed_chain and free_chain]

target_chains = []
binder_chains = []
for n,x in enumerate(fixed_chains):
    if x: target_chains.append(chains[n])
    else: binder_chains.append(chains[n])
logger.debug(f"Target chains: {target_chains}")
logger.debug(f"Binder chains: {binder_chains}")

### Initializations ###
logger.info("Initializing symmetrical protein design...")
pdb_basename = pdb.split("/")[-1].split(".pdb")[0]

chain_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chain_list = [x for x in chain_list]
mpnn_terms = ["score", "seq"]
af_terms = ["plddt", "i_ptm", "i_pae", "rmsd"]
other_terms = ["model_path", "input_pdb"]
labels = ["score"] + af_terms + other_terms + ["seq"]

af_model = mk_af_model(
    protocol="binder",
    initial_guess=initial_guess,
    best_metric="rmsd",
    use_multimer=use_multimer,
    data_dir="/home/tsatler/projects/AFdesign_playground",
    model_names=["model_1_multimer_v3" if use_multimer else "model_1_ptm"],
)
mpnn_model = mk_mpnn_model(weights="soluble" if use_soluble else "original")


### Fix pdb by separating chains
fix_pdb_dir=f"{output_folder}/fix_pdb"
os.makedirs(fix_pdb_dir, exist_ok=True)
pdb_fix = fix_pdb_file(pdb, contigs, fix_pdb_dir, ext="_fx")
logger.info(f"Fixed pdb file saved to {pdb}")


### Run MPNN sampling and AF2 predictions
logger.info("Running MPNN sampling and AF2 predictions...")
af_model.prep_inputs(
    pdb_fix,
    target_chain=",".join(target_chains),
    binder_chain=",".join(binder_chains),
    rm_aa=rm_aa,
    copies=copies,
    homooligomer=copies > 1,
)
target_len = int(af_model._target_len / len(target_chains)) # Target length for each chain
binder_len = int(af_model._binder_len / len(binder_chains)) # Binder length for each chain
mpnn_out = run_mpnn_sampling(pdb, mpnn_model, num_seqs, mpnn_batch, sampling_temp, target_len)

logger.info("Running AF2 predictions...")
af2_out = run_af2_predictions(mpnn_out, af_model, num_seqs, num_recycles, af_terms, output_folder, pdb_basename, pdb, binder_len)

# Generate model paths for all sequences
model_paths = [f"{output_folder}/{pdb_basename}_{n}.pdb" for n in range(num_seqs)]
all_labels = mpnn_terms + af_terms + other_terms
data = [[af2_out[label][n] for label in all_labels] for n in range(num_seqs)]
all_labels[0] = "mpnn"


### Save data to CSV ###
logger.info("Saving results to CSV...")
df = pd.DataFrame(data, columns=all_labels)

# Write df to CSV
output_path_all = (
    f"{results_dataframe}/af2_results_all.csv"
    if results_dataframe
    else f"{output_folder}/af2_results_all.csv"
)
write_df_to_csv(df, output_path_all)

# Filter df for best results and write to CSV
df_best = df[(df["rmsd"] < 3) & (df["plddt"] > 0.7)]
output_path_best = (
    f"{results_dataframe}/af2_best.csv"
    if results_dataframe
    else f"{output_folder}/af2_best.csv"
)
write_df_to_csv(df_best, output_path_best)
