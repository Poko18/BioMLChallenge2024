import os
import logging
import time
import argparse

from colabdesign import clear_mem
from colabdesign.af import mk_af_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the argument parser
parser = argparse.ArgumentParser(description="Run AlphaFold model on PDB file.")
parser.add_argument("input_pdb", type=str, help="Path to the input PDB file.")
parser.add_argument("--num_recycles", type=int, default=1, help="Number of recycling steps.")
parser.add_argument("--num_models", type=int, default=5, help="Number of models to use.")
parser.add_argument("--recycle_mode", type=str, default="sample", choices=["sample", "greedy"], help="Recycle mode.")
parser.add_argument("--target_chain", type=str, default="B", help="Chain identifier for the target.")
parser.add_argument("--target_flexible", action="store_true", help="Whether the target sequence is flexible.")
parser.add_argument("--binder_chain", type=str, default="A", help="Chain identifier for the binder.")
parser.add_argument("--soft_iters", type=int, default=120, help="Number of soft iterations.")
parser.add_argument("--hard_iters", type=int, default=32, help="Number of hard iterations.")
parser.add_argument("--output_file", type=str, default="output.pdb", help="Output PDB file name.")
parser.add_argument("--data_dir", type=str, help="Directory for model data.")

args = parser.parse_args()
input_pdb = args.input_pdb
num_recycles = args.num_recycles
num_models = args.num_models
recycle_mode = args.recycle_mode
target_chain = args.target_chain
target_flexible = args.target_flexible
binder_chain = args.binder_chain
soft_iters = args.soft_iters
hard_iters = args.hard_iters
output_file = args.output_file
data_dir = args.data_dir

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Start timing
t0 = time.time()

clear_mem()
af_model = mk_af_model(protocol="binder",
                       use_multimer=True,
                       recycle_mode=recycle_mode,
                       num_recycles=num_recycles,
                       data_dir=data_dir,
                       )

af_model.prep_inputs(
    pdb_filename=input_pdb,
    chain=target_chain, # Target chain
    binder_chain=binder_chain, # Binder chain
    use_multimer=True,
    rm_target_seq=target_flexible,
)

af_model.set_weights(
    plddt=0.1, dgram_cce=1.0, rmsd=0.01 # , i_pae=0.2
)
logging.info("Weights: %s", af_model.opt["weights"])

af_model.set_optimizer(optimizer="sgd",
                       learning_rate=0.1,
                       norm_seq_grad=True)

models = af_model._model_names[:num_models]
# Hardcode semi-greedy design
af_model.design_pssm_semigreedy(
    soft_iters,
    hard_iters,
    dropout=True,
    num_recycles=num_recycles,
    models=models,
)

# Save output PDB
af_model.save_pdb(output_file)
logging.info("Saved output PDB to: %s", output_file)
logging.info("Time elapsed: %s", time.time() - t0)
