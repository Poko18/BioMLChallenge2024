import argparse
import fcntl
import logging
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import pyrosetta
from Bio.PDB import PDBParser, cealign, Chain
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn, rosetta, Vector1
from pyrosetta.rosetta import *
from pyrosetta.rosetta.protocols import analysis, docking, rigid, rosetta_scripts
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

from utils import parse_pdb

# Initialize logging and pyrosetta
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
init("-beta_nov16 -holes:dalphaball")

parser = argparse.ArgumentParser(description="Binder analysis")
parser.add_argument("input_pdb", type=str, help="Path to input PDB files")
parser.add_argument("target_chain", type=str, help="name of target chain (C,D..)")
parser.add_argument("binder_chain", type=str, help="name of binder chain (C,D..)")
parser.add_argument("output_df", type=str, help="Output csv file for adding metrics")
parser.add_argument("xml_file", type=str, help="Path to rosetta XML file")
args = parser.parse_args()

input_pdb = args.input_pdb
target_chain = args.target_chain
binder_chain = args.binder_chain
output_df = args.output_df
xml = args.xml_file
objs = rosetta_scripts.XmlObjects.create_from_file(xml)


### Functions ###

def unbind(pose, partners: str) -> None:
    """Applies rigid body translation to unbind the partners from the pose."""
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose, partners, Vector1([-1, -1, -1]))
    trans_mover = rigid.RigidBodyTransMover(pose, JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose)

def calculate_ddg(pose, partners: str, relax: bool = True) -> float:
    """Calculates the delta delta G (ddG) of binding for the given pose."""
    if relax:
        relax_pose(pose)
    relaxPose = pose.clone()
    scorefxn = get_fa_scorefxn()
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 0)
    bound_score = scorefxn(relaxPose)
    unbind(relaxPose, partners)
    unbound_score = scorefxn(relaxPose)
    return round((bound_score - unbound_score), 3)

def align_structures(pdb1: str, pdb2: str) -> float:
    """Superimposes pdb1 on pdb2 and returns the RMSD."""
    pdb_parser = PDBParser(QUIET=True)
    ref_structure = pdb_parser.get_structure("ref", pdb1)
    sample_structure = pdb_parser.get_structure("sample", pdb2)
    aligner = cealign.CEAligner()
    aligner.set_reference(ref_structure)
    aligner.align(sample_structure)
    return aligner.rms

def get_sasa(pose, probe_radius: float = 1.4) -> Tuple[float, float]:
    """Calculates the total and hydrophobic solvent accessible surface area (SASA) of the pose."""
    rsd_sasa = pyrosetta.rosetta.utility.vector1_double()
    rsd_hydrophobic_sasa = pyrosetta.rosetta.utility.vector1_double()
    rosetta.core.scoring.calc_per_res_hydrophobic_sasa(
        pose, rsd_sasa, rsd_hydrophobic_sasa, probe_radius
    )
    return sum(rsd_sasa), sum(rsd_hydrophobic_sasa)

def calculate_charge(chain: Chain, ph: float = 7.4) -> float:
    """Calculates the protein chain's charge at a specific pH."""
    sequence = "".join(
        aa1[aa3.index(residue.get_resname())]
        for residue in chain.get_residues()
        if residue.get_resname() in aa3
    )
    return ProteinAnalysis(sequence).charge_at_pH(ph)

def calculate_sap_score(pose, chain: str = "B") -> float:
    """Calculates the SAP score for a given chain of the pose."""
    select_chain = XmlObjects.static_get_mover(
        f'<SwitchChainOrder name="so" chain_order="{chain}"/>'
    )
    chain_pose = pose.clone()
    select_chain.apply(chain_pose)
    sap_score_metric = XmlObjects.static_get_simple_metric(
        '<SapScoreMetric name="sap_metric"/>'
    )
    sap_score_value = sap_score_metric.calculate(chain_pose)
    return sap_score_value

def calculate_rg(chain: Chain) -> float:
    """Calculates the radius of gyration (Rg) of a protein chain using only the alpha carbons (CA)."""
    ca_atoms = [atom for atom in chain.get_atoms() if atom.get_name() == "CA"]
    ca_coords = np.array([atom.get_coord() for atom in ca_atoms])
    return np.sqrt(
        np.mean(np.sum((ca_coords - np.mean(ca_coords, axis=0)) ** 2, axis=-1)) + 1e-8
    )

def get_binder_ca_xyz(pdb_file: str, binder_chain: str = "A") -> np.ndarray:
    """Extracts the CA atom coordinates for the binder chain from the PDB file."""
    pdb = parse_pdb(pdb_file)
    ca_xyz = pdb["xyz"][:, 1, :]
    chain_mask = np.array(
        [chain_id == binder_chain for chain_id, res_num in pdb["pdb_idx"]]
    )
    binder_xyz = ca_xyz[chain_mask]
    return binder_xyz

def max_distance_between_atoms(coords: np.ndarray) -> float:
    """Computes the maximum distance between any two atoms."""
    dists = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    return np.max(dists)

def interface_terms(pdb: str) -> Tuple[float, float, float, int, int]:
    """Analyzes the interface terms for a given PDB file."""
    pose = pose_from_pdb(pdb)
    interface_analyzer = analysis.InterfaceAnalyzerMover()
    interface_analyzer.apply(pose)
    data = interface_analyzer.get_all_data()
    return (
        data.dG[1],  # dG (delta G) energy value of the interface
        data.dSASA[1], # dSASA (delta Solvent Accessible Surface Area) value of the interface
        (data.dG_dSASA_ratio * 100), # dG to dSASA ratio multiplied by 100
        data.delta_unsat_hbonds, # number of delta unsatisfied hydrogen bonds at the interface
        data.interface_hbonds, # number of hydrogen bonds formed at
    )

def relax_pose(pose, binder_chain: str = "A"):
    """Applies relaxation to the pose based on the binder chain."""
    if binder_chain == "A":
        fastrelax = objs.get_mover("FastRelax")
        fastrelax.apply(pose)
    elif binder_chain == "B":
        fastrelax = objs.get_mover("FastRelax_ChainB_bbtrue")
        fastrelax.apply(pose)
    return pose

def get_ddg(pose, relax: bool = True) -> float:
    """Retrieves the ΔΔG value after applying relaxation if specified (from xml file)."""
    if relax:
        relax_pose(pose)
    ddg = objs.get_filter("ddg")
    ddg.apply(pose)
    return ddg.score(pose)

def shape_complementarity(pose) -> float:
    """Calculates the shape complementarity score for the pose."""
    shape_comp = objs.get_filter("interface_sc")
    shape_comp.apply(pose)
    return shape_comp.score(pose)

def interface_buried_sasa(pose):
    """Calculates the buried SASA at the interface for the pose."""
    dsasa = objs.get_filter("interface_buried_sasa")
    dsasa.apply(pose)
    return dsasa.score(pose)

def hydrophobic_residue_contacts(pose):
    """Calculates the number of hydrophobic residue contacts."""
    hyd_res = objs.get_filter("hydrophobic_residue_contacts")
    hyd_res.apply(pose)
    return hyd_res.score(pose)

def get_cms(pose):
    """Retrieves the CMS score for the pose."""
    cms = objs.get_filter("cms")
    cms.apply(pose)
    return cms.score(pose)

def get_vbuns(pose):
    """Retrieves the very buried unsatisfied bonds score from the pose."""
    vbuns = objs.get_filter("vbuns")
    vbuns.apply(pose)
    return vbuns.score(pose)

def get_sbuns(pose):
    """Retrieves the buried unsatisfied bonds score from the pose."""
    sbuns = objs.get_filter("sbuns")
    sbuns.apply(pose)
    return sbuns.score(pose)

def interface_vbuns(pose, partners: str) -> Tuple[float, float]:
    """Calculates the vbuns score for bound and unbound states."""
    bound_vbuns = get_vbuns(pose)
    unboundpose = pose.clone()
    unbind(unboundpose, partners)
    unbound_vbuns = get_vbuns(unboundpose)
    return round(bound_vbuns, 3), round(unbound_vbuns, 3)

def interface_sbuns(pose, partners: str) -> Tuple[float, float]:
    """Calculates the sbuns score for bound and unbound states."""
    bound_sbuns = get_sbuns(pose)
    unboundpose = pose.clone()
    unbind(unboundpose, partners)
    unbound_sbuns = get_sbuns(unboundpose)
    return round(bound_sbuns, 3), round(unbound_sbuns, 3)

def write_df_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write the DataFrame to a CSV file, with file locking and optional header."""
    mode = "a" if os.path.isfile(output_path) else "w"
    with open(output_path, mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # lock the file for exclusive access
        rng = np.random.default_rng()
        time.sleep(rng.uniform(0, 0.1))
        df.to_csv(f, header=(mode == "w"), index=False)
        fcntl.flock(f, fcntl.LOCK_UN)  # release the lock

##################


### Metric calculations ###

pdb = input_pdb
pose = pose_from_pdb(input_pdb)
parser = PDBParser()
structure = parser.get_structure("protein", pdb)
chain = structure[0][binder_chain]

rg = calculate_rg(chain)
charge = calculate_charge(chain, ph=7.4)
sap = calculate_sap_score(pose, binder_chain)
dG, dSASA, dG_dSASA_ratio, int_unsat_hbonds, int_hbonds = interface_terms(pdb)
hyd_con = hydrophobic_residue_contacts(pose)
shape_comp = shape_complementarity(pose)
interface_sasa = interface_buried_sasa(pose)
rpose = relax_pose(pose, binder_chain)
ddg = calculate_ddg(
    rpose, partners=f"{binder_chain}_{''.join(target_chain.split(','))}", relax=False
)
ddg_score = get_ddg(rpose, relax=False)
ddg_dsasa_100 = (ddg / interface_sasa) * 100
ddgscore_dsasa_100 = (ddg_score / interface_sasa) * 100
cms = get_cms(pose)
ddg_cms = (ddg / cms)
vbuns_bound, vbuns_unbound = interface_vbuns(
    rpose, partners=f"{binder_chain}_{''.join(target_chain.split(','))}"
)
vbuns_int = vbuns_bound - vbuns_unbound
sbuns_bound, sbuns_unbound = interface_sbuns(
    rpose, partners=f"{binder_chain}_{''.join(target_chain.split(','))}"
)
sbuns_int = sbuns_bound - sbuns_unbound
max_binder_distance = max_distance_between_atoms(get_binder_ca_xyz(pdb, binder_chain))

# Important metrics: ddg, rg, charge, sap, int_unsat_hbonds, int_hbonds, hyd_contacts,
# shape_comp, ddg_dsasa_100, cms, ddg_cms_100, vbuns_int, sbuns_int, max_binder_distance
data = {
    "ddg": ddg,
    "rg": rg,
    "charge": charge,
    "sap": sap,
    # "dG": dG,
    # "dSASA": dSASA,
    # "dG_dSASA_ratio": dG_dSASA_ratio,
    "int_unsat_hbonds": int_unsat_hbonds,
    "int_hbonds": int_hbonds,
    "hyd_contacts": hyd_con,
    "shape_comp": shape_comp,
    # "ddg_score": ddg_score,
    "ddg_dsasa_100": ddg_dsasa_100,
    # "ddgscore_dsasa_100": ddgscore_dsasa_100,
    "cms": cms,
    "ddg_cms": ddg_cms,
    # "vbuns_bound": vbuns_bound,
    # "vbuns_unbound": vbuns_unbound,
    "vbuns_int": vbuns_int,
    # "sbuns_bound": sbuns_bound,
    # "sbuns_unbound": sbuns_unbound,
    "sbuns_int": sbuns_int,
    "max_binder_distance": max_binder_distance,
}

# Convert dictionary to DataFrame
df = pd.DataFrame([data])
write_df_to_csv(df, output_df)
