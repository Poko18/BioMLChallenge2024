# BioML Challenge 2024 - Protein Binders for hCD20 🧬

## Overview

This repository is dedicated to the **BioML Challenge 2024: Bits to Binders** competition, organized by the University of Texas at Austin BioML Society. The aim of this competition is to advance protein design using modern AI tools. Our task is to design a protein sequence that effectively binds to a specific cancer antigen target, with the goal of activating CAR-T cell proliferation and killing responses.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation-%F0%9F%9B%A0%EF%B8%8F)
3. [Workflow Overview](#workflow-overview)
   - [Binder Blueprints](#binder-blueprints)
   - [RFdiffusion of Protein Binders](#rfdiffusion-of-protein-binders)
   - [ProteinMPNN and AlphaFold2 Validation](#proteinmpnn-and-alphafold2-validation)
   - [Binder (backbone) redesign](#binder-backbone-redesign-optional)
   - [Partial Diffusion Refinement](#partial-diffusion-refinement-optional)
   - [Pyrosetta Filtering](#pyrosetta-filtering)
4. [License](#license)
5. [Get in Touch](#get-in-touch)

## Installation 🛠️

To run the workflow, you need to install [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), [ColabDesign](https://github.com/sokrypton/ColabDesign), and [pyrosetta](https://pypi.org/project/pyrosetta-installer/).

## Workflow Overview

Here's a step-by-step breakdown of the workflow used to design and validate the protein binders:

### Binder Blueprints

We were constrained to designing binders with a maximum length of 80 amino acids (AA). While smaller binders could be extended with linkers, we aimed to engineer larger binders at the full 80 AA, giving us more flexibility and space for engineering.

To achieve this, we developed a `BinderBlueprint` generator to create different adjacency matrices for binders of a defined length and secondary structural elements (SSEs). These matrices are later used with RFdiffusion for scaffold-guided diffusion of binders towards desired target hotspots.

Follow the example below to generate binder blueprints, or run the provided script `scripts/binder_blueprints.py` to generate binder blueprints for different scaffolds - binders with different structural elements.

If you want to provide just the size of the binder (with the length of beta strands being half that of helices):

```python
blueprint = BinderBlueprints(
        elements=["HHH"], # Three helices
        size=80,
    )
blueprint.save_adj_sse(output_dir=OUTPUT_DIR)
```

You can also specify the sizes of the individual secondary elements and linkers:

```python
blueprint = BinderBlueprints(
        elements=["HHH"], # Three helices
        size=80,
        element_lengths=[25,25,24], # Lengths of each helix
        linker_lengths=[3,3], # Lengths of the linkers between the helices
    )
blueprint.save_adj_sse(output_dir=OUTPUT_DIR)
```

This will generate secondary structure (`ss.pt`) and adjacency matrices (`adj.pt`) for the binders, which can be used as input for scaffold-guided diffusion.

To use our pre-generated binder blueprints, extract them with the following command:

```bash
tar -xzvf data/binder_blueprints.tar.gz
```

---

### RFdiffusion of protein binders

We employed several approaches to design the protein binders, leveraging various RFdiffusion scripts targeting different regions of the hCD20 antigen:

Targeting a single monomer (modeled as a dimer)
Targeting the middle of hCD20
Targeting the entire dimer with random binder scaffolds
Preparing symmetric binders to leverage target symmetry

We shorten the hCD20 protein model to only the part that is positioned on the outer site of membrane - contigs: `D65-87/D131-198/0 C65-87/C131-198/0`. This is so that computations will be quicker. We also need to generate `ss.pt` and `adj.pt` files for RFdiffusion, easiest way is to use `make_secstruc_adj.py` from [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) repository.

#### Targeting Specific Hotspots

The only difference between the first two approaches is the choice of diffusion hotspots:

Targeting a single hCD20 monomer: `[D76,D160,D161,D163,D166,D170,C177,C178,C182]`
Targeting the middle region of hCD20: `[D77,D161,D163,D166,D171,D174,D178,C77,C161,C163,C166,C171,C174,C178]`

```bash
python /RFdiffusion_path/run_inference.py \
    diffuser.T=50 \
    inference.output_prefix={output} \
    scaffoldguided.target_path={target} \
    scaffoldguided.scaffoldguided=True \
    ppi.hotspot_res=[{hotspots}] \
    scaffoldguided.target_pdb=True \
    scaffoldguided.target_ss={ss} \
    scaffoldguided.target_adj={adj} \
    scaffoldguided.scaffold_dir={scaffolds} \
    potentials.guiding_potentials=[ \
        "type:binder_ROG,weight:3", \
        "type:interface_ncontacts", \
        "type:binder_distance_ReLU" \
    ] \
    potentials.guide_scale=2 \
    potentials.guide_decay="quadratic" \
    inference.num_designs={num_of_diffusions} \
    denoiser.noise_scale_ca=0 \
    denoiser.noise_scale_frame=0
```

#### Targeting the Entire Dimer with Random Binder Scaffolds

For targeting the entire hCD20 dimer, we use the same hotspot residues but specify the contigs, as we are not doing scaffold guided diffusion.
Hotspots: `[D77,D161,D163,D166,D171,D174,D178,C77,C161,C163,C166,C171,C174,C178]`
Contigs: `D65-87/D131-198/0 C65-87/C131-198/0 80-80`

```bash
python /RFdiffusion_path/run_inference.py \
    diffuser.T=50 \
    inference.output_prefix={output} \
    inference.input_pdb={target} \
    contigmap.contigs=[{contigs}] \
    ppi.hotspot_res=[{hotspots}] \
    potentials.guiding_potentials=[ \
        "type:binder_ROG,weight:3", \
        "type:interface_ncontacts", \
        "type:binder_distance_ReLU" \
    ] \
    potentials.guide_scale=2 \
    potentials.guide_decay="quadratic" \
    inference.num_designs={num_of_diffusions}
```

#### Designing Symmetric Binders

To exploit the `C2` symmetry of hCD20, we generated symmetric binders. This required a small modification to RFdiffusion to allow for [non-receptor hotspots](https://github.com/RosettaCommons/RFdiffusion/commit/642e3643dcde2f7d70d6847f82cf889f29adf7d4).
The contigs specify the symmetry relations as: `A65-87/A131-198 80-80/0 A65-87/A131-198 80-80/0`. For symmetric designs, we adjusted the potentials to avoid designs that did not interact properly, often using lower contact weights (`"type:olig_contacts,weight_intra:0.1,weight_inter:0.2"`) or even skipping guiding potentials.

```bash
python /RFdiffusion_path/run_inference.py \
    --config-name=symmetry \
    diffuser.T=20\
    inference.input_pdb={target} \
    inference.num_designs={num_of_diffusions} \
    inference.symmetry="C2" \
    inference.ckpt_override_path=/RFdiffusion_path/models/Complex_beta_ckpt.pt \
    inference.output_prefix={output} \
    contigmap.contigs=[{contigs}] \
    ppi.hotspot_res=[{hotspots}] \
    potentials.guiding_potentials=[{guiding_potentials}] \
    potentials.olig_intra_all=True \
    potentials.olig_inter_all=True \
    potentials.guide_scale=2 \
    potentials.guide_decay="quadratic"
```

---

### ProteinMPNN and AlphaFold2 validation

To validate the generated protein binders, we employed a two-step process combining **ProteinMPNN** for sequence design and **AlphaFold2** for structural validation.

For non-symmetric binders, use the provided script `scripts/mpnn_af2.py`, which integrates ProteinMPNN for sequence generation and AlphaFold2 for structural validation:

```bash
python scripts/mpnn_af2.py {pdb} {output} {target_chain} {binder_chain} \
    --sampling_temp {sampling_temp} \
    --num_recycles {num_recycles} \
    --num_seqs {num_seqs} \
    --num_filt_seq {num_filtered_seqs} \
    --results_dataframe {output_folder} \
    --save_best_only
```

For symmetric binders, use the script `scripts/mpnn_af2_sym.py`, which is adapted for symmetric designs:

```bash
python scripts/mpnn_af2_sym.py {pdb} {output} {contig} \
  --num_seqs {num_seqs} \
  --sampling_temp {sampling_temp} \
  --num_recycles {num_recycles} \
  --rm_aa="C" \
  --mpnn_batch {mpnn_batch} \
  --results_dataframe {output_dir}/sym_diff \
  --fix_pdb --save_best_only --initial_guess \
  --use_multimer --use_soluble
```

---

### Binder (backbone) redesign [Optional]

To explore variations in binder design, we used the `scripts/binder_redesign.py` script, which utilizes the **ColabDesign** library. This script allows for the refinement of protein binders or the generation of binder scaffold variants that can be used as input for **partial diffusion**.

```bash
python /scripts/binder_redesign.py \
    input_pdb={pdb} \
    --num_recycles={number_of_recycles} \
    --num_models={number_of_models} \
    --recycle_mode={recycle_mode_choice} \
    --target_chain={target_chain} \
    --target_flexible \
    --binder_chain={binder_chain} \
    --soft_iters={soft_iterations} \
    --hard_iters={hard_iterations} \
    --output_file={output_pdb} \
    --data_dir={model_data_directory} # path to alphafold2 params
```

---

### Partial diffusion refinement [Optional]

To further explore the structural space and related binder scaffolds, we applied **partial diffusion** to the best designed binders. This allowed us generate variations of the original designs.

```bash
python /RFdiffusion_path/run_inference.py \
    inference.input_pdb={pdb} \
    contigmap.contigs=[{contigs}] \
    inference.num_designs={number_of_diffusions} \
    diffuser.partial_T={steps}
```

These newly refined designs were subsequently validated using [ProteinMPNN and AlphaFold2](#proteinmpnn-and-alphafold2-validation).

---

### Pyrosetta filtering

For the final selection of binders, we used **PyRosetta** to compute a range of structural and energetic metrics, helping us identify the most promising candidates. The script `scripts/pyrosetta_metrics.py` generates these metrics, with key ones including `ddg`, `rg`, `charge`, `sap`, `shape_comp`, `cms`, `ddg_cms`, and `vbuns`.

```bash
python scripts/pyrosetta_metrics.py {pdb} {target_chain} {binder_chain} {output_dataframe} {xml_file}
```

> **_NOTE:_** The output dataframe from AlphaFold2 validation can be reused, with new values appended for each protein.

---

## License

This project is open-source and available under the MIT License. Feel free to explore, modify, and build upon it. For more details, see the [LICENSE](LICENSE) file.

## Get in Touch

For questions, suggestions, or collaborations, feel free to reach out tadej.satler@gmail.com
