# BioML Challenge 2024 - Protein Binders for hCD20 🧬

## Overview

This repository is dedicated to the **BioML Challenge 2024: Bits to Binders** competition, organized by the University of Texas at Austin BioML Society. The aim of this competition is to advance protein design using modern AI tools. Our task is to design a protein sequence that effectively binds to a specific cancer antigen target, with the goal of activating CAR-T cell proliferation and killing responses.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation-%F0%9F%9B%A0%EF%B8%8F)
3. [Workflow Overview](#workflow-overview)
   - [Binder Blueprints](#binder-blueprints)
   - [RFdiffusion of Protein Binders](#rfdiffusion-of-protein-binders)
4. [License](#license)
5. [Get in Touch](#get-in-touch)

## Installation 🛠️

To run the workflow, you need to install [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), [ColabDesign](https://github.com/sokrypton/ColabDesign), and [pyrosetta](https://pypi.org/project/pyrosetta-installer/).

## Workflow Overview

Here's a step-by-step breakdown of the workflow used to design and validate the protein binders:

### Binder Blueprints

We were constrained to designing binders with a maximum length of 80 amino acids (AA). While smaller binders could be extended with linkers, we aimed to engineer larger binders at the full 80 AA, giving us more flexibility and space for engineering.

To achieve this, we developed a `BinderBlueprint` generator (`scripts/binder_blueprints.py`) to create different adjacency matrices for binders of a defined length and secondary structural elements (SSEs). These matrices are later used with RFdiffusion for scaffold-guided diffusion of binders towards desired target hotspots.

Follow the example below to generate binder blueprints, or use the notebook `scripts/generate_scaffolds.ipynb` to generate binder blueprints for different scaffolds:

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

This will generate secondary structure elements and adjacency matrices for the binders, which can be used as input for scaffold-guided diffusion.

### RFdiffusion of protein binders

We took several approaches to designing the protein binders, leveraging various RFdiffusion scripts. These scripts target different regions of the hCD20 antigen, including:

- Targeting a single monomer (modeled as a dimer)
- Targeting the middle of hCD20
- Targeting the entire dimer, where binders interact with both domains
- Preparing symmetric binders to utilize target symmetry

More about diffusion: [WIP]
[WIP]

## License

This project is open-source and available under the [MIT License](LICENSE).

## Get in Touch

For questions, suggestions, or collaborations, feel free to reach out tadej.satler@gmail.com
