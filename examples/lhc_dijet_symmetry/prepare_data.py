"""
Prepare LHC dijet data for symmetry discovery.

Loads the LHCO R&D anomaly detection dataset, clusters particles into jets
using the anti-kT algorithm (R=1.0) via FastJet, and extracts the transverse
momentum components (px, py) of the two leading jets per event.

Requirements
------------
- pandas, numpy, torch, tables (PyTables for HDF5)
- fastjet (python bindings): pip install fastjet
- Dataset: events_anomalydetection_v2.h5
  Download from https://zenodo.org/record/4536377

Usage
-----
    python prepare_data.py [--input PATH] [--output PATH] [--n-events N]
"""

import argparse
import numpy as np
import pandas as pd
import torch

def prepare_lhc_data(
    file_path: str = "events_anomalydetection_v2.h5",
    n_events: int = 20000,
    output_path: str = "lhc_dijet_data.pt",
) -> torch.Tensor:
    """
    Load HDF5 data, cluster with FastJet, extract leading dijet momenta.

    Returns
    -------
    X : torch.Tensor, shape (n_valid_events, 4)
        Columns: [p1x, p1y, p2x, p2y] in GeV.
    """
    import fastjet as fj

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data with Pandas...")
    df = pd.read_hdf(file_path, start=0, stop=n_events)
    raw_data = df.values
    num_events = raw_data.shape[0]

    # Separate particles (700 per event, 3 features: pT, eta, phi)
    particles = raw_data[:, :-1].reshape(num_events, 700, 3)

    final_jets = []

    # Anti-kT algorithm with R=1.0 (standard ATLAS/CMS large-R jet definition)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 1.0)

    print("Clustering with FastJet (anti-kT, R=1.0)...")
    for i in range(num_events):
        event_particles = particles[i]

        # Filter out zero-padding
        valid_particles = event_particles[event_particles[:, 0] > 0]

        # Build PseudoJets from (pT, eta, phi)
        pjs = []
        for pt, eta, phi in valid_particles:
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(pt, eta, phi, 0)
            pjs.append(pj)

        # Run clustering
        jets = jet_def(pjs)

        # Extract px, py of the two leading jets (sorted by pT)
        if len(jets) >= 2:
            p1x, p1y = jets[0].px(), jets[0].py()
            p2x, p2y = jets[1].px(), jets[1].py()
            final_jets.append([p1x, p1y, p2x, p2y])

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{num_events} events...")

    X = torch.tensor(final_jets, dtype=torch.float32).to(device)
    print(f"Successfully processed {len(X)} events.")
    print(f"Tensor shape: {X.shape}")

    # Save for later use
    torch.save(X.cpu(), output_path)
    print(f"Saved to {output_path}")

    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LHC dijet data")
    parser.add_argument("--input", default="events_anomalydetection_v2.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--output", default="lhc_dijet_data.pt",
                        help="Output path for prepared tensor")
    parser.add_argument("--n-events", type=int, default=20000,
                        help="Number of events to process")
    args = parser.parse_args()

    prepare_lhc_data(args.input, args.n_events, args.output)
