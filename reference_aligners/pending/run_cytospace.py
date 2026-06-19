import os
import cytospace
import argparse
import logging
import pandas as pd
from utils import fmt_nonzero_4


def cytospace_align_data(dataset_folder: str):
    """
    Run CytoSPACE alignment on a prepared dataset in the given folder.
    Saves predicted gene expression per spot (GEP) to custom output folder.

    Args:
        dataset_folder: Path to dataset folder

    Returns: None
    """
    assert os.path.isdir(dataset_folder), f"Dataset folder not found: {dataset_folder}"

    # Create temporary output folder at dataset_folder/temp_cytospace_output
    output_folder = os.path.join(dataset_folder, "results", "temp_cytospace_output")
    os.makedirs(output_folder, exist_ok=True)

    scRNA_Path = os.path.join(dataset_folder, "scData_GEP.csv")
    cellType_Path = os.path.join(dataset_folder, "scData_Cells.csv")
    st_Path = os.path.join(dataset_folder, "stData_GEP.csv")
    st_Coords = os.path.join(dataset_folder, "stData_Spots.csv")

    cytospace.main_cytospace(
        scRNA_path=scRNA_Path,
        cell_type_path=cellType_Path,
        st_path=st_Path,
        coordinates_path=st_Coords,
        st_cell_type_path=None,
        n_cells_per_spot_path=None,
        solver_method="lap_CSPR",
        output_folder=output_folder,
    )

    print("Start post-processing")

    """ Create cytospace_GEP.csv in dataset_folder/results out of assigned locations """

    # Step 1: Load spots
    st_df = pd.read_csv(st_Coords, index_col=0)

    # Step 2: Load scData_GEP.csv with pandas
    scRNA_df = pd.read_csv(scRNA_Path, index_col=0)

    # Step 3: Load assigned locations with pandas
    assigned_locations_path = os.path.join(output_folder, "assigned_locations.csv")
    assigned_locations_df = pd.read_csv(assigned_locations_path, index_col=0)

    # Step 4: Create binary matrix of assigned locations
    # Rows: CellID, Columns: SpotID, Values: 1 if Cell assigned to Spot, else 0
    assigned_matrix_df = pd.crosstab(
        assigned_locations_df["OriginalCID"], assigned_locations_df["SpotID"]
    )
    assigned_matrix_df = assigned_matrix_df.reindex(
        index=scRNA_df.columns, columns=st_df.index, fill_value=0
    )

    # Step 5: Multiply scRNA GEP with assigned matrix to get spot GEPs
    cytospace_GEP_df = scRNA_df.dot(assigned_matrix_df)
    df_formatted = cytospace_GEP_df.map(fmt_nonzero_4)

    # Schreibe Ergebnis als CSV (Zeilen=Gene, Spalten=Spots). Obere linke Zelle = "GEP"
    out_file = os.path.join(dataset_folder, "results", "cytospace_GEP.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_formatted.to_csv(out_file, index=True, index_label="GEP")
    print(f"Saved cytospace GEP to {out_file}")


if __name__ == "__main__":
    """
    Run CytoSPACE alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Tangram alignment on a dataset folder"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="Path to dataset folder (default: development workspace mouse cortex)",
    )
    args = parser.parse_args()
    cytospace_align_data(args.dataset)
