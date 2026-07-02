import os
from pathlib import Path


ABIDE_I_2D_REGRESSION = Path(
    "/projectnb/ace-genetics/ABIDE/2D_MRI_VAE_regression"
).resolve()
ABIDE_I_RAW = Path("/projectnb/ace-ig/ABIDE/ABIDE_I/ABIDE_I_BIDS").resolve()
FASTSURFER_II = Path("/projectnb/ace-ig/ABIDE/ABIDE_II/ABIDE_II_fastsurfer").resolve()
FASTSURFER_II_NEW = Path(
    "/projectnb/ace-genetics/ABIDE/ABIDE_II/ABIDE_II_fastsurfer"
).resolve()
FASTSURFER_I = Path("/projectnb/ace-ig/ABIDE/ABIDE_I/ABIDE_I_fastsurfer").resolve()
ABIDE_II_RAW = Path("/projectnb/ace-ig/ABIDE/ABIDE_II/ABIDE_II_BIDS").resolve()
ABIDE_II_RAW_NEW = Path("/projectnb/ace-genetics/ABIDE/ABIDE_II/original/").resolve()
ABIDE_I_MNI = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_I/ABIDE_I_BIDS/derivatives/MNI/"
).resolve()
ABIDE_I_FASTSURFER = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_I/ABIDE_I_fastsurfer"
).resolve()
ABIDE_II_MNI = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_II/ABIDE_II_BIDS/derivatives/MNI/"
).resolve()
ABIDE_DATA_FOLDER_I = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_ANTS/ABIDE/").resolve()
ABIDE_DATA_FOLDER_I_FREESURFER_RECON = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_I_freesurfer_recon/ABIDE/"
).resolve()
ABIDE_I_BIDS = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS").resolve()
ABIDE_II_BIDS = Path("/projectnb/ace-ig/ABIDE/ABIDE_II/ABIDE_II_BIDS").resolve()
ABIDE_DATA_FOLDER_II = Path("/projectnb/ace-ig/ABIDE/ABIDE_II_T1/ABIDE_II/").resolve()
ABIDE_I_PHENOTYPE = Path("/projectnb/ace-ig/ABIDE/Phenotypic_V1_0b.csv").resolve()
ABIDE_II_PHENOTYPE = Path(
    "/projectnb/ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv"
).resolve()
ABIDE_II_PHENOTYPE_Long = Path(
    "/projectnb/ace-ig/ABIDE/ABIDEII_Long_Composite_Phenotypic.csv"
).resolve()
ABIDE_I_IMAGE_FEATURES = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_I/ABIDE_I_fastsurfer/group_stats/BN_Atlas_lr_merged.csv"
).resolve()
ABIDE_II_IMAGE_FEATURES = Path(
    "/projectnb/ace-genetics/ABIDE/ABIDE_II/ABIDE_II_fastsurfer/group_stats/BN_Atlas_lr_merged.csv"
).resolve()
ACE_PHENOTYPE = Path("/projectnb/ace-ig/ace_phenotype.csv").resolve()
TENSORBOARD_LOG_DIR = Path(
    "/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/tensorboard_new"
).resolve()
TENSORBOARD_GENETICS = Path(
    "/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/genetics_tensorboard"
).resolve()
TENSORBOARD_CROSS_MODALITY = Path(
    "/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/cross_modality_tensorboard"
).resolve()
ACE_FILE_with_relatedness = Path(
    "/projectnb/ace-genetics/jueqiw/dataset/BrainGenePathway/ACE/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb_with_related.csv"
)
ACE_FILE = Path(
    "/projectnb/ace-genetics/jueqiw/dataset/BrainGenePathway/ACE/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb.csv"
)
ACE_IMG_FEAT = Path(
    "/projectnb/ace-genetics/jueqiw/dataset/CrossModalityLearning/ACE/rearranged_ACE_image_features_without_relatedness.csv"
)
# SSC pathway CSV (was an inline literal in main.py::load_genetics_data).
SSC_FILE = Path(
    "/projectnb/ace-genetics/jueqiw/dataset/CrossModalityLearning/SSC/CSV/final_SSC_pseudo_KEGG_pathway_with_all_genes_p_threshold_0.1_effect_size_LD_50kb.csv"
)
# Root that get_ACE_subjects() globs for ACE T1w NIfTIs (was hardcoded in utils.py).
ACE_MRI_FOLDER = Path("/projectnb/ace-ig/ace1-mri/fmriprep/")
ADNI_FILE = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/Gene/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
)
CROSS_VAL_INDEX_ACE = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ACE/10_10_cross_fold_val_index.pkl"
)
CROSS_VAL_INDEX_ADNI = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/10_10_cross_fold_val_index.pkl"
)
RESULT_FOLDER = Path(
    "/projectnb/ace-ig/jueqiw/experiment/BrainGenePathway/results"
).resolve()

DATA_FOLDER = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_ANTS/ABIDE/").resolve()

TADPOLE_FOLDER = DATA_FOLDER / "tadpole"
P_VALUE_FOLDER = DATA_FOLDER / "genetics_data" / "p_value"
SNP_FOLDER = (
    DATA_FOLDER
    / "genetics_data"
    / "ADNI_Test_Data"
    / "ImputedGenotypes"
    / "plink_preprocess"
)

# ACE data path
# ACE_GENO_FILE = "/projectnb/ace-ig/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/selected_geno_pheno.csv"
ACE_GENO_FILE = "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/total_geno.csv"
ACE_IMG_FILE_DESTRIEUX = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/sMRI_destrieux_cortical_thickness_average.csv"
ACE_IMG_FILE_BRAINNETOME = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/ACE_img_Brainnetome.csv"
ACE_IMG_GENO_FOLDER = Path("/project/ace-ig/jueqiw/data/ACE/joint")

# +-----------------+
# | ACE paired path |
# +-----------------+
ACE_PAIRED_FOLDER = Path(
    "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/GMIND_new/pair_train_val_test"
)

# +------------+
# | SSC Folder |
# +------------+
SSC_FOLDER = Path("/project/ace-ig/jueqiw/data/SSC/")
ACE_IMG_GENE_INNER = Path(
    "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/ACE_img_Brainnetome_geno_inner.csv"
)

# +-------------------------------------------------------------------+
# | Dummy-data mode: set CALM_DUMMY_DATA=<dir> to point every data    |
# | path at a local folder (see make_dummy_data.py) so the real       |
# | loaders run end-to-end without any cluster data. Unset -> the     |
# | SCC paths above are used unchanged.                               |
# +-------------------------------------------------------------------+
_DUMMY_DATA_DIR = os.environ.get("CALM_DUMMY_DATA")
if _DUMMY_DATA_DIR:
    _D = Path(_DUMMY_DATA_DIR).resolve()
    ABIDE_I_MNI = _D / "abide_i_mni"
    ABIDE_II_MNI = _D / "abide_ii_mni"
    ACE_MRI_FOLDER = _D / "ace_mri"
    ABIDE_I_PHENOTYPE = _D / "abide_i_phenotype.csv"
    ABIDE_II_PHENOTYPE = _D / "abide_ii_phenotype.csv"
    ABIDE_II_PHENOTYPE_Long = _D / "abide_ii_long_phenotype.csv"
    ABIDE_I_IMAGE_FEATURES = _D / "abide_i_features.csv"
    ABIDE_II_IMAGE_FEATURES = _D / "abide_ii_features.csv"
    ACE_PHENOTYPE = _D / "ace_phenotype.csv"
    ACE_FILE_with_relatedness = _D / "ace_pathway_img.csv"
    ACE_IMG_FEAT = _D / "ace_img_features.csv"
    SSC_FILE = _D / "ssc_pathway.csv"
