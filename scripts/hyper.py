from pathlib import Path
# Set to a numeric results run ID (for example ``3``) to resume
# ``results/3/0.pth`` until it reaches N_EPOCHS. Leave as None for a new run.
MODEL_ID = None 
BALANCE_RATIO = 5 # Negatives / Positives
K = 10 # Number of orthonormal basis components (subclasses) for the positive class
K_NEG= 10
# Number of orthonormal ROWS (subclasses) of each classifier head W to drop out
# per training sample (subclass/prototype dropout). This is a plain COUNT, so it
# is easy to edit: W_pos has K rows and W_neg has K_NEG rows, and N_DROPOUT rows
# are zeroed at random for every sample during training. Must satisfy
# 0 <= N_DROPOUT < min(K, K_NEG); set to 0 to disable dropout.
N_DROPOUT = 0
#KAPPA=10
KAPPA = 5* 50/2048*512 * 3  # Temperature scaling on squared-ReLU projection logits (Eq. 11, Goto et al.)
# Entropy (mutual-information) anti-collapse regularizer. Pure CE only sees the
# *summed* positive logit l_pos = kappa * sum_k ReLU(W_pos[k] z)^2, so nothing
# forces the K orthonormal W_pos rows to be used by different positive samples
# -> argmax collapse onto a single subclass. For a batch's positives with
# p_i = softmax(kappa * ReLU(W_pos z_i)), q = mean_i p_i, the term minimized is
#   mean_i H(p_i) - H(q)   (i.e. it maximizes I(z;k) = H(q) - mean_i H(p_i)):
# per-sample entropy pushed down (each positive commits to one component) while
# batch usage entropy H(q) pushed up toward ln K (components stay balanced).
LAMBDA_ENT = 0.001  # Weight on the entropy (mutual-information) regularizer
# Class-imbalance weighting: the dataset holds ~BALANCE_RATIO negatives per
# positive, so weight the positive (y=1) class by POS_WEIGHT inside cross-entropy
# to penalize missed positives (false negatives) ~BALANCE_RATIO x more than false
# alarms. Default value is BALANCE_RATIO.
# POS_WEIGHT = BALANCE_RATIO  # Positive-class (y=1) loss weight applied in CE
POS_WEIGHT = 1
N_EPOCHS_DROPOUT_STAGE= 5
N_EPOCHS = 50
BATCH_SIZE = 256
N_MODELS = 1
FILTER_WIDTH = 3
# TIR window: 25 nt upstream, the 3-nt AUG start codon, and 6 nt downstream.
TIR_UPSTREAM = 25
TIR_DOWNSTREAM = 6
INPUT_LENGTH = TIR_UPSTREAM + 3 + TIR_DOWNSTREAM


def next_results_dir(results_dir: Path) -> Path:
    """Return the next numeric experiment directory under ``results/``."""
    existing_ids = [
        int(path.name)
        for path in (results_dir.iterdir() if results_dir.exists() else [])
        if path.is_dir() and path.name.isdigit()
    ]
    return results_dir / str(max(existing_ids, default=0) + 1)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROOT_DIR = next_results_dir(RESULTS_DIR)
# Single project-root SQLite database shared by every script. Its schema is
# documented in db.tsv: essential fields (id, species, sequence, sd) plus the
# model-specific `cluster` field.
DB_PATH = "data.db"

LR=1e-3
LR_RATIO = 1e-1  # W_pos learning rate = LR_RATIO * backbone learning rate



hyperparameters = {
    key: value
    for key, value in globals().items()
    if key.isupper() and isinstance(value, (str, int, float, bool))
}
