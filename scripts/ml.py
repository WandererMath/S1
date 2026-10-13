"""
Model architecture, losses, optimization and inference utilities for the
Goto et al. (WACV 2024) orthonormal-matrix binary classifier.

This module was split out of the former ``orth.py``; the training loop, loss
curves and CLI entry point now live in ``train.py``. Import model / loss /
optimizer / inference symbols from here (``ml``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from hyper import *

# ==========================================
# 1. Base Feature Extractor (Backbone)
# ==========================================
class TIR_FeatureExtractor(nn.Module):
	def __init__(self, filter_width=FILTER_WIDTH, input_length=INPUT_LENGTH):
		super().__init__()

		self.conv_block1 = nn.Sequential(
			nn.Conv1d(in_channels=4, out_channels=64, kernel_size=filter_width),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.3)
		)

		self.conv_block2 = nn.Sequential(
			nn.Conv1d(in_channels=64, out_channels=256, kernel_size=filter_width),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.4)
		)

		self.flatten = nn.Flatten()
		# Two valid convolutions followed by two stride-2 pools.  Derive this
		# from the configured TIR length instead of assuming the old 27-nt window.
		output_length = input_length
		for _ in range(2):
			output_length = output_length - filter_width + 1
			output_length = output_length // 2
		if output_length < 1:
			raise ValueError(
				f"input_length={input_length} is too short for two conv/pool blocks "
				f"with filter_width={filter_width}"
			)
		dim_flattened = 256 * output_length  # 1792 for a (4, 34) input and width 3
		self.dim_flattened = dim_flattened

	def forward(self, x):
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		return self.flatten(x)


# ==========================================
# 2. Goto et al. (WACV 2024) Binary Model
#    Class 0 (neg): orthonormal basis W_neg in R^{K_NEG x d}, W_neg W_neg^T = I_{K_NEG}
#    Class 1 (pos): orthonormal basis W_pos in R^{K x d},     W_pos W_pos^T = I_K
#    Logits (Eq. 11): l_c = kappa * ||ReLU(W^c z_norm)||_2^2
#    Both classes learn multiple orthonormal clusters (K for pos, K_NEG for neg).
# ==========================================
class MultiHeadOrthogonalTIR(nn.Module):
	# Class-level fallback so checkpoints pickled before the count-based dropout
	# refactor (and therefore lacking an instance ``n_dropout``) still unpickle
	# and run; eval / inference never touches the dropout path anyway.
	n_dropout = 0

	def __init__(self, num_basis=None, filter_width=FILTER_WIDTH, kappa=KAPPA,
				 num_clusters=None, num_neg_basis=None, n_dropout=N_DROPOUT,
				 input_length=INPUT_LENGTH):
		super().__init__()
		# The projection dimension d of W_pos/W_neg is no longer a hyperparameter:
		# the shared backbone (conv stack -> flatten, NO MLP/trunk) defines it, and
		# W is applied DIRECTLY to the backbone representation.
		self.backbone = TIR_FeatureExtractor(
			filter_width=filter_width, input_length=input_length
		)
		feat_dim = self.backbone.dim_flattened
		# num_clusters kept for backward compatibility with legacy call sites.
		if num_basis is None:
			num_basis = num_clusters if num_clusters is not None else K
		if num_neg_basis is None:
			num_neg_basis = K_NEG
		if num_basis > feat_dim:
			raise ValueError(
				f"num_basis ({num_basis}) must be <= feat_dim ({feat_dim}) so that "
				f"W_pos W_pos^T = I_{num_basis} can hold."
			)
		if num_neg_basis > feat_dim:
			raise ValueError(
				f"num_neg_basis ({num_neg_basis}) must be <= feat_dim ({feat_dim}) so that "
				f"W_neg W_neg^T = I_{num_neg_basis} can hold."
			)
		self.num_basis = num_basis
		self.num_clusters = num_basis  # alias kept for legacy consumers
		self.num_neg_basis = num_neg_basis
		self.feat_dim = feat_dim
		self.kappa = kappa
		# Subclass/Prototype Dropout: a fixed NUMBER of orthonormal rows
		# (subclasses) is dropped per sample during training, so the binary loss
		# cannot funnel all signal into a single dominant subclass. The count
		# comes from hyper.N_DROPOUT (edit it there); 0 disables it. It must not
		# reach a head's width, else every row would be dropped for some sample.
		if not isinstance(n_dropout, int) or isinstance(n_dropout, bool) or n_dropout < 0:
			raise ValueError(
				f"n_dropout ({n_dropout!r}) must be a non-negative integer."
			)
		max_drop = min(num_basis, num_neg_basis)
		if n_dropout >= max_drop:
			raise ValueError(
				f"n_dropout ({n_dropout}) must be < min(num_basis, num_neg_basis) "
				f"({max_drop}) so each head keeps at least one row."
			)
		self.n_dropout = n_dropout

		# Riemannian geometry: both classifier heads live on the Stiefel
		# manifold St(d, n) = {V in R^{d x n} | V^T V = I_n} (Goto et al.,
		# Eq. 13). geoopt's Stiefel uses this (d, n) layout with orthonormal
		# COLUMNS, which is the transpose of the paper's W^c in R^{n x d}
		# (n orthonormal ROWS, W^c W^c^T = I_n). So a stored V corresponds to
		# W^c = V^T, and a projection is z_norm @ V (columns = components).
		self.manifold = geoopt.Stiefel()

		# Negative orthonormal subspace: K_NEG components (columns) of the
		# stored (d, K_NEG) head, i.e. W_neg^T W_neg = I_{K_NEG}. Symmetric
		# to W_pos so the negative class also learns K_NEG subclasses.
		W_neg_init = torch.empty(feat_dim, num_neg_basis)
		nn.init.orthogonal_(W_neg_init)
		self.W_neg = geoopt.ManifoldParameter(W_neg_init, manifold=self.manifold)

		# Positive orthonormal subspace: K components (columns) of the stored
		# (d, K) head, i.e. W_pos^T W_pos = I_K.
		W_pos_init = torch.empty(feat_dim, num_basis)
		nn.init.orthogonal_(W_pos_init)
		self.W_pos = geoopt.ManifoldParameter(W_pos_init, manifold=self.manifold)

	def forward(self, x):
		# Embedding onto the unit hypersphere: z_norm = z / ||z||_2
		feat = self.backbone(x)
		z_norm = F.normalize(feat, p=2, dim=-1)

		# Continuous squared-ReLU projections (Eq. 11): l_c = kappa * ||ReLU(W z_norm)||_2^2
		# Both heads are Stiefel points stored as (d, n_c) with orthonormal columns
		# (transposed representation of the paper's n_c orthonormal rows):
		#   W_neg: (d, K_NEG) -> proj_neg (B, K_NEG), l_neg = kappa * sum_k p_neg[k]^2
		#   W_pos: (d, K)     -> proj_pos (B, K),     l_pos = kappa * sum_k p_pos[k]^2
		proj_neg = F.relu(z_norm @ self.W_neg)   # (B, K_NEG)
		proj_pos = F.relu(z_norm @ self.W_pos)   # (B, K)

		# Subclass/Prototype Dropout (training only). The binary CE loss has no
		# preference between spreading energy across all rows and funnelling it
		# into one row, so the head collapses onto a single dominant subclass.
		# Randomly suppressing a fixed NUMBER of rows (n_dropout, from hyper.py,
		# chosen independently per sample) removes that shortcut. Applied
		# symmetrically to BOTH heads and bypassed entirely for eval / inference,
		# so logits and project_subclass() are unchanged there.
		if self.training and self.n_dropout > 0:
			proj_neg = self._subclass_dropout_mask(proj_neg)
			proj_pos = self._subclass_dropout_mask(proj_pos)

		l_neg = self.kappa * proj_neg.square().sum(dim=-1)  # (B,)
		l_pos = self.kappa * proj_pos.square().sum(dim=-1)  # (B,)

		# Binary logits (B, 2): [class 0 (negative), class 1 (positive)]
		logits = torch.stack([l_neg, l_pos], dim=-1)
		return logits, z_norm

	def _subclass_dropout_mask(self, proj):
		"""
		Fixed-count Subclass/Prototype Dropout over the orthonormal basis dim.

		Independently for every sample, exactly ``self.n_dropout`` basis rows are
		chosen at random and zeroed; the surviving rows are then rescaled by
		``n_c / (n_c - n_dropout)`` (inverted dropout) so the expected squared
		energy -- hence the logit scale -- is preserved. The count is a plain
		integer from hyper.N_DROPOUT, which is easier to reason about and edit
		than a per-component probability.

		A per-sample safety clamp guarantees at least one row always survives,
		so the projection can never become degenerate.

		Args:
			proj: (B, n_c) non-negative ReLU projections for one head.
		Returns:
			(B, n_c) dropout-masked projections (differentiable, elementwise).
		"""
		n_c = proj.shape[-1]
		if self.n_dropout <= 0 or n_c <= 1:
			return proj
		# Never drop the whole head, even if n_dropout was set too large.
		n_drop = min(self.n_dropout, n_c - 1)
		# Pick n_drop rows per sample without replacement: argsort of i.i.d.
		# uniforms gives a uniform random permutation for each row.
		drop_idx = torch.rand(proj.shape, device=proj.device).argsort(dim=-1)[..., :n_drop]
		mask = torch.ones_like(proj)
		mask.scatter_(-1, drop_idx, 0.0)
		# Inverted dropout: rescale survivors to preserve expected energy.
		keep = n_c / (n_c - n_drop)
		return proj * mask * keep

	@torch.no_grad()
	def project_subclass(self, x, class_id=1):
		"""
		Subspace projection onto a class's orthonormal cluster basis.

		Args:
			x: input batch (B, 4, L)
			class_id: 1 -> positive subspace W_pos (K clusters, default);
			          0 -> negative subspace W_neg (K_NEG clusters).

		Returns:
			a_soft: (B, K_c) continuous projection coefficients a = ReLU(W^c z_norm)
			hard:   (B,) dominant component index argmax_k(a_k) in {0, ..., K_c-1}
		"""
		self.eval()
		W = self.W_pos if class_id == 1 else self.W_neg
		z_norm = F.normalize(self.backbone(x), p=2, dim=-1)
		a_soft = F.relu(z_norm @ W)                       # (B, K_c)
		hard = torch.argmax(a_soft, dim=-1)               # (B,)
		return a_soft, hard

	@torch.no_grad()
	def project_subclass_neg(self, x):
		"""Negative-sample subspace projection (K_NEG clusters). Alias convenience."""
		return self.project_subclass(x, class_id=0)


# ==========================================
# 3. Loss Function & Constraint Maintenance
# ==========================================
def _subclass_mi(z_norm, labels, model, class_id, W):
	"""
	Mutual-information regularizer term for ONE class's orthonormal clusters.

	For samples of `class_id` with p_i = softmax(kappa * ReLU(W z_i)) and
	q = mean_i p_i:
	    ent = mean_i H(p_i) - H(q)   (minimize -> each sample commits to one
	                                  component AND components stay balanced)
	    mi  = H(q) - mean_i H(p_i)   (diagnostic; equals -ent)
	Returns (ent, mi) as scalar tensors; (0, 0) when the batch holds fewer than
	two samples of that class.
	"""
	zero = torch.tensor(0.0, device=z_norm.device)
	mask = (labels == class_id)
	z_c = z_norm[mask]
	if z_c.shape[0] < 2:  # need >= 2 samples to estimate a usage distribution
		return zero, zero
	proj = F.relu(z_c @ W)                                     # (n_c, K_c)
	p = F.softmax(model.kappa * proj, dim=-1)                  # per-sample assignment
	per_sample_ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)  # H(p_i)
	mean_ent = per_sample_ent.mean()
	q = p.mean(dim=0).clamp_min(1e-12)                         # (K_c,) empirical usage
	bal_ent = -(q * torch.log(q)).sum()                        # H(q) in [0, ln K_c]
	return mean_ent - bal_ent, bal_ent - mean_ent


def compute_loss(logits, labels, model, lambda_ent=LAMBDA_ENT,
				 z_norm=None, pos_weight=POS_WEIGHT):
	"""
	Binary cross-entropy on the (B, 2) logits (class 0 neg / class 1 pos),
	optionally regularized by an entropy (mutual-information) term against
	subclass collapse.

	Orthonormality of W_pos/W_neg is enforced as a HARD constraint by the
	Stiefel-manifold retraction (geoopt RiemannianSGD + enforce_constraints),
	so no soft orthogonality penalty is applied here.

	Entropy (mutual-information) regularizer (default OFF unless lambda_ent > 0
	and z_norm from model.forward() is passed in). It is applied to BOTH classes
	so the K positive and K_NEG negative orthonormal clusters each stay used and
	sharp. For a batch's samples of a class with
	       p_i = softmax(kappa * ReLU(W^c z_i)),  q = mean_i p_i,
	   it minimizes
	       lambda_ent * (mean_i H(p_i) - H(q))
	   i.e. maximizes the mutual information I(z; k) = H(q) - mean_i H(p_i):
	     - mean_i H(p_i) minimized  -> each sample COMMITS to one component;
	     - H(q) maximized (toward ln K_c) -> components are BALANCED.
	   Pure CE only sees each summed logit l_c = kappa * sum_k proj_k^2,
	   so nothing pressures the K_c orthonormal rows to be used by different
	   samples -> argmax collapse onto one subclass. (A balance-only term -H(q)
	   is insufficient: q is ~uniform at init so it saturates trivially and gives
	   no gradient to carve distinct subclasses; the sharpening term adds that
	   gradient, and -H(q) then spreads the committed samples across all K_c.)
	   The positive term uses W_pos (K) and the negative term uses W_neg (K_NEG);
	   the reported "ent"/"mi" are the sums, with "mi_pos"/"mi_neg" diagnostics.
	"""
	labels = labels.long() if labels.dtype in (torch.float32, torch.float64) else labels
	# Class imbalance: weight the positive (y=1) class by pos_weight (default
	# POS_WEIGHT = BALANCE_RATIO) so missed positives are penalized ~BALANCE_RATIO
	# times more than false alarms when negatives outnumber positives.
	if pos_weight != 1.0:
		weight = torch.tensor([1.0, pos_weight], device=logits.device,
							  dtype=logits.dtype)
		ce_loss = F.cross_entropy(logits, labels, weight=weight)
	else:
		ce_loss = F.cross_entropy(logits, labels)

	# Entropy / mutual-information regularizer over BOTH classes' soft assignments
	# (positive subspace W_pos with K clusters, negative subspace W_neg with K_NEG).
	ent_loss = torch.tensor(0.0, device=logits.device)
	mi = torch.tensor(0.0, device=logits.device)
	mi_pos = torch.tensor(0.0, device=logits.device)
	mi_neg = torch.tensor(0.0, device=logits.device)
	if lambda_ent > 0 and z_norm is not None:
		ent_pos, mi_pos = _subclass_mi(z_norm, labels, model, 1, model.W_pos)
		ent_neg, mi_neg = _subclass_mi(z_norm, labels, model, 0, model.W_neg)
		ent_loss = ent_pos + ent_neg   # minimize -> maximize total I(z; k)
		mi = mi_pos + mi_neg           # total I(z; k) for diagnostics

	total_loss = ce_loss + lambda_ent * ent_loss

	loss_dict = {
		"total": total_loss.item(),
		"ce": ce_loss.item(),
		"ent": ent_loss.item(),
		"mi": mi.item(),
		"mi_pos": mi_pos.item(),
		"mi_neg": mi_neg.item(),
	}
	return total_loss, loss_dict


@torch.no_grad()
def enforce_constraints(model):
	"""
	Project both classifier heads back onto their Stiefel manifold.

	With geoopt's RiemannianSGD the Stiefel retraction is already applied
	inside optimizer.step(), so this is a cheap, idempotent safety
	re-projection (e.g. after manual parameter edits or checkpoint loading)
	that returns each head to V^T V = I_n (== W^c W^c^T = I_n for the paper's
	untransposed head W^c). projx is a no-op for points already on the
	manifold, so retaining this call after optimizer.step() is harmless.
	"""
	model.W_pos.copy_(model.manifold.projx(model.W_pos))
	model.W_neg.copy_(model.manifold.projx(model.W_neg))


def build_optimizer(model, lr=1e-3, momentum=0.9):
	"""Riemannian SGD with momentum, per Goto et al. (WACV 2024).

	Goto et al. optimize the orthonormal classifier matrices W^c on the
	Stiefel manifold using Riemannian SGD (implemented with geoopt [26]); we
	use a plain, non-adaptive Riemannian SGD with momentum to avoid adaptive
	step-size instability and obtain smoother gradient dynamics for the
	orthonormal subspace matrices.

	geoopt's RiemannianSGD computes the Riemannian gradient (projecting onto
	the tangent space of the Stiefel manifold) and applies the Stiefel
	retraction after every step automatically, so W_pos and W_neg stay
	orthonormal without a manual post-step QR.

	Parameter groups are kept decoupled:
	  - backbone: lr = lr,           weight_decay = 1e-4
	  - W_neg:    lr = LR_RATIO*lr,  weight_decay = 0.0
	  - W_pos:    lr = LR_RATIO*lr,  weight_decay = 0.0
	Weight decay is strictly 0.0 for the orthonormal heads (matching the
	paper: the Frobenius norm of an on-manifold head is fixed, so decay is
	meaningless and Goto et al. set it to zero). Note the backbone group must
	carry the base lr explicitly, since the global lr below is overridden by
	the per-group lr for W_pos/W_neg.
	"""
	return geoopt.optim.RiemannianSGD([
		{'params': model.backbone.parameters(), 'lr': lr, 'weight_decay': 1e-4},
		{'params': [model.W_neg], 'weight_decay': 0.0, 'lr': LR_RATIO * lr},
		{'params': [model.W_pos], 'weight_decay': 0.0, 'lr': LR_RATIO * lr},
	], lr=lr, momentum=momentum)


# ==========================================
# 4. Subclass Inference
# ==========================================
@torch.no_grad()
def infer_subclasses(model, dataloader, device="cpu", class_id=1):
	"""
	Assigns each sample of the given class to its dominant orthonormal component
	index via argmax_k ReLU(W^c z_norm).

	Args:
		class_id: 1 (default) -> positives over W_pos, indices in {0, ..., K-1};
		          0 -> negatives over W_neg, indices in {0, ..., K_NEG-1}.
	"""
	model.eval()
	subclass_assignments = []

	for batch_x, batch_y in dataloader:
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)
		mask = (batch_y == class_id)
		if mask.sum() > 0:
			_, hard = model.project_subclass(batch_x[mask], class_id=class_id)
			subclass_assignments.extend(hard.cpu().tolist())

	return subclass_assignments
