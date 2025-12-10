'''
This is the implement of proactive training against backdoor attacks [1].

Reference:
[1] Towards A Proactive ML Approach for Detecting Backdoor Poison Samples. USENIX, 2023.
'''

import os
from typing import List, Tuple, Optional, Iterable, Sequence
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from .base import Base

class ConfusionTraining(Base):
    """
    Fight Poison with Poison.

    Args:
        model:          the (randomly init or pretrained) model that will be trained in CT loop.
        inspection_set: the potentially poisoned training set to be cleansed.
        clean_set:      a small clean set (no triggers) used to create confusion batches.
        num_classes:    number of classes.
        device:         torch device.
        batch_size:     dataloader batch size.
        num_workers:    dataloader workers.
        pin_memory:     pin memory for loaders.
        use_tqdm:       show progress bars.
        lr_threshold:   LRT threshold for class selection.
        lr_maxclass_floor: relaxed LRT threshold for the max-ratio class.
        teacher_model:  (optional) a frozen teacher used only to create confusion labels.
                        If None, a deep-copied snapshot of `model` will be used (recommend teacher is a
                        backdoored model trained on the poisoned set, to mimic the paper
    """
    def __init__(
        self,
        model: nn.Module,
        inspection_set: Dataset,
        clean_set: Dataset,
        num_classes: int,
        device: str | torch.device = 'cpu',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_tqdm: bool = True,
        lr_threshold: float = 2.0,          # likelihood-ratio threshold
        lr_maxclass_floor: float = 1.5,     # relaxed threshold for the max-ratio class
        teacher_model: nn.Module | None = None,
    ):

        super().__init__()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.teacher = (teacher_model.to(self.device).eval() if teacher_model is not None
                        else copy.deepcopy(self.model).to(self.device).eval())

        self.inspection_set = inspection_set
        self.clean_set = clean_set
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_tqdm = use_tqdm

        self.lr_threshold = lr_threshold
        self.lr_maxclass_floor = lr_maxclass_floor

        # internal feature cache written by the forward hook
        self._features_cache = None
        # self._register_hook(self.model)

        # to be continued

    def _register_hook(self, model: nn.Module):
        """
        Register the hook to get the feats of the model
        """
        last_linear_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Linear):
                last_linear_layer = module
                break
        
        if last_linear_layer is None:
            raise ValueError("No FC layer founded in model")

        def hook(module, input, output):
            self._features_cache = input[0]
        
        last_linear_layer.register_forward_hook(hook)
        print("Hook registered for the last linear layer")

    def _get_features(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> Tuple[List[np.ndarray], List[int], List[int], List[float], List[float]]:
        """
        Extract per-sample:
          feats (np.ndarray), label, pred, gt_confidence, loss

        Returns lists aligned by sample order in the provided data_loader.
        """
        model.eval()
        ce_no_red = nn.CrossEntropyLoss(reduction="none")

        feats: List[np.ndarray] = []
        labels: List[int] = []
        preds: List[int] = []
        gt_conf: List[float] = []
        losses: List[float] = []

        iterator = tqdm(data_loader, desc="Extract feats") if self.use_tqdm else data_loader

        for batch in iterator:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["data"], batch["target"]
            else:
                raise ValueError("Unsupported batch format. Expect (data, target) or {'data','target'}.")

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # clear previous cache for safety
            self._features_cache = None

            logits = model(x)  # hook writes self._features_cache
            if self._features_cache is None:
                raise RuntimeError("Hook didn't capture features. Ensure model forward reached the hooked Linear layer.")

            # tensors
            batch_feats = self._features_cache.detach()         # [B, D]
            batch_loss = ce_no_red(logits, y)                   # [B]
            batch_prob = torch.softmax(logits, dim=1)           # [B, C]
            batch_pred = torch.argmax(logits, dim=1)            # [B]

            # to CPU / numpy
            feats_np = batch_feats.cpu().numpy()
            loss_np = batch_loss.cpu().numpy()
            prob_np = batch_prob.cpu().numpy()
            pred_np = batch_pred.cpu().numpy()
            y_np = y.cpu().numpy()

            B = y_np.shape[0]
            for i in range(B):
                feats.append(feats_np[i])
                labels.append(int(y_np[i]))
                preds.append(int(pred_np[i]))
                gt_conf.append(float(prob_np[i, y_np[i]]))
                losses.append(float(loss_np[i]))

        return feats, labels, preds, gt_conf, losses

    # ----------------------------- identification ------------------------------

    @torch.no_grad()
    def identify_poison_samples(
        self,
        clean_indices: Iterable[int],
        data_loader: Optional[DataLoader] = None,
        pca_q: int = 2,
    ) -> List[int]:
        """
        Identify suspicious (likely poisoned) samples using the simplified CT rule:
          1) extract features & predictions for inspection_set
          2) per-class PCA to 2D
          3) fit two Gaussians: (A) others, (B) 'isolated' (pred == class)
          4) likelihood ratio test; collect indices if above thresholds

        Args:
            clean_indices: indices expected to be clean (from high-loss chunklets etc.)
            data_loader: optional loader; if None, built automatically from inspection_set
            pca_q: PCA target rank (default 2 as in the paper implementation)

        Returns:
            suspicious_indices: list of dataset indices flagged as poisoned
        """
        # build loader if needed
        if data_loader is None:
            data_loader = DataLoader(
                self.inspection_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        # 1) extract feats/preds on inspection set
        feats, labels, preds, gt_conf, losses = self._get_features(self.model, data_loader)

        feats_np = np.asarray(feats)                 # [N, D]
        labels_np = np.asarray(labels, dtype=int)    # [N]
        preds_np = np.asarray(preds, dtype=int)      # [N]
        N = feats_np.shape[0]

        # group indices by class and by provided clean chunklet
        class_indices = [[] for _ in range(self.num_classes)]
        class_clean_chunk = [[] for _ in range(self.num_classes)]

        for i in range(N):
            class_indices[labels_np[i]].append(i)
        for i in clean_indices:
            class_clean_chunk[labels_np[i]].append(i)

        for c in range(self.num_classes):
            class_indices[c].sort()
            class_clean_chunk[c].sort()
            if len(class_indices[c]) < 2:
                raise ValueError(f"Class {c} has too few samples (<2).")
            if len(class_clean_chunk[c]) < 2:
                raise ValueError(f"Clean chunklet too small for class {c} (<2).")

        suspicious_indices: List[int] = []
        class_lr: List[float] = []

        # 2) per-class PCA & 2-Gaussian LR test
        for c in range(self.num_classes):
            idx_c = class_indices[c]                  # absolute indices for class c
            n_c = len(idx_c)

            # map clean_chunklet indices (absolute) to local positions in idx_c
            clean_local = []
            p = 0
            for i_abs in idx_c:
                while p < len(class_clean_chunk[c]) and class_clean_chunk[c][p] < i_abs:
                    p += 1
                if p < len(class_clean_chunk[c]) and class_clean_chunk[c][p] == i_abs:
                    clean_local.append(len(clean_local) + (0 if not clean_local else 1))  # not actually needed
            # PCA on features of this class
            X_c = torch.as_tensor(feats_np[idx_c], dtype=torch.float32, device=self.device)  # [n_c, D]
            # torch.pca_lowrank returns (U, S, V), we use V to project
            U, S, V = torch.pca_lowrank(X_c, q=pca_q)
            Z = (X_c @ V[:, :pca_q]).cpu().numpy()   # [n_c, pca_q], default pca_q=2

            # split into 'isolated' (pred == class) vs others
            iso_local, oth_local, labels_local = [], [], []
            for j, i_abs in enumerate(idx_c):
                if preds_np[i_abs] == c:
                    iso_local.append(j)
                    labels_local.append(1)
                else:
                    oth_local.append(j)
                    labels_local.append(0)

            Z_iso = Z[iso_local] if len(iso_local) > 0 else np.zeros((0, pca_q))
            Z_oth = Z[oth_local] if len(oth_local) > 0 else np.zeros((0, pca_q))

            # If both sides have at least 2 points and not degenerate
            if 2 <= Z_iso.shape[0] <= n_c - 2:
                mu0 = Z_oth.mean(axis=0)
                cov0 = np.cov(Z_oth.T) if Z_oth.shape[0] > 1 else np.eye(pca_q)
                mu1 = Z_iso.mean(axis=0)
                cov1 = np.cov(Z_iso.T) if Z_iso.shape[0] > 1 else np.eye(pca_q)
                cov0 = cov0 + 1e-3 * np.eye(pca_q)
                cov1 = cov1 + 1e-3 * np.eye(pca_q)

                # log-likelihoods under single cluster (use 'other') vs two clusters
                ll_single = 0.0
                ll_two = 0.0
                for j in range(n_c):
                    z = Z[j : j + 1]
                    ll_single += multivariate_normal.logpdf(z, mean=mu0, cov=cov0, allow_singular=True).sum()
                    which = labels_local[j]
                    if which == 0:
                        ll_two += multivariate_normal.logpdf(z, mean=mu0, cov=cov0, allow_singular=True).sum()
                    else:
                        ll_two += multivariate_normal.logpdf(z, mean=mu1, cov=cov1, allow_singular=True).sum()

                # normalized likelihood ratio (as in the reference code)
                lr = float(np.exp((ll_two - ll_single) / n_c))
            else:
                lr = 1.0

            class_lr.append(lr)

        # decide which classes to cleanse
        class_lr_arr = np.asarray(class_lr, dtype=float)
        max_lr = float(class_lr_arr.max())

        for c in range(self.num_classes):
            lr = class_lr[c]
            if (lr == max_lr and lr > self.lr_maxclass_floor) or (lr > self.lr_threshold):
                # collect samples in class c whose prediction == class c
                for i_abs in class_indices[c]:
                    if preds_np[i_abs] == c:
                        suspicious_indices.append(i_abs)

        return sorted(suspicious_indices)    
        
    def _build_loader(self, dset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def _pretrain_one_round(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 40,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
    ) -> nn.Module:
        """Pretrain on current distilled set (simple ERM)."""
        model.train()
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        crit = nn.CrossEntropyLoss()

        epoch_iter = range(1, epochs + 1)
        for ep in (tqdm(epoch_iter, desc="Pretrain") if self.use_tqdm else epoch_iter):
            for x, y in train_loader:
                x = x.to(self.device); y = y.to(self.device)
                optim.zero_grad()
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                optim.step()
        return model

    def _compute_class_freq_weights(self, distilled_subset: Subset) -> np.ndarray:
        """Compute sqrt-normalized class frequency weights on a subset."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, y in distilled_subset:
            counts[int(y)] += 1
        total = max(1, int(counts.sum()))
        freq = np.sqrt(counts / total + 1e-3)  # same trick as paper code
        return freq

    def _confusion_train_one_round(
        self,
        model: nn.Module,
        teacher: nn.Module,
        distilled_loader: DataLoader,
        clean_loader: DataLoader,
        class_freq: Sequence[float],
        num_iters: int = 6000,
        lamb: float = 10.0,
        batch_factor: int = 4,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
    ) -> nn.Module:
        """
        CT loop: jointly train on (A) confusion batch from clean_set w/ randomized wrong labels,
        and (B) distilled batch from inspection_set.
        """
        model.train()
        teacher.eval()
        ce_no_red = nn.CrossEntropyLoss(reduction='none')

        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        distilled_iter = iter(distilled_loader)
        clean_iter = iter(clean_loader)

        rounder = 0
        class_freq = np.asarray(class_freq, dtype=np.float32)
        class_freq[class_freq <= 0] = 1.0  # avoid div by zero

        iter_range = range(num_iters)
        for step in (tqdm(iter_range, desc="Confusion train") if self.use_tqdm else iter_range):
            # fetch a clean batch
            try:
                x_clean, _y_clean = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                x_clean, _y_clean = next(clean_iter)
            x_clean = x_clean.to(self.device)

            with torch.no_grad():
                # teacher preds -> shift -> wrong labels
                preds = torch.argmax(teacher(x_clean), dim=1)
                # rolling shift to enforce wrong labels
                if (rounder + step) % self.num_classes == 0:
                    rounder += 1
                target_confusion = (preds + rounder + step) % self.num_classes

            if step % batch_factor == 0:
                # mix with a distilled batch
                try:
                    x_ins, y_ins = next(distilled_iter)
                except StopIteration:
                    distilled_iter = iter(distilled_loader)
                    x_ins, y_ins = next(distilled_iter)
                x_ins = x_ins.to(self.device); y_ins = y_ins.to(self.device)

                x_mix = torch.cat([x_clean, x_ins], dim=0)
                y_mix = torch.cat([target_confusion, y_ins], dim=0)

                logits_mix = model(x_mix)
                loss_mix = ce_no_red(logits_mix, y_mix)  # [B_clean+B_ins]

                Bc = x_clean.shape[0]
                loss_conf = loss_mix[:Bc].mean()

                # class-frequency reweighting on inspection batch
                loss_ins_vals = loss_mix[Bc:]
                y_ins_cpu = y_ins.detach().cpu().numpy()
                weights = torch.as_tensor(1.0 / class_freq[y_ins_cpu], device=loss_ins_vals.device, dtype=loss_ins_vals.dtype)
                loss_ins = (loss_ins_vals * weights).sum() / weights.sum()

                loss = ((lamb - 1.0) * loss_conf + loss_ins) / lamb
            else:
                # confusion-only step
                logits_c = model(x_clean)
                loss = ce_no_red(logits_c, target_confusion).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def _distill_by_loss(
        self,
        model: nn.Module,
        dset: Dataset,
        keep_ratio: Optional[float] = None,
        final_budget: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Rank all inspection samples by per-sample CE loss and:
          - if keep_ratio is not None: keep top-keep_ratio (low-loss) indices
          - else if final_budget is not None: keep top-k
          - else: keep all correctly predicted indices
        Also returns per-class 'median chunklet' indices for later likelihood test.
        """
        loader = self._build_loader(dset, shuffle=False)
        ce_no_red = nn.CrossEntropyLoss(reduction='none')

        model.eval()
        losses, preds, labels = [], [], []
        with torch.no_grad():
            for x, y in (tqdm(loader, desc="Score by loss") if self.use_tqdm else loader):
                x = x.to(self.device); y = y.to(self.device)
                logits = model(x)
                batch_loss = ce_no_red(logits, y)
                batch_pred = torch.argmax(logits, dim=1)

                losses.extend(batch_loss.cpu().tolist())
                preds.extend(batch_pred.cpu().tolist())
                labels.extend(y.cpu().tolist())

        losses = np.asarray(losses); labels = np.asarray(labels); preds = np.asarray(preds)
        N = len(losses)
        order = np.argsort(losses)  # low loss first

        if keep_ratio is not None:
            k = max(1, int(keep_ratio * N))
            kept = order[:k].tolist()
        elif final_budget is not None:
            k = min(N, int(final_budget))
            kept = order[:k].tolist()
        else:
            kept = [i for i in range(N) if preds[i] == labels[i]]

        # build class-wise median chunklet (center band) for later
        median_indices: List[int] = []
        per_class_sorted: List[List[int]] = [[] for _ in range(self.num_classes)]
        for idx in order:
            per_class_sorted[labels[idx]].append(idx)
        # take middle 20% by default (paper uses a configurable ratio; here fix to 0.2)
        median_rate = 0.2
        for c in range(self.num_classes):
            arr = per_class_sorted[c]
            if len(arr) == 0:
                continue
            st = int(len(arr) * (0.5 - median_rate / 2))
            ed = int(len(arr) * (0.5 + median_rate / 2))
            median_indices.extend(arr[st:ed])

        kept.sort()
        median_indices.sort()
        return kept, median_indices

    # ------------------------------- likelihood test & final selection -------------------------------

    @torch.no_grad()
    def _identify_poison_by_lr(
        self,
        model: nn.Module,
        inspection_set: Dataset,
        clean_indices: Iterable[int],
    ) -> List[int]:
        """Simplified CT rule: per-class 2D PCA + two-Gaussian likelihood ratio."""
        loader = self._build_loader(inspection_set, shuffle=False)
        feats, labels, preds, _conf, _loss = self._get_features(model, loader)

        feats = np.asarray(feats)
        labels = np.asarray(labels, dtype=int)
        preds = np.asarray(preds, dtype=int)
        N = len(labels)

        class_idx = [[] for _ in range(self.num_classes)]
        class_clean = [[] for _ in range(self.num_classes)]
        for i in range(N):
            class_idx[labels[i]].append(i)
        for i in clean_indices:
            class_clean[labels[i]].append(i)
        for c in range(self.num_classes):
            class_idx[c].sort()
            class_clean[c].sort()
            if len(class_idx[c]) < 2 or len(class_clean[c]) < 2:
                # keep graceful behavior; skip tiny classes
                pass

        class_lr: List[float] = []
        suspicious: List[int] = []

        for c in range(self.num_classes):
            idx_c = class_idx[c]
            if len(idx_c) < 2:
                class_lr.append(1.0)
                continue

            Xc = torch.as_tensor(feats[idx_c], dtype=torch.float32, device=self.device)
            # PCA to 2D
            try:
                U, S, V = torch.pca_lowrank(Xc, q=2)
                Z = (Xc @ V[:, :2]).cpu().numpy()
            except RuntimeError:
                # fallback: use first two dims
                Z = Xc[:, :2].cpu().numpy()

            iso_local, oth_local, label_local = [], [], []
            for j, abs_i in enumerate(idx_c):
                if preds[abs_i] == c:
                    iso_local.append(j); label_local.append(1)
                else:
                    oth_local.append(j); label_local.append(0)

            if len(iso_local) >= 2 and len(iso_local) <= len(idx_c) - 2 and len(oth_local) >= 2:
                Z_iso = Z[iso_local]; Z_oth = Z[oth_local]
                mu0, mu1 = Z_oth.mean(0), Z_iso.mean(0)
                cov0 = np.cov(Z_oth.T) if Z_oth.shape[0] > 1 else np.eye(2)
                cov1 = np.cov(Z_iso.T) if Z_iso.shape[0] > 1 else np.eye(2)
                cov0 = cov0 + 1e-3 * np.eye(2)
                cov1 = cov1 + 1e-3 * np.eye(2)

                ll_single, ll_two = 0.0, 0.0
                for j in range(len(idx_c)):
                    z = Z[j:j+1]
                    ll_single += multivariate_normal.logpdf(z, mean=mu0, cov=cov0, allow_singular=True).sum()
                    if label_local[j] == 0:
                        ll_two += multivariate_normal.logpdf(z, mean=mu0, cov=cov0, allow_singular=True).sum()
                    else:
                        ll_two += multivariate_normal.logpdf(z, mean=mu1, cov=cov1, allow_singular=True).sum()
                lr = float(np.exp((ll_two - ll_single) / max(1, len(idx_c))))
            else:
                lr = 1.0
            class_lr.append(lr)

        class_lr = np.asarray(class_lr, dtype=float)
        max_lr = float(class_lr.max() if len(class_lr) > 0 else 1.0)

        for c in range(self.num_classes):
            lr = class_lr[c]
            if (lr == max_lr and lr > self.lr_maxclass_floor) or (lr > self.lr_threshold):
                for i_abs in class_idx[c]:
                    if preds[i_abs] == c:
                        suspicious.append(i_abs)

        return sorted(set(suspicious))

    # ------------------------------- public: one-call cleanse -------------------------------

    def cleanse(
        self,
        # schedule across rounds (paper uses ~4-5 rounds)
        distillation_ratios: Sequence[Optional[float]] = (None, 0.6, 0.4, 0.3),  # None==keep 'correct-only' in 1st round
        # per-round CT settings (broadcast if scalar)
        pretrain_epochs: Sequence[int] = (100, 40, 40, 40),
        pretrain_lr: Sequence[float] = (1e-2, 1e-2, 1e-2, 1e-2),
        ct_iters: Sequence[int] = (6000, 6000, 6000, 2000),
        ct_lr: Sequence[float] = (1e-2, 1e-2, 1e-2, 1e-2),
        ct_momentum: Sequence[float] = (0.9, 0.9, 0.9, 0.9),
        ct_weight_decay: float = 1e-4,
        lambs: Sequence[float] = (10.0, 10.0, 10.0, 10.0),
        batch_factors: Sequence[int] = (4, 4, 4, 4),
        final_budget: Optional[int] = None,
        return_indices_only: bool = False,
    ) -> Tuple[Sequence[int], Optional[Subset]]:
        """
        Run multi-round Distillation + Confusion Training, then LR test.
        Returns:
            remain_indices: indices kept as the cleansed training set.
            cleansed_subset: torch.utils.data.Subset(inspection_set, remain_indices) if return_indices_only=False
        """
        assert len(distillation_ratios) == len(pretrain_epochs) == len(pretrain_lr) == len(ct_iters) == len(ct_lr) == len(ct_momentum) == len(lambs) == len(batch_factors), \
            "Per-round schedules must have the same length."

        # working model (trained in-place)
        work_model = self.model

        # loaders
        clean_loader = self._build_loader(self.clean_set, shuffle=True)

        # initialize distilled set as the whole inspection set
        distilled_indices = list(range(len(self.inspection_set)))
        distilled_subset = Subset(self.inspection_set, distilled_indices)

        for r in range(len(distillation_ratios)):
            # class freq weights (sqrt-normalized); last round can optionally set to ones:
            class_freq = self._compute_class_freq_weights(distilled_subset)
            if r == len(distillation_ratios) - 1:
                class_freq[:] = 1.0

            # pretrain on current distilled subset
            distilled_loader = self._build_loader(distilled_subset, shuffle=True)
            work_model = self._pretrain_one_round(
                work_model, distilled_loader,
                epochs=pretrain_epochs[r], lr=pretrain_lr[r],
            )

            # CT loop
            work_model = self._confusion_train_one_round(
                work_model, self.teacher,
                distilled_loader=distilled_loader,
                clean_loader=clean_loader,
                class_freq=class_freq,
                num_iters=ct_iters[r],
                lamb=lambs[r],
                batch_factor=batch_factors[r],
                lr=ct_lr[r],
                weight_decay=ct_weight_decay,
                momentum=ct_momentum[r],
            )

            # distill by loss to update distilled subset for the next round
            keep_ratio = distillation_ratios[r]
            distilled_indices, median_indices = self._distill_by_loss(
                work_model, self.inspection_set,
                keep_ratio=keep_ratio,
                final_budget=final_budget if r == len(distillation_ratios) - 1 else None,
            )
            distilled_subset = Subset(self.inspection_set, distilled_indices)

        self._register_hook(work_model)
        # final LR test to identify suspicious (to drop)
        suspicious = self._identify_poison_by_lr(work_model, self.inspection_set, clean_indices=median_indices)

        # build remain (cleansed) indices
        N = len(self.inspection_set)
        suspicious_set = set(suspicious)
        remain_indices = [i for i in range(N) if i not in suspicious_set]

        if return_indices_only:
            return remain_indices, None
        else:
            return remain_indices, Subset(self.inspection_set, remain_indices)
