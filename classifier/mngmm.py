# %%writefile /kaggle/working/VB-CGCD-main/classifier/mngmm.py
import copy
import os
from collections import defaultdict
from math import log

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import random
from numpy import save
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from prettytable import PrettyTable
from sklearn.decomposition import PCA, FactorAnalysis
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MNGMMClassifier():

    def __init__(self, num_dim, num_classes, with_early_stop):
        self.num_dim = num_dim
        self.num_classes = num_classes

        self.pca = None

        self.global_params = None

        self.label_offset = 0

        self.with_early_stop = with_early_stop

        # Diagonal covariance mode: enabled automatically when samples are too
        # few relative to feature dimensionality.  This reduces parameters from
        # O(d²) to O(d) and prevents ill-conditioned covariance matrices.
        self._use_diagonal_cov = False


    def update_dir_infos(self, log_dir = "logs/", save_dir = "saved_models/"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def init_parameters(self, n_epochs, lr, log_dir, save_dir, batch_size, increment=10, base=50, scaling_factor=1.2, use_correct_scaling_factor=True, early_stop_ratio=0):

        self.num_steps = n_epochs

        self.init_lr = lr

        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        self.increment = increment

        self.num_base = base

        self.scaling_factor = scaling_factor

        self.use_correct_scaling_factor = use_correct_scaling_factor

        self.early_stop_ratio = early_stop_ratio

    def _effective_batch_size(self, n_samples):
        if n_samples < 1:
            raise ValueError("MNGMMClassifier needs at least 1 training sample")
        return min(self.batch_size, n_samples)

    def _stable_covariance(self, cov):
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        eigvals = jnp.clip(eigvals, 1e-4, 1e4)
        return (eigvecs * eigvals[..., None, :]) @ jnp.swapaxes(eigvecs, -1, -2)

    def _stable_covariance_np(self, cov):
        cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-4, 1e4)
        return (eigvecs * eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)

    def model(self, X, y=None, num_classes=2, global_params=None, **kwargs):
        num_features = X.shape[1]

        if global_params is None:
            class_means = numpyro.param("class_means", jnp.zeros((num_classes, num_features)))
            class_covs = numpyro.param("class_covs", jnp.stack([jnp.eye(num_features)] * num_classes))
        else :
            class_means = numpyro.param("class_means", global_params["class_means"])
            class_covs = numpyro.param("class_covs", global_params["class_covs"])
        
        with numpyro.plate("batch", X.shape[0], subsample_size=self._effective_batch_size(X.shape[0])) as ind:
            X_batch = X[ind]
            y_batch = y[ind] if y is not None else None
            
            if y_batch is not None:
                if self._use_diagonal_cov:
                    # Diagonal covariance: only use diagonal elements
                    diag_vars = jnp.clip(jax.vmap(jnp.diag)(class_covs), 1e-4, 1e4)
                    base_dist = dist.MultivariateNormal(
                        class_means[y_batch],
                        jax.vmap(jnp.diag)(diag_vars)[y_batch]
                    )
                else:
                    stable_class_covs = self._stable_covariance(class_covs)
                    base_dist = dist.MultivariateNormal(
                        class_means[y_batch],
                        stable_class_covs[y_batch]
                    )
                numpyro.sample("obs", base_dist, obs=X_batch)


    def run_inference(self, X, y, test_X, test_y, log_prefix="", use_correct_scaling_factor=False):
        if X.shape[0] < 1:
            raise ValueError(f"No training samples available for {log_prefix or 'MNGMM inference'}")

        # =====================================================================
        # Gentler learning rate schedule to prevent NaN
        # Old: init_lr → 10x warmup in 100 steps → exponential decay
        # New: init_lr → 3x warmup in 200 steps → slower decay
        # =====================================================================
        init_lr = self.init_lr

        scheduler = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=init_lr, end_value=init_lr*3, transition_steps=200),
                optax.exponential_decay(init_value=init_lr*3, transition_steps=500, decay_rate=0.95),
            ],
            boundaries=[200]
        )

        self.guide = lambda *args, **kwargs: None

        print("Initializing model")

        # Add gradient clipping to prevent NaN from exploding gradients
        optimizer = numpyro.optim.optax_to_numpyro(
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(scheduler),
            )
        )

        self.svi = SVI(self.model, guide=self.guide, optim=optimizer, loss=Trace_ELBO())

        self.svi_state = self.svi.init(random.PRNGKey(0), X = X, y= y, num_classes=self.num_classes, global_params=self.global_params)

        last_state = None
        nan_count = 0
        max_nan_retries = 3  # Allow a few NaN steps before giving up

        for step in tqdm(range(self.num_steps)):

            early_stop_flag, dets = self.calculate_metrics_on_covariances(self.svi.get_params(self.svi_state), increment=self.increment, use_correct_scaling_factor=use_correct_scaling_factor)

            if(self.with_early_stop & early_stop_flag & (last_state is not None) & (step > 1/3 * self.num_steps)):

                self.svi_state = last_state

                early_stop_flag, dets = self.calculate_metrics_on_covariances(self.svi.get_params(self.svi_state), increment=self.increment, use_correct_scaling_factor=use_correct_scaling_factor)
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(self.svi.get_params(self.svi_state), test_X, test_y)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step) 

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_0", dets[1].item(), step)

                print(f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                      f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}, early_stop_flag = {early_stop_flag}")

                break

            last_state = self.svi_state

            self.svi_state, loss = self.svi.update(self.svi_state, X= X, y=y, num_classes=self.num_classes, covs_dets =dets)

            self.writer.add_scalar(f"{log_prefix}/Loss/train", loss.item(), step) 
            
            if step % 100 == 0:
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(self.svi.get_params(self.svi_state), test_X, test_y)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step) 

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_1", dets[1].item(), step)

                print(f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                      f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}")

            if jnp.isnan(loss):
                nan_count += 1
                print(f"NaN loss detected (attempt {nan_count}/{max_nan_retries})")
                if last_state is not None:
                    self.svi_state = last_state
                if nan_count >= max_nan_retries:
                    print("Early stopping due to repeated NaN loss")
                    break
                continue

            prev_loss = loss

        return self.svi.get_params(self.svi_state)


    def pre_processing(self, features, labels):
        features = np.asarray(features)

        if self.global_params is not None:
            param_dim = int(np.asarray(self.global_params["class_means"]).shape[1])
            if features.shape[1] == param_dim:
                return features, labels

        if self.pca is None:
            max_components = min(features.shape[0] - 1, features.shape[1])
            if max_components < 1:
                raise ValueError(
                    f"Need at least 2 samples for PCA, got {features.shape[0]}"
                )

            n_components = min(self.num_dim, max_components)
            if n_components != self.num_dim:
                print(
                    f"Reducing PCA n_components from {self.num_dim} to {n_components} "
                    f"for {features.shape[0]} samples and {features.shape[1]} features"
                )

            self.pca = PCA(n_components=n_components)
            features = self.pca.fit_transform(features)
        else:
            features = self.pca.transform(features)
        return features, labels


    def train(self, features, labels, test_features, test_labels, current_stage):
        features, labels = self.pre_processing(features, labels)
        raw_features = features.copy()
        raw_labels = labels.copy()

        # =====================================================================
        # Auto-detect whether to use diagonal covariance
        # Rule: if min samples per class < 2 * feature_dim, use diagonal
        # =====================================================================
        unique_labels, label_counts = np.unique(labels.astype(int), return_counts=True)
        min_samples = label_counts.min() if len(label_counts) > 0 else 0
        num_dim = features.shape[1]
        
        if min_samples < 2 * num_dim:
            self._use_diagonal_cov = True
            print(f"Auto-enabled diagonal covariance (min samples/class={min_samples}, dim={num_dim})")
        else:
            self._use_diagonal_cov = False
    
        if self.global_params is not None:
            progress = self.current_stage / (self.max_stage + 1e-8)
            n_samples = int((0.1 + 0.2 * progress) * len(raw_features))
            replay_x, replay_y = self.sample_old_prototypes_happy(n_samples, temp=0.1)
    
            if replay_x is not None and len(replay_x) > 0:
                features = np.vstack([raw_features, replay_x])
                labels = np.concatenate([raw_labels, replay_y])
                print(f"HAPPY prototype replay added: {len(replay_y)} samples")
    
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        labels = labels.astype(int)

        self.params = self.run_inference(
            jnp.array(features),
            jnp.array(labels),
            jnp.array(test_features),
            jnp.array(test_labels),
            log_prefix=f"stage_{current_stage}_Flearning",
            use_correct_scaling_factor=False
        )
    
        if self.global_params is not None:
            novel_idx = raw_labels >= self.label_offset
            print(f"Number of Pseudo-Novel Samples: {novel_idx.sum()} / {len(raw_features)}")
    
            novel_features = raw_features[novel_idx]
            novel_labels = raw_labels[novel_idx]
    
            replay_x, replay_y = self.sample_old_prototypes_happy(
                n_samples=int(0.05 * len(raw_features)),
                temp=0.1
            )
    
            if replay_x is not None and len(replay_x) > 0:
                features = np.vstack([novel_features, replay_x])
                labels = np.concatenate([novel_labels, replay_y])
            else:
                features = novel_features
                labels = novel_labels

            if len(features) < 1:
                print(f"Skipping stage {current_stage} S-learning because no samples were selected")
                self.global_params = copy.deepcopy(self.params)
                return
    
            self.params = self.run_inference(
                jnp.array(features),
                jnp.array(labels.astype(int)),
                jnp.array(test_features),
                jnp.array(test_labels),
                log_prefix=f"stage_{current_stage}_Slearning",
                use_correct_scaling_factor=self.use_correct_scaling_factor
            )
    
        self.global_params = copy.deepcopy(self.params)


    def calculate_acc(self, params, test_features, test_labels):

        pred_test_labels, _ = self._predict(jnp.array(test_features), params)

        correct = jnp.sum(pred_test_labels == test_labels).tolist()
        
        return correct, len(test_features), 100. * (correct / float(len(test_features)))

    def test(self, test_features, test_labels):

        test_features, test_labels = self.pre_processing(test_features, test_labels)

        # pred_test_labels, _ = self._predict(jnp.array(test_features), self.params)
        pred_test_labels, _ = self._predict(
            jnp.array(test_features),
            self.params,
            happy_bias=False
        )

        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        return correct, len(test_features), 100. * correct / float(len(test_features))

    def _predict(self, X, params, happy_bias=True):
        class_means = params["class_means"]
        class_covs = params["class_covs"]
        log_probs = []
    
        eye = jnp.eye(class_covs.shape[-1])
    
        for i in range(class_means.shape[0]):
            if self._use_diagonal_cov:
                diag_var = jnp.clip(jnp.diag(class_covs[i]) + 1e-4, 1e-4, 1e4)
                cov = jnp.diag(diag_var)
            else:
                cov = self._stable_covariance(class_covs[i] + 1e-4 * eye)
            mvn = dist.MultivariateNormal(class_means[i], cov)
            log_probs.append(mvn.log_prob(X))
    
        log_probs = jnp.stack(log_probs, axis=-1)
    
        if happy_bias and self.global_params is not None and self.label_offset > 0:
            probs = jax.nn.softmax(log_probs, axis=-1)
    
            p_old = jnp.sum(probs[:, :self.label_offset], axis=-1)
            p_new = jnp.sum(probs[:, self.label_offset:], axis=-1)
    
            p_old_mean = jnp.mean(p_old)
            p_new_mean = jnp.mean(p_new)
    
            confidence = jnp.mean(jnp.max(probs, axis=-1))
            uncertainty = 1.0 - confidence
    
            # adaptive dataset-agnostic boost
            bias_gap = jnp.clip(p_old_mean - p_new_mean, 0.0, 1.0)
    
            progress = getattr(self, "current_stage", 1) / (
                getattr(self, "max_stage", 5) + 1e-8
            )
    
            stage_decay = 0.7 + 0.3 * (1.0 - progress)
    
            alpha = 0.25
            new_boost = alpha * bias_gap * stage_decay * (1.0 - uncertainty)
    
            # safety cap: tránh case mọi sample bị đẩy thành novel
            new_boost = jnp.clip(new_boost, 0.0, 0.18)
    
            log_probs = log_probs.at[:, self.label_offset:].add(new_boost)
    
        return jnp.argmax(log_probs, axis=-1), log_probs


    def _set_label_offset(self, label_offset):
        self.label_offset = label_offset

    def _correct_scaling_factors(self, n, total):
        return jnp.sqrt((n) / (total + n))

    def calculate_metrics_on_covariances(self, params, increment, use_correct_scaling_factor):
        class_covs = self._stable_covariance(params["class_covs"])

        early_stop_flag = False

        if self.global_params is None:
            return early_stop_flag, [jnp.ones(1),
                                     jnp.ones(1)]

        global_class_covs = self._stable_covariance(self.global_params["class_covs"])

        if not use_correct_scaling_factor:
            dets = [(jax.vmap(jnp.linalg.det)(global_class_covs[:self.num_base])).mean(), 
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset: self.label_offset + increment])).mean()]
            scaling_factor = (self.label_offset + increment) / self.num_base

        else:
            dets = [(jax.vmap(jnp.linalg.det)(global_class_covs[:self.label_offset])).mean(), 
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset: self.label_offset + increment])).mean()]
            scaling_factor = self._correct_scaling_factors(increment, self.label_offset)

        if(not jnp.isnan(dets[0])):
            if((dets[0] > 1) & (dets[1] > scaling_factor * dets[0])):
                early_stop_flag = True
            if((dets[0] < 1) & (dets[1] < dets[0] / scaling_factor)):
                early_stop_flag = True
        
        return early_stop_flag, dets
    
    def sample_old_prototypes_happy(self, n_samples=1000, temp=0.1):
        if self.global_params is None or self.label_offset <= 0:
            return None, None

        means = np.array(self.global_params["class_means"][:self.label_offset])
        covs = self._stable_covariance_np(np.array(self.global_params["class_covs"][:self.label_offset]))

        # hardness = mean cosine similarity to other class means
        norm_means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
        sim = norm_means @ norm_means.T
        np.fill_diagonal(sim, 0)
        hardness = sim.mean(axis=1)

        prob = np.exp(hardness / temp)
        prob = prob / prob.sum()

        sampled_classes = np.random.choice(
            self.label_offset,
            size=n_samples,
            replace=True,
            p=prob
        )

        xs, ys = [], []
        for c in sampled_classes:
            if self._use_diagonal_cov:
                # Sample from diagonal covariance
                diag_var = np.clip(np.diag(covs[c]), 1e-4, 1e4)
                x = np.random.multivariate_normal(
                    means[c],
                    np.diag(diag_var + 1e-4)
                )
            else:
                x = np.random.multivariate_normal(
                    means[c],
                    covs[c] + 1e-4 * np.eye(covs[c].shape[0])
                )
            xs.append(x)
            ys.append(c)

        return np.array(xs), np.array(ys)

    def run(self, features, labels, test_features, test_labels, current_stage, testing_set):

        # self.train(features, labels, test_features, test_labels, current_stage)
        self.current_stage = current_stage
        self.max_stage = 5  
        self.train(features, labels, test_features, test_labels, current_stage)

        t = PrettyTable(['TestSet','Correct', 'Smaples', 'Accuracy'])

        correct, total, acc = self.test(testing_set['test_all']._x, testing_set['test_all']._y)
        t.add_row(["All", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/All", acc, current_stage)

        correct, total, acc = self.test(testing_set['test_old']._x, testing_set['test_old']._y)
        t.add_row(["Old", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/Old", acc, current_stage)

        correct, total, acc = self.test(test_features, test_labels)
        t.add_row(["New", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/Novel", acc, current_stage)

        correct, total, acc = self.test(testing_set['known_test']._x, testing_set['known_test']._y)
        t.add_row(["S0",correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/S0", acc, current_stage)

        print(t)

        # save the class means , covariances and supports to numpy files
        save(os.path.join(self.save_dir, "class_means.npy"), np.array(self.params["class_means"]))
        save(os.path.join(self.save_dir, "class_covariances.npy"), np.array(self.params["class_covs"]))
