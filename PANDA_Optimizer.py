import torch
from torch.optim.optimizer import Optimizer

class PANDA(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-3, epsilon=1e-8,
                 alpha=0.5, gamma=0.5, delta=0.1, r_min=0.1, r_max=1.0, sigma=1.0, neighborhood_size=5, tau0=1.0, beta_cooling=0.001):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, epsilon=epsilon,
                        alpha=alpha, gamma=gamma, delta=delta, r_min=r_min, r_max=r_max, sigma=sigma,
                        neighborhood_size=neighborhood_size, tau0=tau0, beta_cooling=beta_cooling)
        super(PANDA, self).__init__(params, defaults)

        # Initialize region and module interaction
        self.region_blocks = {}
        self.module_blocks = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            alpha = group['alpha']
            gamma = group['gamma']
            delta = group['delta']
            r_min = group['r_min']
            r_max = group['r_max']
            sigma = group['sigma']
            neighborhood_size = group['neighborhood_size']
            tau0 = group['tau0']
            beta_cooling = group['beta_cooling']

            params = group['params']
            # Initialize module and region blocks if not present
            if id(params) not in self.module_blocks:
                self.module_blocks[id(params)] = torch.eye(len(params), device=params[0].device)
            if id(params) not in self.region_blocks:
                self.region_blocks[id(params)] = torch.ones(len(params), len(params), device=params[0].device)

            module_block = self.module_blocks[id(params)]
            region_block = self.region_blocks[id(params)]

            # Compute module similarity matrix
            features = []
            for p in params:
                if p.grad is not None:
                    grad_mean = p.grad.mean()
                    grad_var = p.grad.var()
                    features.append(torch.tensor([grad_mean, grad_var], device=p.device))
                else:
                    features.append(torch.zeros(2, device=p.device))

            feature_matrix = torch.stack(features)
            normalized_features = feature_matrix / (feature_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8)
            cosine_sim = normalized_features @ normalized_features.t()
            grad_var_diff_sq = (feature_matrix[:, 1].unsqueeze(1) - feature_matrix[:, 1]).pow(2)
            sim_matrix = cosine_sim * torch.exp(-grad_var_diff_sq / (sigma ** 2))

            # Keep top-k similarities
            k = neighborhood_size
            if k < sim_matrix.size(0):
                _, indices = sim_matrix.topk(k, dim=1, largest=True)
                mask = torch.zeros_like(sim_matrix)
                mask.scatter_(1, indices, 1.0)
                sim_matrix *= mask

            # Update module memory
            module_block.mul_(0.99).add_(sim_matrix, alpha=0.01)
            module_block.fill_diagonal_(1.0)

            # Compute region interactions
            region_block.copy_(module_block.mean(dim=0).unsqueeze(0) * module_block.mean(dim=1).unsqueeze(1))

            for idx, p in enumerate(params):
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                state['step'] += 1

                m, v = state['m'], state['v']

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute importance (confidence measure)
                importance = m.abs().mean() / (v.sqrt().mean() + epsilon)

                # Compute influence score (formerly attention)
                grad_norm = grad.norm(p=2)
                grad_var = grad.var() if not torch.isnan(grad.var()) else torch.tensor(0.0, device=p.device)
                influence_score = torch.sigmoid((grad_norm - grad_var) * gamma)

                # Compute dynamic cooling factor (tau)
                tau_t = tau0 / (1 + beta_cooling * state['step'])

                # Multi-objective adjustment: balance multiple optimization goals
                objective_weight = (alpha * importance + gamma * influence_score + delta * region_block[idx].mean())

                # Compute dynamic range adjustment (r)
                positivity = influence_score * importance
                r = r_min + positivity * (r_max - r_min) * torch.sigmoid(objective_weight)

                dynamic_weight = torch.sigmoid(objective_weight + tau_t)

                # Apply parameter update
                lr_t = lr / (v.sqrt().mean() + epsilon)
                p.add_(-lr_t, m / (dynamic_weight ** r))

        return loss