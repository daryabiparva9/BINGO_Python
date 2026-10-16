"""
Maximum-entropy priors over DIRECTED graphs on n genes whose OUT-degree
distribution follows a power law  P(k) ~ k^(-gamma)  for k >= K0 (here gamma = 2.5, K0 = 4),
given the mean out-degree MEAN_K (= expected #edges / n, the eta BINGO already uses).
The amplitude of the power law is fixed either by requiring the histogram to join the power law
smoothly at K0 (default, F_TAIL = None) or by an explicit tail fraction F_TAIL = P(k >= K0).

Two constructions (see the accompanying explanation for the derivations):

  Option C  -  constrain the out-degree histogram itself:
               P(S) = prod_i  pi[k_i] / C(n-1, k_i)
               pi = max-entropy out-degree pmf: exact power law from K0 on, binomial-shaped below K0.

  Option B  -  hidden fields: each gene i gets an expected out-degree kappa_i ~ rho(kappa) ∝ kappa^(-gamma),
               then every edge i->j is an independent coin with probability kappa_i / (n-1).
               Only MEAN_K is imposed; the tail fraction is whatever the model produces.
               P(S) = prod_i  B[k_i],   B[k] = int rho(kappa) q^k (1-q)^(n-1-k) dkappa,  q = kappa/(n-1).

Both priors depend on a graph only through its out-degrees and can be evaluated for any
adjacency matrix A (A[i, j] = 1 means an edge i -> j) with log_prior(A).
"""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq
from scipy.stats import binom
     # None: histogram joins the power law smoothly at K0 (no cliff).
                   # a number, e.g. 0.0745: impose P(k >= K0) = F_TAIL exactly instead.
N_SAMPLES = 3
# histogram bins (inclusive ranges), widening with k so the sparse tail is averaged
BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 19), (20, 99)]
BASE_SIZE, SIZE_SCALE = 60, 120       # node area = BASE_SIZE + SIZE_SCALE * sqrt(out-degree)
                 # light = few targets, dark red = hub
K_COLOR_MAX = 50                      # same colour scale in every figure (out-degree 50+ = darkest)
rng = np.random.default_rng()          # put a number in here, e.g. default_rng(1), for reproducible graphs
N_SAMPLES = 3
# histogram bins (inclusive ranges), widening with k so the sparse tail is averaged
BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 19), (20, 99)]
BASE_SIZE, SIZE_SCALE = 60, 120       # node area = BASE_SIZE + SIZE_SCALE * sqrt(out-degree)
                 # light = few targets, dark red = hub
rng = np.random.default_rng()  
N_GENES =100

GAMMA = 2.5        # power-law exponent of the out-degree tail
K0 = 4             # power law is imposed for k >= K0 ("degrees larger than 3")
MEAN_K = 2.0       # mean out-degree = expected #edges / n  (200 edges on 100 genes)
KAPPA_MAX = 50.0   # Option B: largest expected out-degree (n-1 gives a tiny spike at k = n-1)
F_TAIL = None 
def log_binom(n, k):
    """log C(n-1, k) for k = 0..n-1 (number of ways to choose k targets among the other n-1 genes)."""
    return gammaln(n) - gammaln(k + 1) - gammaln(n - k)


# ---------------------------------------------------------------------------------------------
# Option C: maximum entropy subject to
#     <N_k>/n = A k^-gamma  for k >= K0
#     <k_out> = MEAN_K
#     and either  pi_{K0-1} / pi_{K0} = (K0 / (K0-1))^gamma   (smooth join, default)
#             or  sum_{k >= K0} pi_k = F_TAIL                   (fixed tail fraction)
# Below K0 the solution is  pi_k ∝ C(n-1,k) x^k, with one extra multiplier on k = K0-1 for the smooth join.
# ---------------------------------------------------------------------------------------------
class OptionC:
    def __init__(self, n=N_GENES, gamma=GAMMA, k0=K0, mean_k=MEAN_K, f_tail=F_TAIL):
        self.n = n
        k = np.arange(n)
        self.lC = log_binom(n, k)
        tail, low = k >= k0, k < k0
        r = np.where(tail, np.maximum(k, 1.0) ** -gamma, 0.0)

        if f_tail is None:
            # smooth join: below K0-1 binomial-shaped; pi_{K0-1} and the tail lie on one power law A k^-gamma.
            # Unknowns x (low-degree shape) and the split between parts; fixed by normalization and the mean.
            below = k < k0 - 1
            def pmf(log_x, frac_pl):
                w = np.zeros(n)
                b = np.exp(self.lC[below] + k[below] * log_x)
                w[below] = (1 - frac_pl) * b / b.sum()
                pl = np.where(k >= k0 - 1, np.maximum(k, 1.0) ** -gamma, 0.0)
                w[k >= k0 - 1] = frac_pl * pl[k >= k0 - 1] / pl.sum()
                return w
            # continuity also between the binomial part and k0-1: pi_{k0-2}/pi_{k0-1} follows the binomial ratio
            def eqs(log_x):
                # choose frac_pl so the binomial part continues into k0-1 with its own ratio
                pl = np.where(k >= k0 - 1, np.maximum(k, 1.0) ** -gamma, 0.0); S = pl.sum()
                b = np.exp(self.lC[below] + k[below] * log_x); Bs = b.sum()
                ratio = np.exp(self.lC[k0 - 1] - self.lC[k0 - 2] + log_x)        # binomial pi_{k0-1}/pi_{k0-2}
                # (frac_pl * pl[k0-1]/S) = ratio * ((1-frac_pl) * b[-1]/Bs)
                c = ratio * b[-1] / Bs; d = pl[k0 - 1] / S
                frac_pl = c / (d + c)
                return frac_pl, (k * pmf(log_x, frac_pl)).sum() - mean_k
            self.log_x = brentq(lambda lx: eqs(lx)[1], -40.0, 10.0, xtol=1e-14)
            pi = pmf(self.log_x, eqs(self.log_x)[0])
        else:
            pi = np.zeros(n)
            pi[tail] = f_tail * r[tail] / r[tail].sum()            # power law, imposed exactly
            low_mean = (mean_k - (k * pi).sum()) / (1 - f_tail)
            if not 0 < low_mean < k0 - 1:
                raise ValueError(f"MEAN_K / F_TAIL incompatible: low-degree genes would need mean {low_mean:.3f}")
            def low_pmf(log_x):
                w = np.exp(self.lC[low] + k[low] * log_x)
                return w / w.sum()
            self.log_x = brentq(lambda lx: (k[low] * low_pmf(lx)).sum() - low_mean, -50.0, 50.0, xtol=1e-14)
            pi[low] = (1 - f_tail) * low_pmf(self.log_x)

        self.pi = pi                                           # out-degree distribution of one gene
        self.lam = -(np.log(pi) - self.lC)                     # Lagrange multipliers lambda_k (up to a constant)
        self.lam -= self.lam[0]
        self._table = np.log(pi) - self.lC                     # log P(row i | its out-degree k)

    def log_prior(self, A):
        """Exact log P(S) for any directed graph A (n x n, zero diagonal)."""
        return self._table[np.asarray(A).sum(axis=1).astype(int)].sum()

    def sample(self, rng=None):
        """Exact draw: each gene draws its out-degree from pi, then picks its targets uniformly."""
        rng = np.random.default_rng(rng)
        n = self.n
        A = np.zeros((n, n), dtype=np.int8)
        k_out = rng.choice(n, size=n, p=self.pi)
        for i in range(n):
            others = np.delete(np.arange(n), i)
            A[i, rng.choice(others, size=k_out[i], replace=False)] = 1
        return A


# ---------------------------------------------------------------------------------------------
# Option B: hidden fields.  kappa_i ~ rho(kappa) ∝ kappa^-gamma on [kappa_min, n-1]  (standard fitness model)
#   kappa_min is fixed by <k_out> = MEAN_K.  F_TAIL cannot be imposed as well: with independent edges,
#   genes with small kappa randomly reach k >= K0 (binomial noise), so the tail fraction comes out too high.
# ---------------------------------------------------------------------------------------------
class OptionB:
    def __init__(self, n=N_GENES, gamma=GAMMA, mean_k=MEAN_K, kappa_max=KAPPA_MAX, n_quad=20000):
        self.n, self.gamma = n, gamma
        self.kappa_max = kappa_max = float(n - 1) if kappa_max is None else float(kappa_max)
        a = gamma - 1.0

        def pareto_mean(kmin):
            num = (kmin ** (1 - a) - kappa_max ** (1 - a)) / (a - 1)
            den = (kmin ** -a - kappa_max ** -a) / a
            return num / den

        self.kappa_min = brentq(lambda km: pareto_mean(km) - mean_k, 1e-6, kappa_max * 0.999)

        u = (np.arange(n_quad) + 0.5) / n_quad                 # quadrature nodes = quantiles of rho
        q = self._inv_cdf(u) / (n - 1)
        k = np.arange(n)
        # log B[k] = log E_rho[ q^k (1-q)^(n-1-k) ], computed in log space
        logs = k[None, :] * np.log(q[:, None]) + (n - 1 - k)[None, :] * np.log1p(-q[:, None])
        mx = logs.max(axis=0)
        self._table = mx + np.log(np.exp(logs - mx).mean(axis=0))
        self.pi = np.exp(self._table + log_binom(n, k))        # implied out-degree pmf (exact)
        self.theta = lambda kappa: np.log((n - 1 - kappa) / kappa)   # field of a gene with expected degree kappa

    def _inv_cdf(self, u):
        a, lo, hi = self.gamma - 1.0, self.kappa_min, self.kappa_max
        return (lo ** -a - u * (lo ** -a - hi ** -a)) ** (-1.0 / a)

    def log_prior(self, A):
        """Exact log P(S) (fields integrated out) for any directed graph A."""
        return self._table[np.asarray(A).sum(axis=1).astype(int)].sum()

    def sample(self, rng=None, return_kappa=False):
        """Exact draw: kappa_i ~ rho, then each edge i->j independently with probability kappa_i/(n-1)."""
        rng = np.random.default_rng(rng)
        n = self.n
        kappa = self._inv_cdf(rng.random(n))
        A = (rng.random((n, n)) < (kappa / (n - 1))[:, None]).astype(np.int8)
        np.fill_diagonal(A, 0)
        return (A, kappa) if return_kappa else A


# ---------------------------------------------------------------------------------------------
# Baseline: maximum entropy with ONLY the expected number of edges fixed (BINGO's prior)
#   H = theta * m  ->  every ordered pair i -> j is an independent edge with p = <m> / (n(n-1))
#   Out-degrees are Binomial(n-1, p): no hubs.
# ---------------------------------------------------------------------------------------------
class EdgesOnly:
    def __init__(self, n=N_GENES, mean_edges=MEAN_K * N_GENES):
        self.n = n
        self.p = mean_edges / (n * (n - 1))
        self.theta = np.log((1 - self.p) / self.p)
        self.pi = binom.pmf(np.arange(n), n - 1, self.p)          # out-degree pmf

    def log_prior(self, A):
        m = int(np.asarray(A).sum())
        return m * np.log(self.p) + (self.n * (self.n - 1) - m) * np.log1p(-self.p)

    def sample(self, rng=None):
        rng = np.random.default_rng(rng)
        A = (rng.random((self.n, self.n)) < self.p).astype(np.int8)
        np.fill_diagonal(A, 0)
        return A

