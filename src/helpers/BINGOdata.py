import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BINGOParams:
    # Crank-Nicolson step length (should be > 0.05)
    etraj: float = 0.1
    # Other step length parameters
    egamma: float = 0.1
    ea: float = 0.005
    eb: float = 0.005
    ebeta: float = 0.125
    er: float = 0.0001
    eq: float = 0.0002
    # Heuristic temperature for topology sampling (1 = correct sampling)
    Theur: float = 1.0
    # Number of MCMC iterations
    its: int = 3000
    # Number of trajectory steps between measurements (4 is a good default)
    nstep: int = 4
    # Number of pseudo-inputs (accuracy/speed tradeoff)
    nr_pi: int = 50
    # Prior probability for a link existing.
    # link_pr = p/(1-p) where p is the prior probability.
    # If None, defaults to 1/n_genes inside the sampler.
    link_pr: Optional[float] = None


@dataclass
class BINGOData:
    """
    ts   : list of 2D arrays, each shaped [n_genes x n_timepoints]
    Tsam : list of sampling specs — either a scalar dt or a 1D time vector
           scalar dt is expanded to a full time vector in __post_init__
    params : BINGOParams — needed to get nstep for fine grid construction
    """
    ts: list[np.ndarray]
    Tsam: list
    params: BINGOParams
    # Optional fields
    input: Optional[list[np.ndarray]] = None
    sure: Optional[np.ndarray] = None  # known links prior [n_genes x n_genes]

    def __post_init__(self):
        # Normalise Tsam: scalar dt -> full time vector
        Tsam_norm = []
        for i, (ts_i, tsam_i) in enumerate(zip(self.ts, self.Tsam)):
            n_tp = ts_i.shape[1]
            if np.isscalar(tsam_i):
                Tsam_norm.append(np.arange(n_tp) * float(tsam_i))
            else:
                tsam_i = np.asarray(tsam_i).ravel()
                if len(tsam_i) != n_tp:
                    raise ValueError(
                        f"Tsam[{i}] has {len(tsam_i)} points but "
                        f"ts[{i}] has {n_tp} timepoints"
                    )
                Tsam_norm.append(tsam_i)
        self.Tsam = Tsam_norm

        # Gene count consistency check
        n_genes = self.ts[0].shape[0]
        for i, ts_i in enumerate(self.ts):
            if ts_i.shape[0] != n_genes:
                raise ValueError(
                    f"ts[{i}] has {ts_i.shape[0]} genes but ts[0] has {n_genes}"
                )
        self.n_genes = n_genes
        self.n_experiments = len(self.ts)

        # Fine-grid indices and times for trajectory plotting
        nstep = self.params.nstep
        self.plot_index = []  # per experiment: indices into concatenated fine trajectory
        self.fine_times = []  # per experiment: time values on the fine grid
        indlast = 0
        for ts_i, tsam_i in zip(self.ts, self.Tsam):
            n_tp = ts_i.shape[1]
            n_fine = nstep * (n_tp - 1) + 1
            self.plot_index.append(np.arange(n_fine) + indlast)
            indlast += n_fine
            d = np.diff(tsam_i) / nstep
            self.fine_times.append(
                np.concatenate([[0.0], np.cumsum(np.repeat(d, nstep))])
            )
            
            


@dataclass
class BINGOState:
    # Everything here gets updated each MCMC iteration
    q:     np.ndarray  # process noise variance       [n_genes]
    gamma: np.ndarray  # GP signal variance           [n_genes]
    r:     np.ndarray  # measurement noise variance   [n_genes]
    xs:    np.ndarray  # latent trajectory            [n_genes x n_fine_timepoints]
    P:     np.ndarray  # log prior of topology        [n_genes]
    bets:  np.ndarray  # relevance parameters         [n_genes x n_genes+n_in]
    J:     np.ndarray  # cost function values         [n_genes]
    S:     np.ndarray  # topology (adjacency) matrix  [n_genes x n_genes+n_in]
    psi:   np.ndarray  # pseudo-inputs                [n_genes+n_in x nr_pi]
    ma:    np.ndarray  # mean reversion rate          [n_genes]
    mb:    np.ndarray  # mean reversion level         [n_genes]
    Ser:   np.ndarray  # experiment index matrix 