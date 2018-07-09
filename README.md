# The committee machine: Computational to statistical gaps in learning a two-layers neural network

### Benjamin Aubin, Antoine Maillard, Jean Barbier, Florent Krzakala, Nicolas Macris and Lenka Zdeborova


Heuristic tools from statistical physics have been used in the past to locate the phase transitions and compute the optimal learning and generalization errors in the teacher-student scenario in multi-layer neural networks. In this contribution, we provide a rigorous justification of these approaches for a two-layers neural network model called the committee machine. We also introduce a version of the approximate message passing (AMP) algorithm for the committee machine that allows to perform optimal learning in polynomial time for a large set of parameters. We find that there are regimes in which a low generalization error is information-theoretically achievable while the AMP algorithm fails to deliver it; strongly suggesting that no efficient algorithm exists for those cases, and unveiling a large computational gap.

ArXiv link: https://arxiv.org/pdf/1806.05451.pdf


We provide a demo ('Demo_AMP_SE_K=2.ipynb') of the AMP algorithm for the committee machine at K=2 and compare it to its State Evolution.
