import numpy as np

def reconstruct_pdf_from_quantiles(quantiles):
    """
    Reconstructs a stepwise PDF from its quantiles.
    """
    Nq = len(quantiles)
    p_steps = (1.0 / (Nq - 1)) / (quantiles[1:] - quantiles[:-1])
    z_steps = quantiles
    z_steps_extended = np.concatenate(([z_steps[0]-0.001],z_steps,[z_steps[-1]+0.001]))
    p_steps_extended = np.concatenate(([0],p_steps,[0]))
    return z_steps_extended, p_steps_extended