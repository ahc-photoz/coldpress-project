import numpy as np
import sys

def reconstruct_pdf_from_quantiles(quantiles):
    """
    Reconstructs a stepwise PDF from its quantiles.
    """
    Nq = len(quantiles)
    # Add a small epsilon to avoid division by zero for delta-like functions
    dz = quantiles[1:] - quantiles[:-1] + 1e-9
    p_steps = (1.0 / (Nq - 1)) / dz
    z_steps = quantiles
    z_steps_extended = np.concatenate(([z_steps[0]-0.001],z_steps,[z_steps[-1]+0.001]))
    p_steps_extended = np.concatenate(([0],p_steps,[0]))
    return z_steps_extended, p_steps_extended
    
def plot_from_quantiles(quantiles, output_filename, markers=None, source_id=None, method='all'):
    """
    Generates and saves a plot of a single PDF from its quantiles.

    Args:
        quantiles (np.ndarray): The array of quantile values for one PDF.
        output_filename (str): The path where the plot file will be saved.
        markers (dict, optional): A dictionary of {name: value} to mark with vertical lines.
        source_id (str, optional): An identifier for the plot title. Defaults to None.
        method (str, optional): PDF reconstruction method ('steps', 'spline', 'all').
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting.", file=sys.stderr)
        return

    from .decode import cdf_to_pdf
    
    plt.figure(figsize=(8, 6))

    if method == 'steps' or method == 'all':
        z_steps, p_steps = reconstruct_pdf_from_quantiles(quantiles)
        plt.step(z_steps[:-1], p_steps, where='post', label='PDF (steps)')

    if method == 'spline' or method == 'all':
        zvector = np.linspace(quantiles[0], quantiles[-1], 500)
        pdf = cdf_to_pdf(quantiles, zvector=zvector, method='spline')
        plt.plot(zvector, pdf, label='PDF (spline)')

    # --- Updated logic to plot markers with colors and styles ---
    if markers:
        # Define lists of styles and colors to cycle through
        linestyles = [':', '--', '-.']
        # Use matplotlib's default color cycle names (C1, C2, etc.) for consistency
        colors = [f'C{i}' for i in range(1, 10)] 

        for i, (name, value) in enumerate(markers.items()):
            if value is not None and np.isfinite(value):
                # Cycle through styles and colors using the index and modulo operator
                current_style = linestyles[i % len(linestyles)]
                current_color = colors[(i+2) % len(colors)]
                
                plt.axvline(
                    x=value,
                    linestyle=current_style,
                    color=current_color,
                    label=f'{name} = {value:.4f}',
                    alpha=0.9
                )

    plt.xlabel('Redshift (z)')
    plt.ylabel('Probability Density P(z)')
    
    title = 'Reconstructed PDF'
    if source_id:
        title += f' for Source {source_id}'
    plt.title(title)
    
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_filename)
    plt.close()