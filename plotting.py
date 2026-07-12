import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import nibabel as nib
from nilearn import plotting
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from textwrap import fill

def plot_combined_multimodal_figure(
    binary_dict,
    continuous_dict,
    affine,
    bg_img=None,
    label_rename_dict=None,
    fontsize=12,
    figsize_base=3,
    dot_size_factor=1200, 
    label_wrap_width=12
):
    """Generates a composite figure with binary map slices and a comparison matrix.

        This function creates a two-panel figure:
        1. Top Panel (a): A row of axial brain slices visualizing the binary masks provided.
        2. Bottom Panel (b): An N x N matrix comparing the inputs.
        - Diagonal: Displays the calculated volume of the mask in mm^3.
        - Upper Triangle: Displays the Dice similarity coefficient (and Pearson correlation 
            if continuous data is provided) inside a colored circle.
        - Lower Triangle: Displays a voxel-wise regression scatter plot (if continuous data 
            is provided) or an overlay of the two binary masks.

        Args:
            binary_dict (dict): A dictionary where keys are binary map names 
                (here corresponding with cognititve domains or degree) and values 
                are 3D numpy arrays representing binary masks.
            continuous_dict (dict): A dictionary where keys are continuous map names 
                (matching binary_dict) and values are 3D numpy arrays representing 
                continuous values (e.g., t-maps). Used for regression plots and 
                Pearson correlation.
            affine (numpy.ndarray): The 4x4 affine transformation matrix associated with the 
                image data.
            bg_img (str or nibabel.Nifti1Image, optional): The background image to use for 
                plotting the brain slices. Defaults to None.
            label_rename_dict (dict, optional): A dictionary mapping original keys from 
                binary_dict to display names. Defaults to None.
            fontsize (int, optional): The font size for titles and labels. Defaults to 12.
            figsize_base (int, optional): A base multiplier for calculating the total figure 
                size based on the number of inputs. Defaults to 3.
            dot_size_factor (int, optional): A scaling factor for the size of the circles 
                in the upper triangle of the matrix. Defaults to 1200.
            label_wrap_width (int, optional): The maximum number of characters before wrapping 
                text labels. Defaults to 12.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib figure object containing the 
            combined plots.

        Raises:
            ValueError: If no non-empty keys are found in the binary_dict after filtering.
        """
    
    # --- 1. Key Matching, Sorting & FILTERING ---
    temp_keys = []
    if label_rename_dict:
        for style_key in label_rename_dict.keys():
            if style_key in binary_dict:
                temp_keys.append(style_key)
            elif f"fLNM_{style_key}" in binary_dict:
                temp_keys.append(f"fLNM_{style_key}")
    
    if not temp_keys:
        temp_keys = sorted(list(binary_dict.keys()))

    # FILTER: Remove keys if the map is empty (sum is 0)
    ordered_keys = []
    for k in temp_keys:
        if np.any(binary_dict[k]):  # Returns True if there is at least one non-zero voxel
            ordered_keys.append(k)

    keys = ordered_keys
    n = len(keys)
    if n == 0: raise ValueError("No matching non-empty keys found.")
    
    # Helper to clean and wrap labels
    def get_label(k):
        if not label_rename_dict: 
            raw = k.replace("fLNM_", "")
        else:
            clean_key = k.replace("fLNM_", "")
            raw = label_rename_dict.get(clean_key, label_rename_dict.get(k, clean_key))
        return fill(raw, width=label_wrap_width)

    # --- 2. Global Calculations (Slice, Background) ---
    total_mask = np.zeros(binary_dict[keys[0]].shape, dtype=bool)
    for k in keys:
        total_mask = total_mask | binary_dict[k].astype(bool)
    
    voxel_vol = abs(np.linalg.det(affine[:3, :3]))
    
    # --- 3. Figure Layout ---
    total_height = figsize_base * (n + 1.5) 
    total_width = figsize_base * n
    
    fig = plt.figure(figsize=(total_width, total_height))
    fig.patch.set_facecolor('white') 
    
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, n + 0.5], hspace=0.1)
    
    # ==========================
    # PANEL 1: Row of Brains
    # ==========================
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=gs_main[0], wspace=0.05)
    
    for idx, key in enumerate(keys):
        ax = fig.add_subplot(gs_top[0, idx])
        ax.set_facecolor('white')
        
        img = nib.Nifti1Image(binary_dict[key].astype(np.float32), affine)
        
        # Plot single axial slice
        display = plotting.plot_stat_map(
            img, 
            bg_img=bg_img, 
            axes=ax, 
            display_mode='z',
            cut_coords=[6],
            cmap=ListedColormap(['#1764ab']), # Blue
            colorbar=False, 
            annotate=False,
            draw_cross=False,
            threshold=0.1,
            transparency=0.8,
            black_bg=False,
            radiological=True,   # true-left drawn on RIGHT of image, matching the R/L labels below
            dim=0
        )

        ax.set_title(get_label(key), fontsize=fontsize, fontweight='bold', pad=12, va="center", wrap=True, y=1.05)
        
        if idx == 0:
            inner_ax = display.axes[list(display.axes.keys())[0]].ax
            inner_ax.text(0.02, 0.9, 'R', transform=inner_ax.transAxes, fontweight='bold', color='black')
            inner_ax.text(0.98, 0.9, 'L', transform=inner_ax.transAxes, fontweight='bold', color='black', ha='right')

    # ==========================
    # SEPARATOR LINE
    # ==========================
    # Place the separator dynamically in the gap between panel a (brains) and panel b (matrix),
    # biased a bit toward the matrix, so it scales with the number of panels n instead of a fixed
    # y that overlaps panel a when there are fewer significant domains.
    _pa = gs_main[0].get_position(fig); _pb = gs_main[1].get_position(fig)
    _y_sep = _pb.y1 + 0.22 * (_pa.y0 - _pb.y1)
    line = Line2D([0.2, 0.8], [_y_sep, _y_sep],
                  transform=fig.transFigure,
                  color='lightgrey',
                  linewidth=2.5,
                  zorder=10)
    fig.add_artist(line)

    # ==========================
    # PANEL 2: Matrix
    # ==========================
    gs_bottom = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=gs_main[1], wspace=0.1, hspace=0.1)
    
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            ax = fig.add_subplot(gs_bottom[i, j])
            ax.set_facecolor('white')

            # Data Prep
            mask_i = binary_dict[key_i].astype(bool)
            mask_j = binary_dict[key_j].astype(bool)
            inter = (mask_i & mask_j).sum()
            tot = mask_i.sum() + mask_j.sum()
            dice = 2 * inter / tot if tot > 0 else 0.0
            
            pearson = None
            has_continuous = (continuous_dict and key_i in continuous_dict and key_j in continuous_dict)

            if has_continuous:
                c_i = continuous_dict[key_i].flatten()
                c_j = continuous_dict[key_j].flatten()
                valid_mask = (np.isfinite(c_i) & np.isfinite(c_j))
                if valid_mask.sum() > 10:
                    pearson = np.corrcoef(c_i[valid_mask], c_j[valid_mask])[0, 1]

            # --- DIAGONAL: Volume ---
            if i == j:
                vol_vox = mask_i.sum()
                vol_mm = vol_vox * voxel_vol
                ax.text(0.5, 0.5, f"Mask volume:\n{vol_mm:.0f} $mm^3$", 
                        ha='center', va='center', fontsize=fontsize, fontweight='bold')
                ax.axis('off')

            # --- LOWER TRIANGLE: Regression OR Map Overlay ---
            elif i > j:
                if has_continuous:
                    c_i = continuous_dict[key_i].flatten()
                    c_j = continuous_dict[key_j].flatten()
                    mask_valid = np.isfinite(c_i) & np.isfinite(c_j)
                    
                    x_data, y_data = c_j[mask_valid], c_i[mask_valid]
                    if len(x_data) > 5000:
                        idx_rand = np.random.choice(len(x_data), 5000, replace=False)
                        x_plot, y_plot = x_data[idx_rand], y_data[idx_rand]
                    else:
                        x_plot, y_plot = x_data, y_data

                    ax.scatter(x_plot, y_plot, color='#08488e', s=1, alpha=0.3, rasterized=True)
                    
                    try:
                        m, b = np.polyfit(x_data, y_data, 1)
                        line_x = np.linspace(x_data.min(), x_data.max(), 100)
                        ax.plot(line_x, m*line_x + b, color='black', linewidth=1.5)
                    except: pass
                    
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.grid(True, linestyle=':', alpha=0.6)
                
                # --- NO CONTINUOUS DATA: OVERLAY PLOTS ---
                else:
                    img_i = nib.Nifti1Image(binary_dict[key_i].astype(np.float32), affine)
                    img_j = nib.Nifti1Image(binary_dict[key_j].astype(np.float32), affine)

                    # Aesthetic Colors: Blue and Red -> Overlap is Purple
                    cmap_i = ListedColormap(['#CC3333']) # Red
                    cmap_j = ListedColormap(['#0077BB']) # Blue
                    
                    # 1. Base Map (Blue) - Using plot_stat_map with 'transparency'
                    display = plotting.plot_stat_map(
                        img_i, 
                        bg_img=bg_img, 
                        axes=ax, 
                        display_mode='z', 
                        cut_coords=[6], 
                        cmap=cmap_i, 
                        transparency=0.6,
                        annotate=False, 
                        draw_cross=False,
                        black_bg=False,
                        colorbar=False,
                        threshold=0.1,
                        radiological=True   # match the R/L labelling (true-left on image right)
                    )
                    
                    # 2. Overlay Map (Red)
                    display.add_overlay(
                        img_j, 
                        cmap=cmap_j, 
                        transparency=0.6
                    )

            # --- UPPER TRIANGLE: Dots ---
            else:
                ax.axis('off')
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

                norm = plt.Normalize(vmin=0, vmax=1)
                cmap = plt.get_cmap('Blues')
                color_val = cmap(norm(dice))

                ax.scatter([0], [0], s=(figsize_base**2) * dot_size_factor, color=color_val, alpha=1.0)

                text_color = 'white' if dice > 0.5 else 'black'
                label_str = f"Dice\n{dice:.2f}"
                if pearson is not None:
                    label_str += f"\n\nr\n{pearson:.2f}"
                
                ax.text(0, 0, label_str, ha='center', va='center', 
                        fontsize=fontsize, fontweight='bold', color=text_color)

            # --- LABELS for Matrix Edges ---
            if j == 0 and i > -1:
                ax.text(-0.25, 0.5, get_label(key_i), transform=ax.transAxes, 
                        ha='center', va='center', fontsize=fontsize, fontweight='bold', rotation=90)
            
            if i == n - 1:
                ax.text(0.5, -0.25, get_label(key_j), transform=ax.transAxes,
                        ha='center', va='top', fontsize=fontsize, fontweight='bold')

    # ==========================
    # PANEL LABELS (a and b)
    # ==========================
    fig.text(0.1, gs_main[0].get_position(fig).y1, 'a', transform=fig.transFigure,
             fontsize=fontsize + 6, fontweight='bold', va='top')

    fig.text(0.1, gs_main[1].get_position(fig).y1, 'b', transform=fig.transFigure,
             fontsize=fontsize + 6, fontweight='bold', va='top')

    return fig

import numpy as np
import pandas as pd

def calculate_similarity_matrices(binary_dict, continuous_dict=None, keys=None):
    """Calculates similarity matrices (Dice and Pearson) for a set of brain maps.

        Computes a symmetric matrix of Dice similarity coefficients based on binary masks
        and, optionally, a symmetric matrix of Pearson correlation coefficients based on
        continuous map values.

        Args:
            binary_dict (dict): A dictionary where keys are ROI names and values are 3D 
                numpy arrays representing binary masks.
            continuous_dict (dict, optional): A dictionary where keys are ROI names 
                (matching binary_dict) and values are 3D numpy arrays representing 
                continuous values (e.g., probability maps). Defaults to None.
            keys (list of str, optional): A specific list of keys to include in the 
                calculation. If None, all keys in binary_dict are used in sorted order. 
                Defaults to None.

        Returns:
            tuple: A tuple containing:
                - dice_df (pandas.DataFrame): A square DataFrame containing pairwise Dice 
                coefficients.
                - pearson_df (pandas.DataFrame or None): A square DataFrame containing 
                pairwise Pearson correlation coefficients. Returns None if 
                continuous_dict is not provided.
        """
    if keys is None:
        keys = sorted(list(binary_dict.keys()))
    
    n = len(keys)
    dice_matrix = np.zeros((n, n))
    pearson_matrix = np.zeros((n, n)) if continuous_dict else None

    for i, key_i in enumerate(keys):
        # Diagonal Dice is always 1.0
        dice_matrix[i, i] = 1.0
        if pearson_matrix is not None:
            pearson_matrix[i, i] = 1.0
            
        for j in range(i + 1, n):
            key_j = keys[j]
            
            # --- Dice Calculation ---
            mask_i = binary_dict[key_i].astype(bool)
            mask_j = binary_dict[key_j].astype(bool)
            inter = (mask_i & mask_j).sum()
            tot = mask_i.sum() + mask_j.sum()
            dice = 2 * inter / tot if tot > 0 else 0.0
            
            dice_matrix[i, j] = dice
            dice_matrix[j, i] = dice # Symmetrical
            
            # --- Pearson Calculation ---
            if pearson_matrix is not None:
                if key_i in continuous_dict and key_j in continuous_dict:
                    c_i = continuous_dict[key_i].flatten()
                    c_j = continuous_dict[key_j].flatten()
                    valid_mask = (np.isfinite(c_i) & np.isfinite(c_j))
                    
                    if valid_mask.sum() > 10:
                        r = np.corrcoef(c_i[valid_mask], c_j[valid_mask])[0, 1]
                    else:
                        r = np.nan
                    
                    pearson_matrix[i, j] = r
                    pearson_matrix[j, i] = r # Symmetrical

    # Convert to DataFrames for easier indexing/labeling
    dice_df = pd.DataFrame(dice_matrix, index=keys, columns=keys)
    pearson_df = pd.DataFrame(pearson_matrix, index=keys, columns=keys) if pearson_matrix is not None else None
    
    return dice_df, pearson_df
