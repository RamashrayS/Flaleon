import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def setup_plot_style():
    """
    Sets up a clean, professional aesthetic for matplotlib plots.
    """
    plt.rcParams['figure.facecolor'] = '#1e1e24'
    plt.rcParams['axes.facecolor'] = '#1e1e24'
    plt.rcParams['text.color'] = '#f0f0f5'
    plt.rcParams['axes.labelcolor'] = '#f0f0f5'
    plt.rcParams['xtick.color'] = '#c0c0cb'
    plt.rcParams['ytick.color'] = '#c0c0cb'
    plt.rcParams['grid.color'] = '#3c3c45'
    plt.rcParams['axes.edgecolor'] = '#3c3c45'
    plt.rcParams['font.size'] = 11

def plot_alignment(df, save_path=None):
    """
    Plots aligned SoLEXS and HEL1OS lightcurves to inspect overlap.
    """
    setup_plot_style()
    plt.figure(figsize=(12, 6))
    
    # Use UTC-like scale or relative time in hours
    rel_time_hours = (df['TIME'] - df['TIME'].min()) / 3600.0
    
    plt.plot(rel_time_hours, df['solexs_counts'], label='SoLEXS (Soft X-ray)', color='#38bdf8', alpha=0.85, linewidth=1.5)
    plt.plot(rel_time_hours, df['helios_counts'], label='HEL1OS (Hard X-ray)', color='#f43f5e', alpha=0.85, linewidth=1.5)
    
    plt.yscale('log')
    plt.title("Aligned Solar Observations (Aditya-L1)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time since day start (Hours)", fontsize=12)
    plt.ylabel("Counts per Second", fontsize=12)
    plt.legend(framealpha=0.2, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor='#1e1e24')
        print(f"[INFO] Alignment plot saved to {save_path}")
    plt.close("all")

def plot_predictions(df, y_true, y_pred, title="Flare Predictions", save_path=None):
    """
    Plots observations with shaded regions highlighting true and predicted flares.
    """
    setup_plot_style()
    plt.figure(figsize=(12, 6))
    
    rel_time_hours = (df['TIME'] - df['TIME'].min()) / 3600.0
    
    plt.plot(rel_time_hours, df['solexs_counts'], label='SoLEXS Flux', color='#38bdf8', alpha=0.7, linewidth=1.2)
    
    # Plot true flare regions as light green shaded areas
    # Find contiguous regions of flare_now == 1
    y_true_bool = y_true.astype(bool)
    if y_true_bool.any():
        plt.fill_between(rel_time_hours, 0.1, df['solexs_counts'].max() * 2, 
                         where=y_true_bool, color='#10b981', alpha=0.2, label='Ground Truth Flare')
                         
    # Plot predicted flare regions as red outline/hatches or light red shaded areas
    y_pred_bool = y_pred.astype(bool)
    if y_pred_bool.any():
        plt.fill_between(rel_time_hours, 0.1, df['solexs_counts'].max() * 2, 
                         where=y_pred_bool, color='#f43f5e', alpha=0.15, label='Predicted Flare Warning', hatch='//')
                         
    plt.yscale('log')
    plt.ylim(bottom=max(0.1, df['solexs_counts'].min() * 0.5), top=df['solexs_counts'].max() * 2)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time since day start (Hours)", fontsize=12)
    plt.ylabel("Counts per Second", fontsize=12)
    plt.legend(framealpha=0.2, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor='#1e1e24')
        print(f"[INFO] Prediction plot saved to {save_path}")
    plt.close("all")

def plot_feature_importance(importances, feature_names, top_n=15, save_path=None):
    """
    Plots horizontal feature importances.
    """
    setup_plot_style()
    
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(top_n), top_importances[::-1], color='#8b5cf6', edgecolor='#3c3c45', height=0.6)
    
    plt.yticks(range(top_n), top_names[::-1], fontsize=10)
    plt.xlabel("Feature Importance Value", fontsize=12)
    plt.title("Feature Importance Ranking", fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, axis='x', ls="-", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor='#1e1e24')
        print(f"[INFO] Feature importance plot saved to {save_path}")
    plt.close("all")
