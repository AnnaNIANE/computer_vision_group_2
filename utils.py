import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def generate_heatmap(log_file, output_path='output/heatmap.png', width=1280, height=720):
    """Génère une carte thermique à partir du journal de suivi en utilisant Seaborn."""
    # Lire les positions du journal
    df = pd.read_csv(log_file)
    if df.empty:
        return

    # Extraire les centres des bounding boxes
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    centers = df[['center_x', 'center_y']].values

    if len(centers) == 0:
        return

    # Créer une figure et une heatmap avec Seaborn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=df['center_x'], y=df['center_y'], cmap="Reds", fill=True, thresh=0, levels=100)
    plt.title("Heatmap of Object Positions")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Ajuster les limites pour correspondre à la taille de la vidéo
    plt.xlim(0, width)
    plt.ylim(0, height)

    # Sauvegarder la figure
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == "__main__":
    generate_heatmap('output/tracking_log.csv')