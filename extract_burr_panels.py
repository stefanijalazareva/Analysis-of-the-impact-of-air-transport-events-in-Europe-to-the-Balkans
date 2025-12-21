"""
Extract specific panels from burr_parameter_deep_analysis.png
Creates a new figure with panels 1, 2, and 4 (first row: 1 & 2, second row: 4 centered)
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

# Load the original image
img_path = Path('results/burr_deep_analysis/burr_parameter_deep_analysis.png')
img = mpimg.imread(img_path)

# The original is a 3x3 grid
# We need to extract panels at positions [0,0], [0,1], and [1,0]
height, width = img.shape[:2]

# Calculate panel dimensions (assuming equal spacing)
panel_height = height // 3
panel_width = width // 3

# Extract the three panels
panel_1 = img[0:panel_height, 0:panel_width]  # Top-left (0,0)
panel_2 = img[0:panel_height, panel_width:2*panel_width]  # Top-middle (0,1)
panel_4 = img[panel_height:2*panel_height, 0:panel_width]  # Middle-left (1,0)

# Create a new figure with 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Display the panels
axes[0].imshow(panel_1)
axes[0].axis('off')

axes[1].imshow(panel_2)
axes[1].axis('off')

axes[2].imshow(panel_4)
axes[2].axis('off')

# Adjust layout
plt.tight_layout()

# Save the new combined figure
output_path = Path('results/burr_deep_analysis/burr_parameter_relationships.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved combined figure to: {output_path}")

plt.close()
