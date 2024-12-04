# term_project

* **MiDaS.py** : Only MiDaS model.

* **Version 2** : Including `DIP_term_project.py` and `combined_image_V2.jpg`
  - `estimate_depth_map`
  - `calculate_edge_density`
  - `combine_maps`
  - `apply_variable_blur`

* **Version 3** : Including `V3.py` and `combined_image_V3.jpg`
  - `compute_foreground_mask`
  - `compute_focus_map_from_mask`
  - `apply_strong_blur`

* **Version 4** : Including `V4.py`, `combined_image_V4_C.jpg`, and `combined_image_V4_M.jpg`
  - Refine `compute_foreground_mask`
    - Visualize the mask after each morphological operation (`binary_edges.jpg`, `dilated.jpg`, `closed.jpg`)
    - Remove shape filters (`aspect_ratio`)
    - Set `min_area` to 1% of the total pixels of the image
  - Modify the parameters of Canny

* **Version 5** : Including `V5.py` and `combined_image_enhanced_blur.jpg`
  - Modify `compute_foreground_mask`
    - Replace `min_area` with `top_ratio` to adaptively select the largest connected components
    - Automatically adjust `min_area` to avoid empty foreground masks
  - Estimate and adjust blur amount
    - Estimate the current blur amount in the image
    - Allow user to input desired blur amount
    - Automatically adjust blur parameters based on desired blur
  - Improve blur effect in `apply_strong_blur`
    - Adjust weight calculation to enhance blur effect
  - Save intermediate results to `output` folder
  - Change language to Traditional Chinese
