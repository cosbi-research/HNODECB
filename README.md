# Robust parameter estimation and identifiability analysis with Hybrid Neural Ordinary Differential Equations in Computational Biology
Code for the paper ***Robust parameter estimation and identifiability analysis with Hybrid Neural Ordinary Differential Equations in Computational Biology***.

## Code

The project has been developed in Julia 1.9.1 and relies on the packages listed in the [Project](./Project.toml) file. The repository is structured as follows:

### Test Case Settings and Observation Datasets

- The directory [test case settings](test_case_settings) contains the derivative functions of the three benchmark models (both the fully mechanistic and HNODE versions) along with the original parameters, initial states, and training intervals.
- The directory [datasets](datasets) contains the code to generate the *in-silico* training datasets.

### Pipeline
- [Step 1](): the training-validation split is performed at the beginning of the scripts in each of the following steps.
- [Step 2A](step2a_hyperparameter_tuning): code to tune the hyperparameters (both first and second stage).
- [Step 2B](step2b_model_trainer): code to train the HNODE models.
- [Step 3](step3_parameters_identifiability): code to perform the identifiability analysis.
- [Step 4](step4_confidence_intervals): code to estimate the confidence intervals.

### Other analyses
- [supplementary_cell_ap_model_identifiability](supplementary_cell_ap_model_identifiability): identifiability analysis of the parameters in the original cell apoptosis model.
- [supplementary_lotka_volterra_regularization](supplementary_lotka_volterra_regularization): analysis of the regularizer profile with different values of $\alpha$ in the Lotka Volterra HNODE model.
- [supplementary_original_model_glyc_fit_to_noisy_data](supplementary_original_model_glyc_fit_to_noisy_data): parameter fit with the original yeast glycolysis model to the dataset $DS_{0.05}$.
- [supplementary_identifiability_hyperparameter_analysis](supplementary_identifiability_hyperparameter_analysis): analysis of the impact of the choice of $\delta$ and $\epsilon$ on the identifiability results.

### Paper figures and tables
- [paper_latex_table_printer](paper_latex_table_printer): code to generate the tables of the paper.
- [paper_plot_generator](paper_plot_generator): code to generate the plots of the paper.
<!---Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{yazdani2020systems,
  title   = {Systems biology informed deep learning for inferring parameters and hidden dynamics},
  author  = {Yazdani, Alireza and Lu, Lu and Raissi, Maziar and Karniadakis, George Em},
  journal = {PLoS computational biology},
  volume  = {16},
  number  = {11},
  pages   = {e1007575},
  year    = {2020}
}
```
-->

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
