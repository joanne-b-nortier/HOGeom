# Latent geometry organizes higher-order interactions

1. Set up a virtualenv and install required packages using the file [`requirements.txt`](requirements.txt)

2. Download and install the *geometric-randomization* repository from 
[here](https://github.com/networkgeometry/b-mercator/tree/main).

3. Empirical datasets can be downloaded from the repository [Hypergraphx-data](https://hgx-team.github.io/hypergraphx-data/).

4. Figures for the **Main text** can be generated using code in `code-for-figures-MAIN.ipynb`.

    i. Figures can be directly generated using precomputed results from the `data` folder. 
    
    ii. To reproduce results for **Figure 3a**, data for the numerical integration of $\langle \mathcal{I}^{(3,4)} \rangle$ have been provided in the file [`data/data-fig3-from-wolfram-numeric-integrations.csv`](data/data-fig3-from-wolfram-numeric-integrations.csv) and can be regenerated using the Wolfram notebook [`scaling-of-nestedness-figure3.nb`](scaling-of-nestedness-figure3.nb). 
    
    iii. To reproduce results for **Figure 3b.ii**, data has been provided in the file [`data/overlap_l3_l4_AVERAGED.csv`](data/overlap_l3_l4_AVERAGED.csv) and can be regenerated using the file 
    [`nestedness_whole_network.py`](nestedness_whole_network.py).

6. Figures for the **Supplementary Materials** can be generated using the notebook [`code-for-figues-SUPP.ipynb`](code-for-figues-SUPP.ipynb). 
