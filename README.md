# NMR Chemical Shift Assignment with Deep Learning
NMR 1H-15N chemical shift assignment with Deep Learning based only on protein sequence encodings from a protein language model ProtT5.


# Protein Language Model-based Assignment of NMR 1H-15N Chemical Shifts

Adel Schmucklermann, Markus Haak, Tobias Senoner, Burkhard Rost, Janosch Hennig, Michael Heinzinger

Biomolecular nuclear magnetic resonance spectroscopy (NMR) is a powerful technique for obtaining information about structure and dynamics of biological macromolecules and their interactions at atomic resolution and all time scales. A prerequisite is the assignment of 1H-15N chemical shift correlations to corresponding residues of the protein sequence. Traditionally, triple resonance NMR experiments of 1H, 15N, and 13C nuclei are needed to assign 1H-15N chemical shift correlations. While measuring 1H-15N correlations is cheap, fast, and requires low sample concentrations, measuring triple resonance experiments requires high concentrations and long measurement times. This limits throughput and the general applicability of biomolecular NMR. However, a direct and reliable assignment of residues solely based on 1H and 15N shifts is hitherto impossible due to the ambiguity of peak assignments without 13C shifts.

Here, we propose to assign 1H and 15N resonances directly to individual residues through deep learning. First, we filtered experimental 1H and 15N shifts from the Biological Magnetic Resonance Data Bank (BMRB) for high quality annotations by keeping only solution state experiments of poly-L-peptides and removing entries with unusual experimental conditions likely to denature the protein (e.g. pH, temperature and ionic strength). We clustered the remaining data at 30% pairwise sequence identity and split representatives into three sets (train, test, validation). Next, we encoded protein sequences as numerical vector representations (so called embeddings) obtained from the protein Language Model (pLM) ProtT5. On top of these embeddings, we trained a feed-forward neural network (FNN) to directly predict 1H and 15N shifts from sequence. Preliminary results on leveraging the distance between predicted and experimentally measured shifts for peak assignment indicates this to be a promising strategy, which would revolutionize the field of biomolecular NMR spectroscopy.

![Poster_NMR](https://github.com/schmucklermann/NMR_Chemical_Shift_Assignment_with_Deep_Learning/assets/74202191/3b56c53a-4748-4bce-bcd9-732badf53b28)
