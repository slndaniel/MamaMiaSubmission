# A Parameter-Efficient Deep Learning Based Model for Segmentation with Radiomic Feature Extraction

Our segmentation [model](https://github.com/slndaniel/MamaMiaSubmission/blob/main/monai/networks/nets/LiteSegTR.py) demonstrates competitive segmentation performance for breast tumours, evaluated with the public [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042/files/). In the first table below, the performance for our model and state-of-the-art models can be found. In the second table, the corresponding pairwise-wilcoxon test with the p-values is given for the Dice and HD95 metrics. Below the tables, the architecture is illustrated, it uses SegResNet as the backbone and replaces the residual blocks with our lightweight residual blocks.

### Comparison of different architectures for 3D breast tumour segmentation  
*Evaluated on the public [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042/files/).*

| Models        | Segmentation Score | Dice   | HD95  | MParams | GFLOPs | GB/Memory | Inference time (sec) |
|---------------|--------------------|--------|-------|---------|--------|-----------|-----------------------|
| nnU-Net       | 82.80              | 79.96  | 21.54    | 30.70   | 555    | 2.436     | **2.58**              |
| SegResNet     | 83.01          | 80.89 | 22.30 | 18.79   | 576    | 1.905     | 12.52                |
| HCMA-UNet     | 81.74              | 79.04  | 23.34 | 2.87    | **38** | 2.053     | 5.96                  |
| Ours | **83.61**           | **80.95**  | **20.61** | **2.61**| 87     | **1.658** | 10.83                 |


### Wilcoxon-signed-rank test  
*P-values for pairwise comparisons between models based on Dice and HD95 scores. Bold values indicate statistical significance at α = 0.05.*  

| **Dice p-values**   | SegResNet | nnU-Net | HCMA-UNet |   | **HD95 p-values** | SegResNet | nnU-Net | HCMA-UNet |
|---------------------|-----------|---------|-----------|---|-------------------|-----------|---------|-----------|
| **Ours** | 0.1355    | **0.0374**  | **0.0000** |   | **Ours** | 0.5131    | **0.0373**  | 0.3704    |
| SegResNet           | --        | 0.8780  | **0.0009** |   | SegResNet           | --        | 0.3550  | 0.1692    |
| nnU-Net             |           | --      | **0.0188** |   | nnU-Net             |           | --      | 0.8085    |
| HCMA-UNet           |           |         | --        |   | HCMA-UNet           |           |         | --        |

<img width="2980" height="3195" alt="litesegtrforpaperdf drawio_woFR drawio" src="https://github.com/user-attachments/assets/b6f7aafd-fb6e-482a-9633-9c7e387726d5" />

