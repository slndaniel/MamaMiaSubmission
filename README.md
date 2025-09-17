# LiteSegTR: Lightweight Deep Learning model for Tumour Segmentation and Treatment Response Prediction in DCE-MRI Images

Our segmentation model [LiteSegTR](https://github.com/slndaniel/MamaMiaSubmission/blob/main/monai/networks/nets/LiteSegTR.py) demonstrates competitive segmentation performance for breast tumours, evaluated with the public [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042/files/). In the first table below, the performance for LiteSegTR and state-of-the-art models can be found. In the second table, the corresponding pairwise-wilcoxon test with the p-values is given for the Dice and HD95 metrics. Below the tables, the LiteSegTR architecture is illustrated, it uses SegResNet as the backbone and replaces the residual blocks with our lightweight residual blocks.

### Comparison of different architectures for 3D breast tumour segmentation  
*Evaluated on the public [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042/files/). LiteSegTR is our proposed model.*

| Models        | Segmentation Score | Dice   | HD95  | MParams | GFLOPs | GB/Memory | Inference time (sec) |
|---------------|--------------------|--------|-------|---------|--------|-----------|-----------------------|
| nnU-Net       | 82.99              | 79.98  | 21    | 30.70   | 555    | 2.436     | **2.58**              |
| SegResNet     | **83.73**          | **80.48** | **19.52** | 18.79   | 576    | 1.905     | 12.52                |
| HCMA-UNet     | 82.64              | 79.47  | 21.30 | 2.87    | **38** | 2.053     | 5.96                  |
| LiteSegTR (Ours) | 83.26           | 80.37  | 20.79 | **2.61**| 87     | **1.658** | 10.83                 |


### Wilcoxon-signed-rank test  
*P-values for pairwise comparisons between models based on Dice and HD95 scores. Bold values indicate statistical significance at α = 0.05.*  

| **Dice p-values**   | SegResNet | nnU-Net | HCMA-UNet |   | **HD95 p-values** | SegResNet | nnU-Net | HCMA-UNet |
|---------------------|-----------|---------|-----------|---|-------------------|-----------|---------|-----------|
| **LiteSegTR (Ours)** | 0.6096    | 0.5500  | **0.0003** |   | **LiteSegTR (Ours)** | 0.8901    | 0.2000  | 0.2844    |
| SegResNet           | --        | 0.4532  | **0.0025** |   | SegResNet           | --        | 0.2788  | 0.1152    |
| nnU-Net             |           | --      | **0.0278** |   | nnU-Net             |           | --      | 0.9910    |
| HCMA-UNet           |           |         | --        |   | HCMA-UNet           |           |         | --        |




<img width="2086" height="2237" alt="litesegtrforpaperdf drawio" src="https://github.com/user-attachments/assets/df7d798f-fa51-4532-893a-99ea54f8f45d" />
