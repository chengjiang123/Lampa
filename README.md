# **Lampa: Linear Complexity Model Tracking for High Energy Physics**
[![python](https://img.shields.io/badge/-Python_3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

**Structured State Space Models with Locality-Sensitive Hashing for High Energy Physics**  
*Accepted at AISTATS 2025*

---

## **Training**
```bash
python3 tracking_trainer.py -m models
```

## **Model Catalog**
This repository implements state-of-the-art linear-complexity architectures for particle tracking:


| Model Name        | Architecture                          | Reference/Notes |
|-------------------|---------------------------------------|-----------------|
| `dgcnn`, `gravnet`| DGCNN/GravNet                         | [Dynamic Graph CNN (arXiv:1801.07829)](https://arxiv.org/abs/1801.07829) |
| `rwkv`,`rwkv7`    | RWKV/RWKV7                            | [RWKV (arXiv:2305.13048)](https://arxiv.org/abs/2305.13048) |
| `hept`            | HEPT                                  | [Linear Attention Sequence Parallelism (arXiv:2402.12535)](https://arxiv.org/abs/2402.12535) |
| `flatformer`      | FlatFormer                            | [Flattened Window Attention (arXiv:2301.08739)](https://arxiv.org/abs/2301.08739) |
| `fullmamba2` | Mamba                           | [Mamba (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)|
| `gatedelta`       | Gated DeltaNet                        | [Gated DeltaNet (arXiv:2412.06464)](https://arxiv.org/abs/2412.06464) |
| `hmambav1`        | Mamba-b (E2LSH variant)               | Uses E2LSH partitioning before Mamba |
| `hmambav2`        | Mamba-b (LSH embedding)               | Uses LSH embeddings before Mamba  |
| `lshgd`           | Mamba-b (LSH + Linear RNN)            | LSH within linear RNN |
| `fullhybrid2`     | Mamba-a (HEPT/Mamba mix)            | Partial hybrid layer balancing [LSH-Based Mamba Variants (arXiv:2501.16237)](https://arxiv.org/abs/2501.16237) |
| `hydra`     | Mamba-a (Hybrid Hydra)            | Quasi-separable Mixer hybrid layer |
| `fullfullhybrid2` | Enhanced Hybrid SSM                   | Full hybrid configuration |
| `pemamba2`        | Mamba2 + Sliding Window               | Mamba2 fused with flatten/sliding window sorting/grouping |
| `gdlocal1`        | Local Aggregation SSM                 | Experimental local aggregation layer with SSM |


## Key Implementation Notes

### Hyperparameter Sensitivity
Model performance is highly sensitive to:
- Learning rate schedules
- Optimizer configurations  
- Hybrid layer balancing ratios

*Optimal configurations vary significantly between architectures*

### Design Philosophy
- Combining best elements of attention, RNN, and SSM architectures
- Novel hybrid approaches balancing computational efficiency and physics performance
- Efficient models for hit level tasks

--
![Lampa](visual/Lampa.png) 

### Codes

 **Dataset**: Trained on **TrackML** (6-60k hits per event). Preprocessing codes modified from [HEPT](https://github.com/Graph-COM/HEPT/tree/main) and [GNN_Tracking](https://github.com/gnn-tracking/gnn_tracking/tree/main).


 **Acknowledgments**
We thank the **TrackML challenge** for providing the dataset and acknowledge the following papers that inspired this work:

**[1]**: [arXiv:2312.03823](https://arxiv.org/abs/2312.03823)
**[2]**: [arXiv:2402.12535](https://arxiv.org/abs/2402.12535)
**[3]**: [arXiv:2407.13925](https://arxiv.org/abs/2407.13925)


---

Stay connected for updates!
