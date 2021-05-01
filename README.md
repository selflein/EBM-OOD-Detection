# Uncertainty Estimation in Deep Neural Networks

### Train CIFAR 10 Baseline (on local GPU)
```
python uncertainty_est/models/ce_baseline.py with configs/ce_baseline_wrn_local.yaml
```

### Train on Slurm Cluster
```
seml {job_name} queue configs/ce_baseline_wrn.yaml
```

```
seml {job_name} start
```

## Acknowledgements

* Implementation of RealNVP from https://github.com/chrischute/real-nvp
* Implementation of Glow from https://github.com/chrischute/glow
