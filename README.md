### This is a README dedicated to the experiments of [Efficient Algorithms for Logistic Contextual Slate Bandits with Bandit Feedback](https://arxiv.org/abs/2506.13163).


To run the files, use the following command:

`python3 main.py --alg_name [] --reward_type []--num_contexts [] --theta_star [] --normalize_theta_star --horizon []--failure_level [] --arm_dim [] --slot_count [] --item_count [] --theta_seed [] --arm_seed [] --reward_seed []`

All the experiments can be reproduced in ```reproducible_experiments.ipynb```.

The following algorithms have been implemented:
- `Slate-GLM-OFU`: Our algorithm
- `Slate-GLM-TS`: Our algorithm
- `Slate-GLM-TS-Fixed`: Our algorithm
- `ada-OFU-ECOLog`: [Faury et al. [2022]](https://arxiv.org/pdf/2201.01985)
- `TS-ECOLog`: [Faury et al. [2022]](https://arxiv.org/pdf/2201.01985)
- `MPS`: [Dimakopoulou et al. [2019]](https://www.ijcai.org/proceedings/2019/0308.pdf)
- Modified version of `Ordered Slate Bandit`: [Kale et al. [2010]](https://papers.nips.cc/paper_files/paper/2010/file/390e982518a50e280d8e2b535462ec1f-Paper.pdf)
- Modified version of `ETC-Slate`: [Rhuggenaath et al. [2020]](https://arxiv.org/pdf/2004.09957)

If you find our work useful, please consider citing us:
```
@InProceedings{pmlr-v286-goyal25a,
  title = 	 {Efficient Algorithms for Logistic Contextual Slate Bandits with Bandit Feedback},
  author =       {Goyal, Tanmay and Sinha, Gaurav},
  booktitle = 	 {Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1533--1568},
  year = 	 {2025},
  editor = 	 {Chiappa, Silvia and Magliacane, Sara},
  volume = 	 {286},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--25 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v286/main/assets/goyal25a/goyal25a.pdf},
  url = 	 {https://proceedings.mlr.press/v286/goyal25a.html}
}
```