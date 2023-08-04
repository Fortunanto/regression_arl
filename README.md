In order to run this project do the following

python train_baseline.py runs base regressor, without modification

python train_no_cqr.py runs basic arl model without CQR

python train.py runs the whole model

in order to change the configuration, one needs to modify the following lines in each file:

centroids = torch.tensor([(1,1),(1,1),(1,1)]).float()
cov = torch.tensor([[[2.0, 1], [1, 2.0]],[[2.0, 1], [1, 2.0]],[[100,10],[10,30]]])
n_samples = [10000,10000,200]

changing the centroids and the sample ratio here changes the test.

install environment.yaml to run the repository.

citations


@article{lahoti2020fairness,
  title={Fairness without demographics through adversarially reweighted learning},
  author={Lahoti, Preethi and Beutel, Alex and Chen, Jilin and Lee, Kang and Prost, Flavien and Thain, Nithum and Wang, Xuezhi and Chi, Ed},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={728--740},
  year={2020}
}


@article{romano2019conformalized,
  title={Conformalized quantile regression},
  author={Romano, Yaniv and Patterson, Evan and Candes, Emmanuel},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}