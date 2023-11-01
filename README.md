# What Planning Problems Can A Relational Neural Network Solve?

**[What Planning Problems Can A Relational Neural Network Solve?](https://concepts-ai.com/p/goal-regression-width/regression_width_neurips2023.pdf)**
<br />
[Jiayuan Mao](http://jiayuanm.com),
[Tomás Lozano-Pérez](https://people.csail.mit.edu/tlp/),
[Joshua B. Tenenbaum](https://web.mit.edu/cocosci/josh.html), and
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/)
<br />
In Conference on Neural Information Processing Systems (NeurIPS) 2023
<br />
[[Paper]](https://concepts-ai.com/p/goal-regression-width/regression_width_neurips2023.pdf)
[[Project Page]](https://concepts-ai.com/p/goal-regression-width/)
[[BibTex]](https://concepts-ai.com/p/goal-regression-width/regression_width_neurips2023.bib)

```
@inproceedings{Mao2023RegressionWidth,
  title={{What Planning Problems Can A Relational Neural Network Solve?}},
  author={Mao, Jiayuan and Lozano-Perez, Tomas and Tenenbaum, Joshua B. and Leslie Pack Kaelbing},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Setup

This repo is based on the released code of "Neural Logic Machines": [GitHub Repo](https://github.com/google/neural-logic-machines).

Please follow the setup instructions in the original repo. To replicate the results:

## Assembly3

```bash
# Depth = 1
jac-run learn_policy_reinforce_find3.py --task find3 --blocks 8 --epochs 100 --depth 1
# Depth = 2
jac-run learn_policy_reinforce_find3.py --task find3 --blocks 8 --epochs 100 --depth 2
```

## BlocksWorld-Clear

```bash
# No recursion
jac-run learn_policy_reinforce_blocks.py --task single-clear --curriculum --blocks 4 --depth 3

# With recursion
jac-run learn_policy_reinforce_blocks.py --task single-clear --curriculum --blocks 4 --depth 6 --recursion=True --io-residual=True
```

## Logistics

```bash
# No recursion
jac-run learn_policy_reinforce_logistics.py --task directed-pathfinding --curriculum --blocks 10 --depth 3

# With recursion
jac-run learn_policy_reinforce_logistics.py --task directed-pathfinding --curriculum --blocks 10 --depth 10 --recursion=True --io-residual=True
```

