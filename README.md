# Sequential vs synchronized entanglement swapping

Simulation code for the paper **Sequential vs. Synchronized Entanglement
Swapping under Reinforcement-Learned Link-Layer Control** (Pitt Quantum
Networking Lab, University of Pittsburgh).

The question: in a four-link quantum repeater chain, when should the
intermediate nodes swap? Two disciplines are compared.

- **Sequential swapping** builds the chain left to right. As soon as a stored
  partial chain and a fresh pair on the next link coexist, they are swapped
  and the result is stored one tier up. Something is always in progress.
- **Synchronized swapping** waits until every link holds a pair at the same
  moment, then collapses the whole binary swap tree in one tick. Nothing is
  ever stored between swaps.

Each link is run by its own reinforcement-learned agent that decides, tick by
tick, whether to wait, discard, purify or deliver a pair. The two disciplines
are scored on the secret key rate of the end-to-end pairs they deliver, across
a sweep of memory coherence times. The short version of the result: with very
short coherence times sequential swapping delivers nothing, as coherence
improves it overtakes synchronized swapping, and the two meet again once
decoherence stops mattering.

## What is here

```
src/
  constants.py          physical constants and simulation defaults
  physics.py            decoherence, swap, entropy, the three rate metrics, F_req and F0
  rl/
    env.py              WN2M2 link environment (Gymnasium): two memories, four actions
    transitions.py      analytic state transitions for each action
    train.py            REINFORCE trainer with the length-penalty backstop and early stopping
    policy.py           policy network, link agent wrapper, policy bank lookup
    hf_hub.py           upload or download the policy bank on the Hugging Face Hub
  controllers/
    buffers.py          LinkBuffer (delivered pairs) and ChainBuffer (partial chains)
    sequential.py       SequentialController
    synchronized.py     NestedController (the synchronized discipline, see note below)
  simulation/
    engine.py           run one chain config under both controllers, over several seeds
    sweep.py            the Tc sweep grid: 56 configs, SLURM array or single node
    aggregate.py        rebuild summary.csv from per-config JSONs
  analysis/
    plot_training.py    figures from the training summary
    plot_controller.py  figures from the sweep summary
requirements.txt
```

`runs/` and `archive/` are git-ignored. They hold the trained policies, eval
JSONs and logs on a lab machine. The policy bank itself is published on the
Hugging Face Hub, see below.

## Physics in one paragraph

Pairs are Werner states, fully described by their fidelity F. A pair stored
for dt seconds in a memory with coherence time Tc decays as
F -> 1/4 + (F - 1/4) exp(-2 dt / Tc). Swapping two pairs gives
F1 F2 + (1 - F1)(1 - F2)/3. One DEJMPS purification round on two pairs at
F0 = 0.94 reaches 0.9575. The six-state protocol extracts 1 - H(F) secret
bits per pair, which hits zero at F_MIN = 0.8114; for a four-link chain each
link must deliver above F_req(4) = 0.9476 for the end-to-end pair to clear
F_MIN, and that is the delivery gate every link agent is trained against.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the trained policies

The 24 link policies (two link lengths, twelve coherence times each) live in
the public Hugging Face model repo
[Quantum-Lab-Pitt/seq-vs-sync-policy-bank](https://huggingface.co/Quantum-Lab-Pitt/seq-vs-sync-policy-bank). Each
is a PyTorch state dict for a 4 -> 64 -> 64 -> 4 MLP over the link state
[F1, F2, p, t]. Filenames are a lookup contract,
`policy_six_n4_L{L}_Tc{Tc_us}us.pt`, and the eval code resolves a policy by
name, so do not rename them.

```bash
pip install huggingface_hub python-dotenv
python src/rl/hf_hub.py --repo-id Quantum-Lab-Pitt/seq-vs-sync-policy-bank --download runs/train
```

That restores the `runs/train/policies/` layout the eval scripts expect. The
repo is public, so no token is needed to download. A token is only needed to
upload a new bank.

## Run

One chain config, both controllers, ten seeds each:

```bash
python src/simulation/engine.py --link_Ls 5 5 5 5 --Tc_int 0.0025
```

`--link_Ls` is the link length in km per hop. `--Tc_int` is the memory
coherence time in seconds and picks the policy. Each link needs a policy
trained at its (L, Tc), so for a mixed chain both lengths have to exist at
that Tc.

The sweep used in the paper:

```bash
python src/simulation/sweep.py --list                 # print the 56-config grid
python src/simulation/sweep.py --group symmetric_tc   # one group on one machine
python src/simulation/sweep.py --task_id $SLURM_ARRAY_TASK_ID   # one config per array task
python src/simulation/aggregate.py runs/eval/adhoc --out runs/eval/summary.csv
```

The grid sweeps the dimensionless ratio Tc / tau (tau is the light travel
time over one link) so that L = 5 km and L = 10 km chains land on the same
points. The paper's numbers use 200 seeds per config and 5 simulated seconds
per seed.

Train one link policy:

```bash
python src/rl/train.py --F0 0.94 --L 5 --n_links 4 --Tc 0.0025 \
    --name six_n4_L5_Tc2500us --outputs_root runs/train
```

The bank was trained with 10000 episodes per iteration and a cap of 1000
iterations with early stopping on. Without `--outputs_root` a run lands in
`runs/train_scratch/` so it cannot overwrite the bank.

Figures:

```bash
python src/analysis/plot_training.py --csv runs/train/summary.csv --outdir plot/training
python src/analysis/plot_controller.py --csv runs/eval/summary.csv --outdir plot/controller
```

## Metrics

Every emitted end-to-end pair carries its fidelity and the time it landed.
Three rates are reported.

| Column | Definition | Use |
|---|---|---|
| `skr_u_yau` | max(0, 1 - H(mean F)) divided by the mean spacing between deliveries | headline, fair to both controllers |
| `skr_throughput` | sum over pairs of max(0, 1 - H(F_i)) divided by total simulated time | diagnostic |
| `skr_u_tround` | same numerator as the headline, divided by each pair's own assembly time | diagnostic only, not comparable across controllers |

Use the `_yau` columns for any comparison. The assembly-time denominator in
`skr_u_tround` is near zero for synchronized swapping by construction, so that
column favours it no matter what happened.

## Naming

The synchronized controller is the class `NestedController` and the string
`nested` in output files and column names. That was its name while the code
was written and it was kept so existing results still match. In the paper it
is "synchronized swapping".

## Authors

| Name | Affiliation |
|---|---|
| Priyam Srivastava | Department of Informatics and Networked Systems, University of Pittsburgh |
| Akshat R. Sabavat | Department of Information Science, University of Pittsburgh |
| Siddharth Jain | Department of Physics and Astronomy, University of Pittsburgh |
| Alan Scheller-Wolf | Tepper School of Business, Carnegie Mellon University |
| Sridhar Tayur | Tepper School of Business, Carnegie Mellon University |
| David Tipper | Department of Informatics and Networked Systems, University of Pittsburgh |
| Prashant Krishnamurthy | Department of Informatics and Networked Systems, University of Pittsburgh |
| Amy Babay | Department of Informatics and Networked Systems, University of Pittsburgh |
| Kaushik P. Seshadreesan | Department of Informatics and Networked Systems, University of Pittsburgh |

## Citation

Preprint: [arXiv:2605.04047](https://arxiv.org/abs/2605.04047). The arXiv
version uses the older term "simultaneous swapping" for what this repo and
the final paper call synchronized swapping. A proceedings entry will replace
this one once it is out.

```bibtex
@misc{srivastava2026seqsync,
  title         = {Sequential vs. Simultaneous Entanglement Swapping under Optimal Link-Layer Control},
  author        = {Srivastava, Priyam and Sabavat, Akshat R. and Jain, Siddharth and Scheller-Wolf, Alan and Tayur, Sridhar and Tipper, David and Krishnamurthy, Prashant and Babay, Amy and Seshadreesan, Kaushik P.},
  year          = {2026},
  eprint        = {2605.04047},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2605.04047},
  url           = {https://arxiv.org/abs/2605.04047}
}
```

The link-layer MDP and the six-state utility follow Yau et al.; the policy
class here is an MLP rather than their Fourier-basis linear softmax.

## License

MIT. See `LICENSE`.
