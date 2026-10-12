#!/usr/bin/env python
"""Publish the gen-3 policy bank to the Hugging Face Hub.

Run from the repo root inside .venv (needs: pip install huggingface_hub python-dotenv).

    # Put HF_TOKEN=hf_... in .env at the repo root (a WRITE token from
    # https://huggingface.co/settings/tokens), or export HF_TOKEN in the shell.
    python src/rl/hf_hub.py --repo-id <hf-user>/seq-vs-sync-policy-bank --dry-run
    python src/rl/hf_hub.py --repo-id <hf-user>/seq-vs-sync-policy-bank
    python src/rl/hf_hub.py --repo-id <hf-user>/seq-vs-sync-policy-bank --verify
    python src/rl/hf_hub.py --repo-id <hf-user>/seq-vs-sync-policy-bank --download /tmp/bank

What goes up (one commit, model repo, private unless --public):

    policies/policy_six_n4_L{5|10}_Tc{us}us.pt   24 torch state dicts
    policies/training_history_*.json             24 per-iteration training records
    summary.csv                                  24-row training summary
    README.md                                    generated model card

Filenames are a lookup contract (src/rl/policy.py resolves a policy by name),
so they go up unchanged under policies/ and --download restores the local
runs/train layout. Nothing on the Hub is ever deleted by this script.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from huggingface_hub import CommitOperationAdd, HfApi

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK_DIR = REPO_ROOT / "runs" / "train" / "policies"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "train" / "summary.csv"

MODEL_CARD = """---
library_name: pytorch
tags:
- reinforcement-learning
- quantum-networking
- entanglement-distribution
---

# Gen-3 link-layer policy bank (sequential vs synchronized swapping)

{n} REINFORCE policies, one per (link length L, memory coherence time T_c)
cell, trained for "Sequential vs. Synchronized Entanglement Swapping under
Reinforcement-Learned Link-Layer Control" (Pitt QN Lab). Each policy runs one
repeater link: it decides whether to keep trying, purify, discard, or deliver
a pair. Policy is a 2-hidden-layer MLP (4 -> 64 -> 64 -> 4) over the per-link
state [F1, F2, p, t], saved as a PyTorch state dict.

## Files

- `policies/policy_six_n4_L{{5|10}}_Tc{{us}}us.pt`: {n} state dicts. Filenames
  are a lookup contract (Tc_us = Tc_mult * L_km * 5); do not rename.
- `policies/training_history_*.json`: per-iteration training record per cell.
- `summary.csv`: one row per cell (convergence flags, J_F, SKR, episode stats).

Load with `PolicyNetwork` from `src/rl/policy.py` in the code repo.

## Cells

{table}

The L = 10 km, T_c = 12500 us cell (mult 250) hit the 1000-iteration cap and is
unconverged; it sits at the regime crossover. All other cells converged.
"""


class PolicyBankHub:
    """One Hugging Face model repo holding the 24-policy bank.

    All Hub traffic goes through the private ``_client`` (an ``HfApi``) built
    at instantiation. Public methods never hand it out.
    """

    EXPECTED_POLICIES = 24
    POLICY_GLOB = "policy_six_n4_L*_Tc*us.pt"
    HISTORY_PREFIX = "training_history_"
    REMOTE_BANK_DIR = "policies"
    REMOTE_SUMMARY = "summary.csv"
    REMOTE_CARD = "README.md"

    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = True) -> None:
        # Token precedence: explicit arg > HF_TOKEN (shell env or repo .env via load_dotenv) > cached login.
        self._client = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
        self.repo_id = repo_id
        self.private = private

    # ---- account / repo ---------------------------------------------------

    def whoami(self) -> str:
        """Authenticated username. Raises if the token is missing or invalid."""
        return self._client.whoami()["name"]

    def ensure_repo(self) -> str:
        """Create the model repo if absent. Never changes an existing repo's visibility."""
        url = self._client.create_repo(
            self.repo_id, repo_type="model", private=self.private, exist_ok=True
        )
        return str(url)

    # ---- local side -------------------------------------------------------

    def collect(
        self, bank_dir: Path = DEFAULT_BANK_DIR, summary_csv: Path = DEFAULT_SUMMARY
    ) -> List[Tuple[Path, str]]:
        """Pair each local file with its remote path. Fails closed on a short or ragged bank."""
        policies = sorted(bank_dir.glob(self.POLICY_GLOB))
        if len(policies) != self.EXPECTED_POLICIES:
            raise RuntimeError(
                f"expected {self.EXPECTED_POLICIES} policies in {bank_dir}, found {len(policies)}"
            )
        pairs: List[Tuple[Path, str]] = []
        for pt in policies:
            history = pt.with_name(pt.name.replace("policy_", self.HISTORY_PREFIX, 1)).with_suffix(".json")
            if not history.exists():
                raise RuntimeError(f"missing training history for {pt.name}: {history.name}")
            pairs.append((pt, f"{self.REMOTE_BANK_DIR}/{pt.name}"))
            pairs.append((history, f"{self.REMOTE_BANK_DIR}/{history.name}"))
        if not summary_csv.exists():
            raise RuntimeError(f"summary csv not found: {summary_csv}")
        pairs.append((summary_csv, self.REMOTE_SUMMARY))
        return pairs

    def build_model_card(self, summary_csv: Path = DEFAULT_SUMMARY) -> str:
        """README.md for the Hub, with the per-cell table read from summary.csv."""
        with summary_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: (float(r["L_km"]), float(r["Tc_us"])))
        table = ["| L (km) | T_c (us) | T_c mult | converged | J_F mean |", "|---|---|---|---|---|"]
        for r in rows:
            table.append(
                f"| {float(r['L_km']):g} | {int(float(r['Tc_us']))} | {float(r['Tc_mult']):g} "
                f"| {r['converged_ok']} | {float(r['JF_mean']):.4f} |"
            )
        return MODEL_CARD.format(n=len(rows), table="\n".join(table))

    # ---- hub side ---------------------------------------------------------

    def upload(
        self,
        bank_dir: Path = DEFAULT_BANK_DIR,
        summary_csv: Path = DEFAULT_SUMMARY,
        commit_message: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Push the bank, summary and model card in one commit. Returns the commit URL."""
        pairs = self.collect(bank_dir, summary_csv)
        card = self.build_model_card(summary_csv)
        plan = [remote for _, remote in pairs] + [self.REMOTE_CARD]
        if dry_run:
            print(f"[dry-run] would upload {len(plan)} files to {self.repo_id} (private={self.private}):")
            for remote in plan:
                print("   ", remote)
            return None
        user = self.whoami()
        self.ensure_repo()
        ops = [
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in pairs
        ]
        ops.append(CommitOperationAdd(path_in_repo=self.REMOTE_CARD, path_or_fileobj=card.encode("utf-8")))
        info = self._client.create_commit(
            repo_id=self.repo_id,
            repo_type="model",
            operations=ops,
            commit_message=commit_message or f"Upload gen-3 policy bank ({len(ops)} files)",
        )
        print(f"uploaded {len(ops)} files as {user}: {info.commit_url}")
        return info.commit_url

    def remote_files(self) -> List[str]:
        """Every path currently in the Hub repo."""
        return sorted(self._client.list_repo_files(self.repo_id, repo_type="model"))

    def verify(
        self, bank_dir: Path = DEFAULT_BANK_DIR, summary_csv: Path = DEFAULT_SUMMARY
    ) -> List[str]:
        """Expected remote paths that are NOT on the Hub (empty list means complete)."""
        expected = {remote for _, remote in self.collect(bank_dir, summary_csv)} | {self.REMOTE_CARD}
        return sorted(expected - set(self.remote_files()))

    def download(self, local_dir: Path) -> Path:
        """Pull the whole repo into local_dir (policies/ lands as a subfolder)."""
        path = self._client.snapshot_download(
            repo_id=self.repo_id, repo_type="model", local_dir=str(local_dir)
        )
        return Path(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="<hf-user-or-org>/<repo-name>")
    ap.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--token", default=None, help="write token (default: $HF_TOKEN from the shell or .env, then cached login)")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--message", default=None, help="commit message")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the upload plan, touch nothing")
    mode.add_argument("--verify", action="store_true", help="check every expected file is on the Hub")
    mode.add_argument("--download", type=Path, metavar="DIR", help="pull the Hub repo into DIR")
    args = ap.parse_args(argv)

    hub = PolicyBankHub(args.repo_id, token=args.token, private=not args.public)
    if args.verify:
        missing = hub.verify(args.bank_dir, args.summary)
        if missing:
            print("missing on the Hub:")
            for m in missing:
                print("   ", m)
            return 1
        print(f"all files present on {args.repo_id}")
        return 0
    if args.download:
        print(f"downloaded to {hub.download(args.download)}")
        return 0
    hub.upload(args.bank_dir, args.summary, commit_message=args.message, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
