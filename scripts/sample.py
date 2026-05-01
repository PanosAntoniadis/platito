"""
Inference script for PLaTITO equilibrium sampling.

Loads a PLaTITO checkpoint (locally or from the HuggingFace Hub), runs
rollout from a provided starting structure and saves the generated coordinates
to disk. The amino acid sequence is read directly from the PDB file.
"""

import os
import torch
import hydra
import mdtraj as md
import lightning as L

from omegaconf import DictConfig, OmegaConf
from platito.models import PLaTITO
from platito.utils.amino_acid_vocab import AA_1_TO_ID, AA_3_TO_1


def load_checkpoint(cfg: DictConfig, device: torch.device) -> PLaTITO:
    """Load a PLaTITO checkpoint from a local path or the HuggingFace Hub.

    Args:
        cfg: Hydra config.
        device: Target torch device.

    Returns:
        PLaTITO model in eval mode on the requested device.
    """
    if cfg.get("checkpoint_path"):
        ckpt_path = cfg.checkpoint_path
        print(f"Loading checkpoint from local path: {ckpt_path}")
    elif cfg.get("hf_repo_id"):
        if not cfg.get("hf_filename"):
            raise ValueError("hf_filename must be set when using hf_repo_id.")
        from huggingface_hub import hf_hub_download

        print(
            f"Downloading checkpoint from HuggingFace Hub: "
            f"{cfg.hf_repo_id}/{cfg.hf_filename}"
        )
        ckpt_path = hf_hub_download(
            repo_id=cfg.hf_repo_id,
            filename=cfg.hf_filename,
        )
    else:
        raise ValueError(
            "Provide either checkpoint_path (local) or "
            "hf_repo_id + hf_filename (HuggingFace Hub)."
        )

    model = PLaTITO.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="sample.yaml",
)
def sample(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(12345, workers=True)

    # Device setup
    if cfg.gpu_device is not None:
        device = torch.device(f"cuda:{cfg.gpu_device}")
        torch.cuda.set_device(cfg.gpu_device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    model = load_checkpoint(cfg, device)

    if cfg.get("logger"):
        import wandb

        wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            entity=cfg.logger.get("entity"),
            dir=cfg.logger.dir,
            mode=cfg.logger.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    if cfg.get("compile", False):
        print(
            "Compiling condition_net and velocity_net with torch.compile ..."
        )
        model.condition_net = torch.compile(model.condition_net)
        model.velocity_net = torch.compile(model.velocity_net)

    if cfg.number_of_steps <= 0:
        raise ValueError(
            f"number_of_steps must be > 0, got {cfg.number_of_steps}."
        )
    if cfg.number_of_trajectories <= 0:
        raise ValueError(
            "number_of_trajectories must be > 0, "
            f"got {cfg.number_of_trajectories}."
        )

    # Load starting structure and extract sequence from topology
    traj = md.load(cfg.pdb_path)
    ca_indices = traj.topology.select("name CA")
    ca_coords = traj.xyz[0, ca_indices, :]
    num_res = len(ca_indices)

    sequence = "".join(
        AA_3_TO_1.get(traj.topology.atom(i).residue.name, "X")
        for i in ca_indices
    )
    residue_ids = [AA_1_TO_ID.get(aa, AA_1_TO_ID["X"]) for aa in sequence]

    # Load sequence embeddings
    sequence_emb = torch.load(cfg.seq_emb_path, weights_only=False)  # [L, D]
    if sequence_emb.shape[0] != num_res:
        raise ValueError(
            f"sequence_emb length ({sequence_emb.shape[0]}) "
            f"!= number of residues ({num_res})."
        )

    temperature = cfg.temperature

    # Build the initial batch by repeating the single starting structure
    x0 = (
        torch.tensor(ca_coords, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(cfg.number_of_trajectories, 1, 1)
    )
    sequence_emb = sequence_emb.unsqueeze(0).repeat(
        cfg.number_of_trajectories, 1, 1
    )

    rest_conditions = {
        "lag": torch.tensor(
            [cfg.step] * cfg.number_of_trajectories, dtype=torch.long
        ).to(device),
        "temp": torch.tensor(
            [temperature] * cfg.number_of_trajectories, dtype=torch.long
        ).to(device),
        "residue_ids": torch.tensor(
            [residue_ids] * cfg.number_of_trajectories, dtype=torch.long
        ).to(device),
        "sequence_emb": sequence_emb.to(device),
    }

    total_time_ns = cfg.step * cfg.number_of_steps
    print(
        f"\n"
        f"{'='*50}\n"
        f"  PLaTITO Sampling\n"
        f"{'='*50}\n"
        f"  Device            : {device}\n"
        f"  Protein           : {cfg.pdb_path}\n"
        f"  Residues          : {num_res}\n"
        f"  Sequence          : {sequence}\n"
        f"  Temperature       : {temperature} K\n"
        f"  Lag step          : {cfg.step} ns\n"
        f"  Trajectories      : {cfg.number_of_trajectories}\n"
        f"  Rollout steps     : {cfg.number_of_steps}\n"
        f"  Total time        : {total_time_ns} ns / trajectory\n"
        f"  ODE method        : {cfg.method} ({cfg.integrator_steps} steps)\n"
        f"  Output            : {cfg.paths.output_dir}\n"
        f"{'='*50}\n"
    )

    generated_coords = model.generate_trajectory(
        x0=x0.to(device),
        ode_steps=cfg.integrator_steps,
        ode_method=cfg.method,
        trajectory_steps=cfg.number_of_steps,
        return_intermediates=cfg.save_intermediate_steps,
        **rest_conditions,
    )

    if cfg.save_intermediate_steps:
        generated_coords = generated_coords.reshape(
            -1, cfg.number_of_steps + 1, num_res, 3
        )
    else:
        generated_coords = generated_coords.reshape(-1, num_res, 3)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    out_path = os.path.join(cfg.paths.output_dir, "generated_coords.pt")
    torch.save(generated_coords, out_path)
    print(f"Saved {generated_coords.shape[0]} trajectories → {out_path}")

    config_path = os.path.join(cfg.paths.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved config → {config_path}")

    if cfg.get("logger"):
        import wandb

        wandb.finish()


if __name__ == "__main__":
    sample()
