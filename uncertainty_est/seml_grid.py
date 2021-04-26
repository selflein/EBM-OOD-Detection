import uuid
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    out_path = Path("./temp") / f"{uuid.uuid4()}.yaml"
    print(OmegaConf.to_yaml(cfg))

    with out_path.open("w") as f:
        f.writelines(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Saving to collection: {cfg.fixed.dataset}")
    run = subprocess.run(
        f"seml {cfg.fixed.dataset} add {str(out_path)}", shell=True, capture_output=True
    )
    print(run.stdout.decode("UTF-8"))
    print(run.stderr.decode("UTF-8"))


if __name__ == "__main__":
    run()
