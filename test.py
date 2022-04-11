import hydra
import os
import argparse
from pathlib import Path
from argparse import Namespace
from tester import Tester

from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

def pad(str, nb):
    return str + " " * (nb - len(str))

    
@hydra.main(config_path = 'conf', config_name = 'defaults')
def main(args):
    OmegaConf.set_struct(args, False)
    console = Console()
    vis = Syntax(OmegaConf.to_yaml(args), "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis)
    console.print(richPanel)
    Path(args.ckpt_dir).mkdir(parents = True, exist_ok = True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    tester = Tester(args)
    ap_dict, mAP = tester.run()
    console.print(f"{pad('Final mAP', 12)} \t: {mAP * 100 : .2f}%")
    console.print("-" * 25)
    for name, ap in ap_dict.items():
        console.print(f"{pad(name, 12)} \t: {ap * 100 : .2f}%")


if __name__ == '__main__':
    main()