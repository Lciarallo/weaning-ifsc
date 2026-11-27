#!/usr/bin/env python3
"""Aprende a Rede 2 e avalia ambas as redes usando o Netica C API oficial.

A discretização é uma entrada produzida pelo Netica Application. Este script
nunca calcula cortes. Veja netica/README_NETICA.md para repetir a etapa gráfica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import weaning_lib as w
import export_models


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-dir", default=os.environ.get("NETICA_API_DIR"), required=not os.environ.get("NETICA_API_DIR"))
    args = parser.parse_args()
    api = Path(args.api_dir).resolve()
    evidence_dir = ROOT / "netica/evidencias"
    train = pd.read_csv(ROOT / "data/weaning_train.csv")
    unsupervised = pd.read_csv(ROOT / "netica/treino_discretizacao.cas", sep="\t")
    pd.testing.assert_frame_equal(unsupervised, train[list(w.VARIABLES)])
    if len(train) != 21:
        raise ValueError("Esperados exatamente os 21 casos de treino originais")

    for split in ("train", "test"):
        df = pd.read_csv(ROOT / f"data/weaning_{split}.csv")
        df[["IDnum", *w.VARIABLES, "Resultado"]].to_csv(
            ROOT / f"netica/casos_{split}_continuos.cas", sep="\t", index=False
        )

    log = []

    def run(command: list[str]) -> str:
        result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        log.append("$ " + " ".join(command).replace(str(ROOT) + "/", "") + "\n" + result.stdout + result.stderr)
        print(result.stdout, end="", flush=True)
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="weaning-netica-build-") as build:
        binary = str(Path(build) / "netica_native")
        run(["gcc", "-Wall", "-Wextra", "-Wno-comment", "-O2", "-I", str(api / "src"),
             "src/netica_native.c", str(api / "lib/64bit/libnetica.a"),
             "-lstdc++", "-lm", "-lpthread", "-lrt", "-o", binary])
        run([binary, "--train", "netica/evidencias/02_discretizacao_automatica.dne",
             "netica/casos_train_continuos.cas", "netica/Weaning_Rede2_Netica.dne"])

        # Registrar o novo arquivo nativo antes de o exportador conferir seus hashes.
        tracked = ["data/weaning_train.csv", "netica/treino_discretizacao.cas",
                   "netica/evidencias/01_entradas_continuas.dne",
                   "netica/evidencias/02_discretizacao_automatica.dne",
                   "netica/Weaning_Rede2_Netica.dne", "src/netica_native.c"]
        tracked += [str(p.relative_to(ROOT)) for p in sorted(evidence_dir.glob("0*.png"))]
        record = {
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "discretization": {
                "software": "Netica Application 7.01", "mode": "Limited Mode",
                "command": "Modify > Auto-Discretize Nodes", "states": 2,
                "rounding_percent": 20, "training_rows": 21,
                "input": "netica/treino_discretizacao.cas",
                "output": "netica/evidencias/02_discretizacao_automatica.dne",
                "data_sources": ["data/weaning_train.csv"],
                "levels": w.NETICA_LEVELS,
            },
            "learning": {"software": "Netica C API 5.04 Linux x86_64",
                         "method": "ReviseCPTsByCaseFile_bn", "alpha": 1,
                         "initial_experience_per_row": 2, "training_rows": 21},
            "sources": ["https://www.norsys.com/download.html",
                        "https://www.norsys.com/netica_c_api.htm",
                        "https://www.norsys.com/WebHelp/NETICA/X_Discretize_or_Combine.htm"],
            "sha256": {p: sha(ROOT / p) for p in tracked},
        }
        manifest = evidence_dir / "execucao.json"
        manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        export_models.export_netica(w.load_data()[0])
        for network in (1, 2):
            model = f"netica/Weaning_Rede{network}_{'Clinica' if network == 1 else 'Netica'}.dne"
            for split, label in (("train", "treino"), ("test", "teste")):
                cases = (f"netica/casos_treino_Rede{network}.csv" if split == "train"
                         else f"netica/casos_teste_Rede{network}_sem_desfecho.csv")
                output = f"netica/evidencias/rede{network}_{label}_probabilidades.csv"
                run([binary, "--predict", model, cases, output])
                tracked.append(output)
        tracked += ["netica/Weaning_Rede1_Clinica.dne", "data/weaning_test.csv"]
        record["sha256"] = {p: sha(ROOT / p) for p in tracked}
        # O teste é usado somente na inferência cega, após discretizar e aprender.
        record["evaluation"] = {"training_rows": 21, "test_rows": 9,
                                "target_entered_as_evidence": False}
        manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        validation = w.validate_against_netica()
        print(validation.to_string(index=False))
        (evidence_dir / "validacao.json").write_text(
            json.dumps(validation.to_dict("records"), indent=2, ensure_ascii=False) + "\n"
        )
        (evidence_dir / "execucao_api.log").write_text("\n".join(log))


if __name__ == "__main__":
    main()
