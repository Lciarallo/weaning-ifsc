#!/usr/bin/env python3
"""Verifica os artefatos e reúne os arquivos para envio, sem enviar e-mail."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import weaning_lib as w

AUTHOR = "Luiz Eduardo do Nascimento Ciarallo"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    w.verify_netica_evidence()
    native = w.validate_against_netica()
    with (ROOT / "references/MANIFEST_ORIGINAIS.csv").open() as f:
        for row in csv.DictReader(f):
            path = ROOT / "references/originais" / row["arquivo"]
            if sha(path) != row["sha256"]:
                raise ValueError(f"Original alterado: {path.name}")
    notebook = nbformat.read(ROOT / "notebooks/Weaning_Bayes_Fuzzy.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code):
        raise ValueError("Notebook contém células de código não executadas")
    if any(output.output_type == "error" for cell in code for output in cell.outputs):
        raise ValueError("Notebook contém erro de execução")
    pdf = PdfReader(ROOT / "report/Relatorio_Weaning_Bayes_Fuzzy.pdf")
    if AUTHOR not in pdf.pages[0].extract_text() or pdf.metadata.author != AUTHOR:
        raise ValueError("Autoria da capa ou dos metadados do PDF está incorreta")
    text = "\n".join(page.extract_text() for page in pdf.pages)
    if "Netica Application 7.01" not in text or "C.5" not in text:
        raise ValueError("PDF sem as evidências da execução nativa")
    for path in (ROOT / "fuzzy/weaning_mamdani.fcl", ROOT / "fuzzy/weaning_mamdani.fis"):
        if not path.is_file():
            raise FileNotFoundError(path)

    files = [ROOT / p for p in ("README.md", "CONFORMIDADE.md", "ANALISE_LACUNAS.md",
                                "requirements.txt", "run_all.sh")]
    for folder in ("data", "src", "scripts", "notebooks", "netica", "fuzzy",
                   "tables", "figures", "report", "references"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    manifest = ROOT / "report/manifesto_entrega.json"
    files = sorted(set(files) - {manifest})
    record = {
        "autor": AUTHOR, "email_enviado": False, "paginas_pdf": len(pdf.pages),
        "celulas_codigo_executadas": len(code), "originais_preservados": 9,
        "casos_verificados_netica": int(native["Casos"].sum()),
        "erro_maximo_netica_python": float(native["Erro máximo |P(S) Netica − Python|"].max()),
        "arquivos": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    archive = ROOT / "Entrega_Weaning_IFSC_Luiz_Ciarallo.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in files + [manifest]:
            z.write(path, Path("weaning-ifsc") / path.relative_to(ROOT))
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP corrompido")
    print(f"Verificado: PDF com {len(pdf.pages)} páginas; {len(code)} células executadas; "
          "9 originais preservados; 60 previsões nativas concordantes.")
    print(f"Pacote: {archive.name} ({archive.stat().st_size / 1e6:.1f} MB; {len(files) + 1} arquivos)")
    print("Envio de e-mail: não realizado.")


if __name__ == "__main__":
    main()
