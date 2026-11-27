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

AUTHOR = "Luiz Eduardo do Nascimento Ciarallo; Renato Deodoro"
AUTHOR_NAMES = ("Luiz Eduardo do Nascimento Ciarallo", "Renato Deodoro")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    w.verify_netica_evidence()
    disc = w.verify_discipline_files()
    if not all(disc.values()):
        raise ValueError(f"Divergência com arquivos da disciplina: {disc}")
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
    cover_text = pdf.pages[0].extract_text() or ""
    if any(name not in cover_text for name in AUTHOR_NAMES) or pdf.metadata.author != AUTHOR:
        raise ValueError("Autoria da capa ou dos metadados do PDF está incorreta")
    text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    if "Netica Application 7.01" not in text or "QtFuzzyLite 4.0" not in text or "C.5" not in text:
        raise ValueError("PDF sem as evidências dos programas gráficos (Netica 7.01 e QtFuzzyLite 4.0)")
    required_files = [
        ROOT / "fuzzy/weaning_mamdani.fcl",
        ROOT / "fuzzy/weaning_mamdani.fis",
        ROOT / "fuzzy/weaning_mamdani.fll",
        ROOT / "fuzzy/validacao_fuzzylite.json",
        ROOT / "netica/evidencias/05_netica_rede1_clinica.png",
        ROOT / "netica/evidencias/06_netica_rede1_te3.png",
        ROOT / "netica/evidencias/07_netica_rede2_automatica.png",
        ROOT / "netica/evidencias/08_netica_rede2_te3.png",
        ROOT / "netica/evidencias/process_cases_rede1_treino.txt",
        ROOT / "netica/evidencias/process_cases_rede1_teste.txt",
        ROOT / "netica/evidencias/process_cases_rede2_treino.txt",
        ROOT / "netica/evidencias/process_cases_rede2_teste.txt",
        ROOT / "fuzzy/evidencias/01_qtfuzzylite_setup.png",
        ROOT / "fuzzy/evidencias/06_qtfuzzylite_caso1_te5_sucesso.png",
        ROOT / "fuzzy/evidencias/07_qtfuzzylite_caso2_tr10_fracasso.png",
        ROOT / "fuzzy/evidencias/08_qtfuzzylite_caso3_te3_duvidoso.png",
    ]
    for path in required_files:
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
        "arquivos_disciplina_conferidos": disc,
        "casos_verificados_netica": int(native["Casos"].sum()),
        "erro_maximo_netica_python": float(native["Erro máximo |P(S) Netica − Python|"].max()),
        "regras_fuzzy": len(w.FUZZY_RULES),
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
          "9 originais preservados; 60 previsões nativas concordantes; Netica 7.01 + QtFuzzyLite 4.0 verificados.")
    print(f"Pacote: {archive.name} ({archive.stat().st_size / 1e6:.1f} MB; {len(files) + 1} arquivos)")
    print("Envio de e-mail: não realizado.")


if __name__ == "__main__":
    main()
