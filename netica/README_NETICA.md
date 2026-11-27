# Execução real no Netica Application 7.01 e Netica C API 5.04

A discretização automática da Rede 2, a compilação visual das duas redes, a inferência interativa do caso TE3 e o processamento em lote (`Cases > Process Cases` com `CTRL_Respira.txt`) foram executados no **Netica Application 7.01** (`Limited Mode`). A aprendizagem da Rede 2 e a validação cruzada independente foram executadas pelo **Netica C API 5.04 Linux x86_64**, também em modo limitado oficial, sem chave.

Ambas as redes (`Weaning_Rede1_Clinica.dne` e `Weaning_Rede2_Netica.dne`) utilizam o nó-alvo **`Resultado`** (`Fracasso`, `Sucesso`) e níveis contínuos (`discrete = FALSE` com vetor `levels`) nas quatro entradas (`NIF`, `RR`, `VT`, `RSBI`), garantindo compatibilidade direta com os arquivos originais da disciplina (`Respira_DB_Treinamento_Process_Cases.txt`, `Respira_DB_Teste_Process_Cases.txt` e `CTRL_Respira.txt`).

## 1. Discretização automática da Rede 2 — somente treino

1. Abra `evidencias/01_entradas_continuas.dne` no Netica Application 7.01. A rede inicial contém apenas quatro variáveis contínuas, sem pontos de corte definidos.
2. Selecione os quatro nós (`Ctrl+A`) e escolha **Modify > Auto-Discretize Nodes**.
3. Escolha `treino_discretizacao.cas`: exatamente as quatro entradas dos **21 casos de treino**, sem ID, classe ou casos de teste.
4. Informe **2 estados** e **20% de arredondamento**.
5. Salve como `evidencias/02_discretizacao_automatica.dne` (`File > Save As`).

O próprio Netica produziu:

| Entrada | Níveis salvos pelo Netica | Corte |
|---|---|---:|
| NIF | −60, −25, −10 | −25 |
| RR | 13, 30, 46 | 30 |
| VT | 120, 350, 950 | 350 |
| RSBI | 10, 90, 270 | 90 |

Os intervalos incluem a borda inferior (`[a, b)`); o último inclui também a borda superior. `src/weaning_lib.py` lê esses níveis diretamente do arquivo nativo.

## 2. Capturas reais da interface gráfica (`evidencias/`)

- `01_auto_discretize_arquivo.png` a `04_auto_discretize_resultado.png`: execução passo a passo de `Modify > Auto-Discretize Nodes` no Netica 7.01.
- `05_netica_rede1_clinica.png`: Rede 1 (discretização clínica, 12 faixas) compilada com `Resultado` (56,5% Fracasso / 43,5% Sucesso).
- `06b_netica_rede1_cpt.png`: janela de CPT (`Table > View/Edit`) do nó `NIF` na Rede 1.
- `06_netica_rede1_te3.png`: inferência interativa do caso crítico **TE3** na Rede 1 (`Fracasso = 55,8%`, `Sucesso = 44,2%`).
- `07_netica_rede2_automatica.png` (e `05_rede_aprendida.png`): Rede 2 compilada com `Automatic Updating` ativo (`Resultado` 56,5% / 43,5%, idêntica à captura `Process Cases.jpg` do professor).
- `08_netica_rede2_te3.png`: inferência interativa do caso **TE3** na Rede 2 (`Fracasso = 9,36%`, `Sucesso = 90,6%`).
- `09_netica_process_cases.png`: diálogo do comando `Cases > Process Cases` no Netica 7.01.

## 3. Processamento em lote na interface (`Cases > Process Cases`)

Para reproduzir na interface gráfica exatamente o procedimento do professor:

1. Abra `Weaning_Rede1_Clinica.dne` ou `Weaning_Rede2_Netica.dne` no Netica Application 7.01.
2. Execute **`Network > Compile`** (`F5`) e ative **`Network > Automatic Updating`**.
3. Escolha **`Cases > Process Cases`** e informe:
   - **Control file:** `references/originais/CTRL_Respira.txt`
   - **Input case file:** `references/originais/Respira_DB_Treinamento_Process_Cases.txt` ou `Respira_DB_Teste_Process_Cases.txt`
   - **Output file:** arquivo `.txt` de saída com `Bel(Resultado=Sucesso)` e `Bel(Resultado=Fracasso)`.

As quatro execuções reais estão preservadas em:
- `evidencias/process_cases_rede1_treino.txt` (21/21 classes concordantes com Python; erro máximo `5,09 × 10⁻⁷`)
- `evidencias/process_cases_rede1_teste.txt` (9/9 classes concordantes; erro máximo `5,46 × 10⁻⁷`)
- `evidencias/process_cases_rede2_treino.txt` (21/21 classes concordantes; erro máximo `4,50 × 10⁻⁷`)
- `evidencias/process_cases_rede2_teste.txt` (9/9 classes concordantes; erro máximo `1,74 × 10⁻⁷`)

## 4. Aprendizagem e validação via Netica C API 5.04

Com Python e GCC instalados e a distribuição oficial do [Netica C API para Linux](https://www.norsys.com/netica_c_api.htm):

```bash
.venv/bin/python scripts/validar_netica.py --api-dir /caminho/Netica_API_504
./run_all.sh
```

O script compila `src/netica_native.c`, cria o nó-alvo `Resultado`, aplica suavização de Laplace $\alpha = 1$, chama `ReviseCPTsByCaseFile_bn` com os 21 casos de treino e realiza inferência cega nos 30 casos de cada rede (**60/60 classes concordantes**, erro máximo **9,87 × 10⁻⁸**). Todos os hashes SHA-256 estão registrados em `evidencias/execucao.json`.

Fonte: [Norsys — discretização automática](https://www.norsys.com/WebHelp/NETICA/X_Discretize_or_Combine.htm).  
Downloads: [Application](https://www.norsys.com/download.html), [C API](https://www.norsys.com/netica_c_api.htm).
