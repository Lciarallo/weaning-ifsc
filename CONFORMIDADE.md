# Conformidade com o enunciado

Auditoria após execução real dos programas gráficos (**Netica Application 7.01** e **QtFuzzyLite 4.0**), validação pela **Netica C API 5.04** e pelo **fuzzylite 4.0**, e regeneração completa dos artefatos.  
Autores: **Luiz Eduardo do Nascimento Ciarallo** e **Renato Deodoro**.

| Item | Requisito | Situação | Evidência |
|---|---|---|---|
| 1 | Contexto, processo, riscos e variáveis | Cumprido | Relatório, Seção 1; notebook Seção 1 |
| 2a | Duas redes Naive Bayes | Cumprido | Python; duas redes `.dne` (nó-alvo `Resultado`); inferência nativa e GUI conferidas |
| 2a | NIF, RR, VT e RSBI como entradas | Cumprido | Quatro nós contínuos com níveis definidos nas duas redes |
| 2a | Discretização da Rede 1 baseada em pesquisa | Cumprido | 12 faixas clínicas justificadas, Seção 3.2 do relatório e captura `05_netica_rede1_clinica.png` |
| 2a | Discretização da Rede 2 automaticamente via Netica, a partir do treino | Cumprido | Netica 7.01, `Auto-Discretize Nodes`; 21 casos, dois estados, 20% de arredondamento (`netica/evidencias/`) |
| 2b | Avaliar treino e teste, com probabilidades por caso | Cumprido | Tabelas completas (treino + teste), Figura 5b, quatro CSVs da API nativa e quatro arquivos reais do `Cases > Process Cases` com `CTRL_Respira.txt` |
| 2c | Comparar as redes | Cumprido | Matrizes, acurácia, acurácia balanceada, sensibilidade, especificidade, Brier, limiares, divergências e capturas TE3 no Netica 7.01 |
| 3a-i | Inferência fuzzy Mamdani | Cumprido | Quatro entradas, saída `Resultado` (`Fracasso`/`Sucesso`); min/min/max; centroide; validado no `scikit-fuzzy`, `fuzzylite 4.0` e **QtFuzzyLite 4.0** |
| 3a-ii | Definir funções de pertinência e justificá-las | Cumprido | Parâmetros, gráficos e 4 mapas `Surface 2D` no QtFuzzyLite 4.0 (`fuzzy/evidencias/02`–`05`); Seção 4.2 |
| 3a-iii | Apresentar variáveis, termos e regras | Cumprido | **12 regras compactas** (4 `Sucesso`, 8 `Fracasso`) cobrindo integralmente as $3^4=81$ combinações; Apêndice A; `.fis`, `.fcl` e `.fll` |
| 3b | Avaliar os 30 casos | Cumprido | Tabela completa, métricas (incl. acurácia balanceada) e discussão do comportamento de platô do centroide Mamdani |
| 3b | Exemplos de sucesso, fracasso e caso duvidoso | Cumprido | TE5 (`72,73`), TR10 (`25,75`) e TE3 (`40,05`); fuzzificação, regras ativadas, agregação, centroide e capturas na aba `Control` do **QtFuzzyLite 4.0** (`06`–`08`) |
| 4a | Relatório organizado em PDF | Cumprido | `report/Relatorio_Weaning_Bayes_Fuzzy.pdf`, com autoria da dupla e evidências visuais do Netica 7.01 e do QtFuzzyLite 4.0 |
| 4a–b | Encaminhar PDF e arquivos por e-mail | Pendente de envio | `Entrega_Weaning_IFSC_Luiz_Ciarallo.zip`; sem declaração de envio realizado |
| 4c | Avaliação cobre todos os itens | Verificação acima | Requisitos técnicos cumpridos; encaminhamento separado |

## Verificações realizadas

- **Conferência dos dados da disciplina:** `verify_discipline_files()` verifica linha a linha que `data/` coincide integralmente com os arquivos originais `Respira_DB_Treinamento.txt`, `Respira_DB_Teste.txt`, `Respira_DB_Treinamento_Process_Cases.txt` e `Respira_DB_Teste_Process_Cases.txt`.
- **Netica Application 7.01 (interface gráfica):**
  - `Modify > Auto-Discretize Nodes` produziu NIF −25, RR 30, VT 350 e RSBI 90. O Python lê os níveis do `.dne` nativo.
  - Ambas as redes (`Weaning_Rede1_Clinica.dne` e `Weaning_Rede2_Netica.dne`) foram compiladas com nó-alvo `Resultado` e `Automatic Updating`, gerando capturas reais (`05_netica_rede1_clinica.png`, `06_netica_rede1_te3.png`, `06b_netica_rede1_cpt.png`, `07_netica_rede2_automatica.png`, `08_netica_rede2_te3.png`, `09_netica_process_cases.png`).
  - `Cases > Process Cases` executado diretamente na GUI com `CTRL_Respira.txt` gerou `process_cases_rede{1,2}_{treino,teste}.txt`: 60/60 classes concordantes com o Python e erro absoluto máximo ≤ 5,5 × 10⁻⁷.
- **Netica C API 5.04:** Rede 2 aprendida somente no treino; inferência cega das duas redes: 60 classes concordantes; diferença máxima 9,87 × 10⁻⁸ (tolerância 10⁻⁶).
- **QtFuzzyLite 4.0 e `fuzzylite 4.0`:** FCL, FIS e FLL validados no motor CLI (`exit code 0`) e carregados na interface gráfica **QtFuzzyLite 4.0** (`qtfuzzylite.exe`), gerando capturas verificadas da aba `Setup` (`Good Rules: 12. Bad Rules: 0.`), 4 mapas `Surface 2D` e simulações na aba `Control` para `TE5` (`72,732`), `TR10` (`25,750`) e `TE3` (`40,048`) em `fuzzy/evidencias/`.
- **Integridade:** Notebook executado sem erros; PDF regenerado e inspecionado visualmente; nove arquivos originais preservados e conferidos por SHA-256 em `scripts/preparar_entrega.py`.

## Limites que permanecem

- São 30 casos históricos, com apenas nove no teste. O total inclui treino e não é uma estimativa independente de generalização.
- As probabilidades posteriores não passaram por calibração independente.
- O leave-one-out da Rede 2 mantém os cortes fixos; não é validação aninhada da discretização. Não há validação externa nem validação clínica.
- Os vídeos contextualizam o problema; não há exigência explícita de citar individualmente todos eles. As fontes efetivamente usadas estão nas referências.
- Gerar um pacote não equivale a enviá-lo. O e-mail permanece pendente e depende de envio explícito pelo aluno.
