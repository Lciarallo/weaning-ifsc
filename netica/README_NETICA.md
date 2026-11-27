# Netica 7.01

## Discretização automática da Rede 2 (somente os 21 casos de treino)

1. Abrir `evidencias/01_entradas_continuas.dne`.
2. Selecionar os quatro nós e usar **Modify > Auto-Discretize Nodes** com `treino_discretizacao.cas`.
3. Informar 2 estados e 20% de arredondamento.

Cortes gerados: NIF −25, RR 30, VT 350 e RSBI 90 (`evidencias/02_discretizacao_automatica.dne`).

## Process Cases

1. Abrir `Weaning_Rede1_Clinica.dne` ou `Weaning_Rede2_Netica.dne`.
2. **Network > Compile**.
3. **Cases > Process Cases**, com `references/originais/CTRL_Respira.txt` e `Respira_DB_Treinamento_Process_Cases.txt` ou `Respira_DB_Teste_Process_Cases.txt`.

As saídas estão em `evidencias/process_cases_rede{1,2}_{treino,teste}.txt`. As capturas de tela estão em `evidencias/`.
