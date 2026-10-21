# Conformidade com o enunciado

Auditoria após execução real do Netica e regeneração dos artefatos.
Autor informado: **Luiz Eduardo do Nascimento Ciarallo**.

| Item | Requisito | Situação | Evidência |
|---|---|---|---|
| 1 | Contexto, processo, riscos e variáveis | Cumprido | Relatório, Seção 1; notebook |
| 2a | Duas redes Naive Bayes | Cumprido | Python; duas redes .dne; inferência nativa conferida |
| 2a | NIF, RR, VT e RSBI como entradas | Cumprido | Quatro nós nas duas redes |
| 2a | Discretização da Rede 1 baseada em pesquisa | Cumprido | Limiares e justificativas, Seção 3.2 do relatório |
| 2a | Discretização da Rede 2 automaticamente via Netica, a partir do treino | Cumprido | Netica 7.01, Auto-Discretize Nodes; 21 casos, dois estados, 20% de arredondamento; netica/evidencias/ |
| 2b | Avaliar treino e teste, com probabilidades por caso | Cumprido | Tabelas completas; quatro CSVs com probabilidades do motor Netica |
| 2c | Comparar as redes | Cumprido | Matrizes, métricas, Brier, limiares e divergências |
| 3a-i | Inferência fuzzy Mamdani | Cumprido | Quatro entradas; min/min/max; centroide |
| 3a-ii | Definir funções de pertinência e justificá-las | Cumprido | Parâmetros e gráficos; Seção 4.2 |
| 3a-iii | Apresentar variáveis, termos e regras | Cumprido | 81 regras completas; Apêndice A; FIS e FCL |
| 3b | Avaliar os 30 casos | Cumprido | Tabela completa e métricas |
| 3b | Exemplos de sucesso, fracasso e caso duvidoso | Cumprido | TE5, TR10 e TE3; fuzzificação, regras, agregação e centroide |
| 4a | Relatório organizado em PDF | Cumprido | report/Relatorio_Weaning_Bayes_Fuzzy.pdf, com autoria e evidências Netica |
| 4a–b | Encaminhar PDF e arquivos por e-mail | Pendente de envio | Entrega_Weaning_IFSC_Luiz_Ciarallo.zip; sem declaração de envio realizado |
| 4c | Avaliação cobre todos os itens | Verificação acima | Requisitos técnicos cumpridos; encaminhamento separado |

## Verificações realizadas

- Cortes produzidos pelo Netica: NIF −25, RR 30, VT 350 e RSBI 90. O Python lê
  os níveis do .dne nativo, sem cortes calculados em Python como substitutos.
- Rede 2 aprendida pelo Netica C API 5.04 somente no treino. Inferência cega das
  duas redes: 60 classes concordantes; diferença máxima de probabilidade
  9,87 × 10⁻⁸, tolerância 10⁻⁶.
- Rede 1 exportada com sintaxe válida e aceita pelo Netica; Rede 2 salva pelo
  motor nativo e aberta/compilada no Netica Application 7.01.
- FCL e FIS aceitos pelo importador do fuzzylite 4.0. Corrigidos os termos
  Triangle/Trapezoid, os universos e a posição da agregação no FCL.
- Notebook executado; PDF regenerado e inspecionado; nove arquivos originais
  preservados e conferidos por SHA-256.
- scripts/preparar_entrega.py confere autoria, execução do notebook, evidências
  Netica e originais, gera manifesto e testa a integridade do ZIP.

## Limites que permanecem

- São 30 casos históricos, com apenas nove no teste. O total inclui treino e
  não é uma estimativa independente de generalização.
- As probabilidades posteriores não passaram por calibração independente.
- O leave-one-out da Rede 2 mantém os cortes fixos; não é validação aninhada da
  discretização. Não há validação externa nem validação clínica.
- Os vídeos contextualizam o problema; não há exigência explícita de citar
  individualmente todos eles. As fontes efetivamente usadas estão nas referências.
- Gerar um pacote não equivale a enviá-lo. O e-mail permanece pendente e depende
  de uma instrução explícita de envio.
