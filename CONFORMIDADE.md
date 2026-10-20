# Conformidade com o trabalho — auditoria ponto a ponto

Legenda: **✅ cumprido** · **⚠️ cumprido com ressalva** · **❌ não cumprido**.

## Enunciado original

| # | Item do enunciado | Status | Evidência no projeto |
|---|---|---|---|
| 1 | Descrição do processo de interrupção da ventilação mecânica, riscos e variáveis | ✅ | Relatório, Seção 1; notebook, Seção 1 |
| 2a | Proposição e implementação de **duas redes Naive Bayes** | ✅ | `src/weaning_lib.py`; notebook, Seções 2.1–2.6; redes `.dne` |
| 2a | Variáveis de entrada NIF, RR, VT e RSBI | ✅ | Quatro nós nas duas redes |
| 2a | Discretização da Rede 1 baseada em pesquisa/base de conhecimento | ✅ | Relatório, Tabela 4; notebook, Seção 2.3 |
| 2a | Discretização da Rede 2 **automaticamente via Netica**, só com treino | ⚠️ | Cortes da execução de referência reproduzidos, auditados (partições 10/11) e exportados em `.dne`; o Netica não foi executado neste ambiente |
| 2b | Avaliar as redes no treino e no teste | ✅ | Relatório, Tabelas 8–9; notebook, Seção 2.6 |
| 2b | Apresentar probabilidades de sucesso e fracasso por caso | ✅ | Tabelas completas dos 21 casos de treino e 9 de teste |
| 2c | Comparar os resultados das duas redes | ✅ | Matrizes de confusão, métricas, Brier, certeza média, limiares e divergências (Seções 2.7–2.10) |
| 3a-i | Sistema de inferência fuzzy **Mamdani** | ✅ | `src/weaning_lib.py`; notebook, Seção 3.1 |
| 3a-ii | Funções de pertinência baseadas em pesquisa/literatura | ✅ | Relatório, Tabela 15 e Figura 7; justificativa clínica |
| 3a-iii | Variáveis, termos linguísticos de entrada/saída e base de regras | ✅ | 4 entradas, 5 termos de saída, 81 regras; Apêndice A e notebook |
| 3b | Avaliar o sistema na base completa (30 casos) | ✅ | Relatório, Tabela 17 e Figura 11; notebook, Seção 3.5 |
| 3b | Pelo menos um exemplo de sucesso, um de fracasso e um crítico/duvidoso | ✅ | TE5, TR10 e TE3; notebook, Seção 4; Figuras 13–18 |
| 3b | Mostrar fuzzificação, desfuzzificação e regras ativadas | ✅ | Tabelas de μ, α, agregação e centroide nos três casos |
| 4a | Relatório organizado, com gráficos, tabelas e esquemas, em PDF | ✅ | `report/Relatorio_Weaning_Bayes_Fuzzy.pdf`, 34 páginas |
| 4b | Todos os arquivos gerados enviados | ✅ | `Entrega_Weaning_IFSC.zip` (103 arquivos) |
| 4c | Todos os itens do trabalho contemplados | ⚠️ | Conteúdo completo; ressalva de execução do Netica e de citação individual dos vídeos |

## Itens adicionais pedidos no briefing

| Item | Status | Evidência |
|---|---|---|
| Quatro variáveis e análise da dependência RR–VT–RSBI | ✅ | Correlação de Spearman e leave-one-out (Seções 2.2 e 2.10) |
| Tabela comparando cortes clínicos × Netica | ✅ | Relatório, Tabela 6 |
| Tabelas caso a caso com P(Sucesso) e P(Fracasso) das duas redes | ✅ | Relatório, Tabelas 8–9; notebook, Seção 2.6 |
| Discussão de limiar conservador | ✅ | Seções 2.8 e 3.5 |
| Cálculo posterior passo a passo | ✅ | Notebook, Seção 2.6.1 (TE3 na Rede 1) |
| Funções de pertinência com parâmetros exatos e gráficos | ✅ | Tabela 15, Figura 7 e notebook |
| Base de regras completa e consistente | ✅ | 81 regras, Apêndice A; verificação de completude no notebook |
| Tabela dos 30 casos fuzzy, matriz de confusão e métricas | ✅ | Tabelas 17–18 e Figura 12 |
| Região agregada e centroide nos três casos | ✅ | Figuras 14, 16 e 18; Tabelas 22, 25 e 28 |
| Comparação geral Naive Bayes × Fuzzy | ✅ | Seção 5; Tabelas 29–30; Figura 19 |
| Notebook explicando cada passo e mostrando **cada gráfico** | ✅ | 19 figuras exibidas individualmente, com legendas e leitura; 81 células |
| Código completo, limpo e comentado | ✅ | `src/weaning_lib.py`, `plotting.py`, `generate_artifacts.py`, `export_models.py` |
| Reprodução em um comando | ✅ | `./run_all.sh` |

## Ressalvas reais

1. **Netica:** o software é proprietário e não foi executado nesta máquina. A Rede 2 usa os cortes da execução de referência (captura fornecida), reproduzidos e auditados apenas com o treino. O arquivo `netica/Weaning_Rede2_Netica.dne` pode ser aberto no Netica para gerar a captura e reprocessar os casos. Se o professor exigir evidência de execução nativa, este é o único ponto que precisa de uma ação externa.
2. **Vídeos:** os cortes e conceitos são compatíveis com a base de conhecimento e a literatura; os vídeos indicados no enunciado não são citados individualmente. Não há dependência metodológica deles.
3. **Amostra:** 30 casos, 9 no teste. O intervalo de confiança de 95% para 9/9 é 70,1%–100%; as conclusões são exploratórias.
4. **Autoria:** a capa do PDF está sem nome/matrícula porque esses dados não foram fornecidos. Basta informar para inserir.

## Conclusão

Não há item do enunciado sem entrega. Há **duas ressalvas de rastreabilidade** — execução nativa do Netica e citação individual dos vídeos — e uma limitação estatística inerente à base pequena. Para uma submissão estritamente “100% Netica”, basta abrir o `.dne`, rodar `Process Cases`/`Test With Cases` e anexar a captura; o restante está pronto.
