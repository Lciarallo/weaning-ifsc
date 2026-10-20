# Uso dos modelos no Netica

Arquivos:

- `Weaning_Rede1_Clinica.dne`: rede com discretização por conhecimento clínico (3 faixas por variável).
- `Weaning_Rede2_Netica.dne`: rede com dois intervalos automáticos por variável, reproduzindo a execução de referência do Netica.
- `casos_treino_Rede1.csv` e `casos_treino_Rede2.csv`: os 21 casos já discretizados (rótulos ASCII), com a coluna `Desmame` (gabarito).
- `casos_teste_*_sem_desfecho.csv`: 9 casos para `Cases > Process Cases` (inferência cega).
- `casos_teste_*_com_desfecho.csv`: 9 casos para `Cases > Test With Cases` (auditoria).

As CPTs e as priors já estão gravadas nos arquivos `.dne`, estimadas exclusivamente
com os 21 casos de treino e suavização de Laplace (alpha = 1). Para reproduzir:

1. Abra o `.dne` no Netica (File > Open).
2. Compile a rede (Network > Compile).
3. Para inferência: Cases > Process Cases > escolha o CSV sem desfecho.
4. Para métricas: Cases > Test With Cases > escolha o CSV com desfecho.
5. Confira as crenças em `Desmame` no painel de nós.

Os cortes da Rede 2 são `[−60,−25) | [−25,−10]`, `[13,30) | [30,46]`,
`[120,350) | [350,950]` e `[10,90) | [90,270]`. Eles foram obtidos apenas
com o treino (medianas/medianas arredondadas; partições 10/11 ou 11/10).
Nenhum valor da base de teste foi usado para defini-los.

A inspeção do `.dne` é opcional: toda a inferência do relatório foi replicada
em Python (`src/weaning_lib.py`) e validada contra `sklearn.CategoricalNB`.
