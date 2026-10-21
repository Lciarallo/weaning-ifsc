# Execução real no Netica

A discretização da Rede 2 foi executada no **Netica Application 7.01**, em
`Limited Mode`, via Wine em uma tela Xvfb isolada. A aprendizagem da Rede 2 e a
inferência das duas redes foram executadas pelo **Netica C API 5.04 Linux x86_64**,
também em modo limitado oficial, sem chave.

## 1. Discretização automática — somente treino

1. Abra `evidencias/01_entradas_continuas.dne` no Netica Application. A rede inicial
   contém apenas quatro variáveis contínuas, sem pontos de corte definidos.
2. Selecione os quatro nós (`Ctrl+A`) e escolha **Modify > Auto-Discretize Nodes**.
3. Escolha `treino_discretizacao.cas`: exatamente as quatro entradas dos
   **21 casos de treino**, sem ID, classe ou casos de teste.
4. Informe **2 estados** e **20% de arredondamento**. As opções foram fixadas
   antes da avaliação, seguindo o procedimento documentado pela Norsys.
5. Salve como `evidencias/02_discretizacao_automatica.dne` (`File > Save As`).

O próprio Netica produziu:

| Entrada | Níveis salvos pelo Netica | Corte |
|---|---|---:|
| NIF | −60, −25, −10 | −25 |
| RR | 13, 30, 46 | 30 |
| VT | 120, 350, 950 | 350 |
| RSBI | 10, 90, 270 | 90 |

Os intervalos incluem a borda inferior; o último também inclui a borda superior.
NIF = −25 pertence à **segunda faixa**, corrigindo a rotulagem anterior.
`src/weaning_lib.py` lê esses níveis do arquivo nativo. Não existe uma tabela de
cortes fixos alternativa nem cálculo de medianas para substituir o programa.

As capturas `evidencias/01_*.png` a `04_*.png` mostram a seleção do arquivo, as
duas opções e os cortes efetivamente gerados. A captura `05_rede_aprendida.png`
mostra a rede final compilada no Netica 7.01.

## 2. Aprendizagem e validação nativas

Baixe a distribuição oficial do [Netica C API para Linux](https://www.norsys.com/netica_c_api.htm).
Com Python e GCC instalados, execute a partir da raiz do projeto:

```bash
python3 scripts/validar_netica.py --api-dir /caminho/Netica_API_504
./run_all.sh
```

O script compila `src/netica_native.c` contra a biblioteca oficial, lê os níveis
produzidos no passo 1 e cria o nó `Desmame`, pai das quatro entradas. Inicializa
cada linha com probabilidades 0,5/0,5 e experiência 2: um pseudocaso por estado,
equivalente a Laplace α = 1. Depois chama `ReviseCPTsByCaseFile_bn` com
`casos_train_continuos.cas` e salva `Weaning_Rede2_Netica.dne` pelo próprio Netica.

Só depois desse ajuste, faz inferência nos 21 casos de treino e nove de teste das
duas redes. Insere **somente as quatro entradas**, mesmo quando o arquivo inclui
gabarito. As 60 classes concordaram com o Python; o maior erro absoluto em
P(Sucesso) foi **9,87 × 10⁻⁸**, abaixo da tolerância 10⁻⁶.

- `evidencias/execucao.json`: versões, opções, origem dos dados e SHA-256.
- `evidencias/execucao_api.log`: comandos e saída real da API.
- `evidencias/rede{1,2}_{treino,teste}_probabilidades.csv`: inferências nativas.
- `evidencias/validacao.json`: comparação independente com o Python.

O `run_all.sh` verifica essas evidências e regenera os artefatos Python. Preserva
a Rede 2 nativa; não a substitui por uma exportação Python. Se os dados, níveis
ou modelos forem alterados, refaça a etapa correspondente e execute novamente
`validar_netica.py` antes de regenerar o relatório.

## 3. Avaliar as redes na interface

1. Abra `Weaning_Rede1_Clinica.dne` ou `Weaning_Rede2_Netica.dne`.
2. Escolha `Network > Compile` e ative `Network > Automatic Updating`.
3. Use `Cases > Process Cases` com `casos_teste_RedeN_sem_desfecho.csv` para
   inferência, ou `Cases > Test With Cases` com o arquivo `_com_desfecho.csv`.

Na Rede 1, os casos contêm estados categóricos ASCII. Na Rede 2, contêm valores
numéricos contínuos: os próprios níveis do Netica fazem a discretização.

Fonte: [Norsys — discretização automática](https://www.norsys.com/WebHelp/NETICA/X_Discretize_or_Combine.htm).
Downloads: [Application](https://www.norsys.com/download.html),
[C API](https://www.norsys.com/netica_c_api.htm).
