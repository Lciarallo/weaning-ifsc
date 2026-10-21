# Análise de lacunas — 20 de outubro de 2026

Base da revisão: commit `a321edb` e enunciado original em
`references/originais/Trabalho - Bayes e Fuzzy 2026.pdf`.

## Correções autorizadas

1. Executar a discretização automática da Rede 2 no Netica com os 21 casos de
   treino. Preservar a rede anterior como referência, registrar versão, opções,
   entradas e hashes, guardar a rede produzida e evidências reais da execução.
2. Fazer o Python consumir os cortes efetivamente produzidos pelo Netica e
   confrontar as probabilidades das duas implementações. Os nove casos de teste
   só entram na avaliação posterior.
3. Corrigir os identificadores da exportação `.dne` e a sintaxe da exportação
   FCL; validar os arquivos nas implementações correspondentes.
4. Atualizar tabelas, figuras, notebook, relatório e checklist com os resultados
   reais. Retirar afirmações não demonstradas de calibração ou de envio.
5. Preparar o pacote de entrega, preservando os nove arquivos originais e seus
   hashes. A identificação dos autores depende dos nomes e matrículas informados;
   o envio de e-mail não será declarado sem comprovante.

## Critérios de aceite

- Discretização executada pelo Netica, com treino como única fonte de dados.
- Rede e registro de execução auditáveis, sem substituir execução por cortes
  calculados em Python.
- Concordância da inferência Netica/Python dentro da precisão numérica.
- Notebook executado sem erros e PDF regenerado e inspecionado.
- Exportação FCL aceita pelo fuzzylite 4.0.
- Dados originais preservados; limitações da amostra mantidas.

## Resultado da execução

- Netica Application 7.01 executado: dois estados, arredondamento 20%, somente
  as quatro entradas dos 21 casos de treino. Cortes −25, 30, 350 e 90.
- Rede 2 aprendida pelo Netica C API 5.04; 60 inferências das duas redes
  concordantes com o Python, com erro máximo 9,87 × 10⁻⁸.
- FCL e FIS aceitos pelo fuzzylite 4.0; nove originais intactos por SHA-256.
- Autoria confirmada pelo usuário: Luiz Eduardo do Nascimento Ciarallo.
- Notebook executado; PDF com 35 páginas e evidências; ZIP preparado. O envio
  de e-mail permanece separado da geração dos arquivos.
