# Sistema de inferência fuzzy Mamdani

Arquivos:

- `weaning_mamdani.fis`: modelo para MATLAB/Octave (`readfis('weaning_mamdani.fis')`).
- `weaning_mamdani.fcl`: modelo para fuzzylite 4.0 (importar pelo console/Engine).
- `../tables/11_fuzzy_mfs.csv`: universos, termos e parâmetros.
- `../tables/12_fuzzy_regras.csv`: as 81 regras, pontuação e consequente.
- `../tables/13_fuzzy_30casos.csv`: avaliação nos 30 casos.

## Configuração

- Método: Mamdani.
- Entradas: NIF (−60 a 0 cm.H₂O), RR (5 a 60 resp/min), VT (100 a 1000 mL), RSBI (0 a 300 resp/min/L).
- Saída: Potencial de desmame (0 a 100).
- E: mínimo; implicação: mínimo; agregação: máximo; defuzzificação: centroide (COG).
- 3 termos por entrada; 5 termos na saída; 81 regras (produto cartesiano completo).

## Decisão

- Potencial ≥ 60: sucesso (liberar).
- 40 ≤ potencial < 60: cautela (não liberar; reavaliar).
- Potencial < 40: fracasso (manter suporte).

O limiar 60 é uma escolha de política clínica conservadora; 50 é mantido como referência didática.

## Validação

A implementação manual em `src/weaning_lib.py` foi comparada ao `scikit-fuzzy` em TE1, TE3 e TR10; a diferença máxima no centroide foi inferior a 10⁻⁵. O notebook executa essa validação.
