# Sistema de inferência fuzzy Mamdani

Arquivos:

- `weaning_mamdani.fis`: modelo para MATLAB/Octave (`readfis('weaning_mamdani.fis')`).
- `weaning_mamdani.fcl`: modelo para `fuzzylite 4.0` / QtFuzzyLite 4.0 (norma IEC 61131-7).
- `weaning_mamdani.fll`: modelo na linguagem nativa FuzzyLite Language (`fuzzylite 4.0b1401`) (12 regras).
- `evidencias/`: capturas reais verificadas da execução gráfica no **QtFuzzyLite 4.0** (`01_qtfuzzylite_setup.png` a `08_qtfuzzylite_caso3_te3_duvidoso.png`), incluindo a aba `Setup` (`Good Rules: 12. Bad Rules: 0.`), os 4 mapas `Surface 2D` (`NIF×RR`, `VT×RSBI`, `NIF×RSBI`, `RR×RSBI`) e as simulações de fuzzificação, regras ativadas e defuzzificação na aba `Control` para `TE5` (`72,732`), `TR10` (`25,750`) e `TE3` (`40,048`).

## Configuração

- Método: Mamdani.
- Entradas: `NIF` (−60 a 0 cm.H₂O), `RR` (5 a 60 resp/min), `VT` (100 a 1000 mL), `RSBI` (0 a 300 resp/min/L).
- Saída: `Resultado` (0 a 100), com dois valores linguísticos possíveis: `Fracasso` (`[0, 0, 35, 65]`) e `Sucesso` (`[35, 65, 100, 100]`).
- Operadores: E = mínimo (`min`); implicação = mínimo (`min`); agregação = máximo (`max`); defuzzificação = centroide (`Centroid`).
- 3 termos por entrada; 2 termos na saída (`Fracasso` e `Sucesso`); **12 regras compactas** (4 `Sucesso`, 8 `Fracasso`) que cobrem 100% do domínio de entrada ($3^4 = 81$ combinações).

## Decisão

- `Resultado ≥ 60` (política conservadora; ou `≥ 50` no ponto médio): `Sucesso` (liberar).
- `40 ≤ Resultado < 60`: zona de cautela (decisão operacional conservadora: manter suporte / reavaliar).
- `Resultado < 40`: `Fracasso` (manter suporte).

## Execução

Abrir `weaning_mamdani.fll` no QtFuzzyLite 4.0. A aba `Setup` valida as 12 regras, a aba `Control` simula os casos e `Tools > Surface 2D` gera os mapas.
