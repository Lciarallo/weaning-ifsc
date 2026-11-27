| Caso   |   Regra | NIF           | RR           | VT          | RSBI              |   mu_NIF |   mu_RR |   mu_VT |   mu_RSBI |   alpha | Consequente   |   Score |
|:-------|--------:|:--------------|:-------------|:------------|:------------------|---------:|--------:|--------:|----------:|--------:|:--------------|--------:|
| TE5    |       1 | NIF_Forte     | RR_Adequada  | Qualquer    | RSBI_Favoravel    |    1.000 |   0.750 |   1.000 |     1.000 |   0.750 | Sucesso       |       7 |
| TE5    |       2 | NIF_Forte     | RR_Limitrofe | Qualquer    | RSBI_Favoravel    |    1.000 |   0.250 |   1.000 |     1.000 |   0.250 | Sucesso       |       5 |
| TR10   |       5 | NIF_Fraco     | Qualquer     | Qualquer    | Qualquer          |    0.500 |   1.000 |   1.000 |     1.000 |   0.500 | Fracasso      |      -2 |
| TR10   |       6 | Qualquer      | RR_Elevada   | Qualquer    | Qualquer          |    1.000 |   0.375 |   1.000 |     1.000 |   0.375 | Fracasso      |      -2 |
| TR10   |       7 | Qualquer      | Qualquer     | Qualquer    | RSBI_Desfavoravel |    1.000 |   1.000 |   1.000 |     1.000 |   1.000 | Fracasso      |      -2 |
| TR10   |       8 | NIF_Limitrofe | Qualquer     | VT_Baixo    | Qualquer          |    0.500 |   1.000 |   1.000 |     1.000 |   0.500 | Fracasso      |      -1 |
| TR10   |      10 | NIF_Limitrofe | RR_Limitrofe | Qualquer    | Qualquer          |    0.500 |   0.625 |   1.000 |     1.000 |   0.500 | Fracasso      |       0 |
| TE3    |       1 | NIF_Forte     | RR_Adequada  | Qualquer    | RSBI_Favoravel    |    0.300 |   0.750 |   1.000 |     1.000 |   0.300 | Sucesso       |       7 |
| TE3    |       2 | NIF_Forte     | RR_Limitrofe | Qualquer    | RSBI_Favoravel    |    0.300 |   0.250 |   1.000 |     1.000 |   0.250 | Sucesso       |       5 |
| TE3    |       4 | NIF_Limitrofe | RR_Adequada  | VT_Adequado | RSBI_Favoravel    |    0.700 |   0.750 |   0.057 |     1.000 |   0.057 | Sucesso       |       5 |
| TE3    |       8 | NIF_Limitrofe | Qualquer     | VT_Baixo    | Qualquer          |    0.700 |   1.000 |   0.900 |     1.000 |   0.700 | Fracasso      |      -1 |
| TE3    |      10 | NIF_Limitrofe | RR_Limitrofe | Qualquer    | Qualquer          |    0.700 |   0.250 |   1.000 |     1.000 |   0.250 | Fracasso      |       0 |