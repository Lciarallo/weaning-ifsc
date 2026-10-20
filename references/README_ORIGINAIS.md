# Arquivos originais recebidos

Todos os arquivos enviados pelo usuário foram copiados **byte a byte** para `references/originais/`.
Os hashes SHA-256 foram conferidos contra os arquivos anexados na sessão em 20/10/2026.

| Arquivo | Bytes | SHA-256 |
|---|---:|---|
| `Base de Conhecimento Weaning.pdf` | 904,896 | `1a52b7cf3c04766cbdb5f1dffce95106a79123050fa03627aea66b57220abbe4` |
| `CTRL_Respira.txt` | 58 | `0e0304dc83d3e5d030ef0a1f2f0fae048ac5f558265cc24edf137421b488c136` |
| `fuzzylite-4.0.zip` | 8,096,814 | `12f12f18befaef19d17f8c9a14f2d2c3c7418b9df5323eb87eda8e5188510d56` |
| `Process Cases.jpg` | 98,289 | `db87a35b803cd7b6a0c0902e09f4c9901bee3c3cbb971a43636d2ae516b221fc` |
| `Respira_DB_Teste_Process_Cases.txt` | 174 | `987948f4d9ffd64fde69674a609abc9ffece3696ebce78b12caf12271e05dbbe` |
| `Respira_DB_Teste.txt` | 259 | `59a30f4cc04e902e109783312b2a3b2999e7c5e94730531b2282e67fccd4952d` |
| `Respira_DB_Treinamento_Process_Cases.txt` | 396 | `d9598631e29fa2e63965bf7426b03066737f05e84c81cad7afbeb3ad43b6ec95` |
| `Respira_DB_Treinamento.txt` | 586 | `ac8f41bba83ed3eefcdf479bc1c24baf7835727fe7f78eec4fc1757eeaf5daf7` |
| `Trabalho - Bayes e Fuzzy 2026.pdf` | 254,181 | `9b3abb981db752f938bba315712c190213618496e0848319479568d90145bb0f` |

## Observações

- `Process Cases.jpg` é a captura de tela do Netica fornecida como referência.
- `fuzzylite-4.0.zip` é a biblioteca externa anexada; não faz parte do código do projeto.
- `Trabalho - Bayes e Fuzzy 2026.pdf` é o enunciado original; a cópia renomeada `references/Trabalho_Bayes_Fuzzy_2026.pdf` é apenas uma conveniência de leitura.
- A base de conhecimento aparece também como `.pptx` em `references/`, mas o **PDF enviado** está preservado em `references/originais/`.
- Nenhum desses arquivos foi alterado; a auditoria hash a hash está em `references/MANIFEST_ORIGINAIS.csv`.

## Verificação

```bash
cd references/originais
sha256sum *
```

Os valores devem coincidir com a tabela acima.
