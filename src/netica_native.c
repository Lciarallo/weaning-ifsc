/* Execução real do Netica C API, sem chave: rede de cinco nós em modo limitado.
 * Discretização: produzida previamente pelo Netica Application, nunca neste C.
 * Aprendizagem: contagem com um pseudocaso por estado (Laplace alpha=1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Netica.h"

static environ_ns *env;
static const char *vars[] = {"NIF", "RR", "VT", "RSBI"};

static void check(void) {
    report_ns *error = GetError_ns(env, ERROR_ERR, NULL);
    if (error) {
        fprintf(stderr, "Netica: %s\n", ErrorMessage_ns(error));
        exit(1);
    }
}

static net_bn *read_net(const char *path) {
    stream_ns *stream = NewFileStream_ns(path, env, NULL);
    net_bn *net = ReadNet_bn(stream, NO_WINDOW);
    check();
    DeleteStream_ns(stream);
    return net;
}

static void train(const char *discretized, const char *cases, const char *output) {
    net_bn *net = read_net(discretized);
    node_bn *target = NewNode_bn("Resultado", 2, net);
    SetNodeStateNames_bn(target, "Fracasso,Sucesso");
    SetNodeVisPosition_bn(target, NULL, 300, 500);
    prob_bn uniform[] = {0.5, 0.5};
    SetNodeProbs_bn(target, NULL, uniform);
    SetNodeExperience_bn(target, NULL, 2.0);
    for (int v = 0; v < 4; ++v) {
        node_bn *node = GetNodeNamed_bn(vars[v], net);
        if (GetNodeNumberStates_bn(node) != 2) {
            fprintf(stderr, "Esperados dois estados em %s\n", vars[v]);
            exit(1);
        }
        AddLink_bn(target, node);
        for (state_bn cls = 0; cls < 2; ++cls) {
            SetNodeProbs_bn(node, &cls, uniform);
            SetNodeExperience_bn(node, &cls, 2.0);
        }
    }
    check();
    stream_ns *stream = NewFileStream_ns(cases, env, NULL);
    ReviseCPTsByCaseFile_bn(stream, GetNetNodes_bn(net), 0, 1.0);
    check();
    DeleteStream_ns(stream);
    CompileNet_bn(net);
    check();
    stream = NewFileStream_ns(output, env, NULL);
    WriteNet_bn(net, stream);
    check();
    DeleteStream_ns(stream);
    printf("Rede aprendida e salva pelo Netica: %s\n", output);
    DeleteNet_bn(net);
}

static void predict(const char *model, const char *cases, const char *output) {
    net_bn *net = read_net(model);
    CompileNet_bn(net);
    check();
    node_bn *target = GetNodeNamed_bn("Resultado", net);
    FILE *input = fopen(cases, "r"), *out = fopen(output, "w");
    if (!input || !out) { perror("casos/saida"); exit(1); }
    char line[2048];
    if (!fgets(line, sizeof(line), input)) { fputs("Casos vazios\n", stderr); exit(1); }
    fputs("IDnum,p_fracasso,p_sucesso\n", out);
    int count = 0;
    while (fgets(line, sizeof(line), input)) {
        char *save = NULL;
        char *token = strtok_r(line, ",\t\r\n", &save);
        if (!token) continue;
        int id = atoi(token);
        RetractNetFindings_bn(net);
        for (int v = 0; v < 4; ++v) {
            token = strtok_r(NULL, ",\t\r\n", &save);
            if (!token) { fputs("Caso incompleto\n", stderr); exit(1); }
            node_bn *node = GetNodeNamed_bn(vars[v], net);
            if (GetNodeType_bn(node) == CONTINUOUS_TYPE) {
                EnterNodeValue_bn(node, strtod(token, NULL));
            } else {
                state_bn state = GetStateNamed_bn(token, node);
                EnterFinding_bn(node, state);
            }
            check();
        }
        /* Desfecho, se presente no arquivo, deliberadamente não é lido. */
        const prob_bn *p = GetNodeBeliefs_bn(target);
        check();
        fprintf(out, "%d,%.12g,%.12g\n", id, p[0], p[1]);
        ++count;
    }
    fclose(input);
    fclose(out);
    printf("Inferencia cega: %d casos, %s\n", count, output);
    DeleteNet_bn(net);
}

int main(int argc, char **argv) {
    if (argc != 5 || (strcmp(argv[1], "--train") && strcmp(argv[1], "--predict"))) {
        fputs("Uso: netica_native --train discretizacao.dne treino.cas rede.dne\n"
              "     netica_native --predict rede.dne casos.csv probabilidades.csv\n", stderr);
        return 2;
    }
    char message[MESG_LEN_ns];
    env = NewNeticaEnviron_ns(NULL, NULL, NULL);
    if (InitNetica2_bn(env, message) < 0) { fprintf(stderr, "%s\n", message); return 1; }
    const char *version = NULL;
    GetNeticaVersion_bn(env, &version);
    printf("Netica C API %s; modo limitado sem chave\n", version);
    if (!strcmp(argv[1], "--train")) train(argv[2], argv[3], argv[4]);
    else predict(argv[2], argv[3], argv[4]);
    CloseNetica_bn(env, message);
    return 0;
}
