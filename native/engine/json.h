/* A minimal JSON reader for config.json and safetensors headers. */
#ifndef QD_JSON_H
#define QD_JSON_H

#include <stddef.h>

enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };

typedef struct jv {
    int type;
    double num;        /* J_NUM, J_BOOL (0 / 1) */
    char *str;         /* J_STR */
    int n;             /* J_ARR / J_OBJ: items */
    struct jv *items;  /* J_ARR / J_OBJ */
    char **keys;       /* J_OBJ */
} jv;

jv *json_parse(const char *s, size_t len); /* NULL on a syntax error */
void json_free(jv *v);
jv *json_get(const jv *obj, const char *key); /* NULL if absent or obj is not an object */
double json_num(const jv *obj, const char *key, double dflt);

#endif
