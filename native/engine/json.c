/* A minimal recursive-descent JSON reader (objects, arrays, strings with escapes, numbers,
   true / false / null): enough for config.json and safetensors headers. */
#include "json.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    int err;
    int depth; /* open arrays and objects */
} rd;

static void ws(rd *r) {
    while (r->p < r->end && (*r->p == ' ' || *r->p == '\n' || *r->p == '\r' || *r->p == '\t'))
        r->p++;
}

static int value(rd *r, jv *out);

static char *string(rd *r) {
    if (r->p >= r->end || *r->p != '"') {
        r->err = 1;
        return NULL;
    }
    r->p++;
    size_t cap = 16, n = 0;
    char *s = malloc(cap);
    while (r->p < r->end && *r->p != '"') {
        char c = *r->p++;
        if (c == '\\' && r->p < r->end) {
            char e = *r->p++;
            switch (e) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': /* (names here are ASCII; keep the escape's low byte) */
                if (r->end - r->p < 4) { r->err = 1; free(s); return NULL; }
                c = (char)strtol((char[]){r->p[2], r->p[3], 0}, NULL, 16);
                r->p += 4;
                break;
            default: c = e;
            }
        }
        if (n + 2 > cap) s = realloc(s, cap *= 2);
        s[n++] = c;
    }
    if (r->p >= r->end) {
        r->err = 1;
        free(s);
        return NULL;
    }
    r->p++;
    s[n] = 0;
    return s;
}

static int items(rd *r, jv *out, char close, int obj) {
    int cap = 4;
    out->items = calloc(cap, sizeof(jv));
    if (obj) out->keys = calloc(cap, sizeof(char *));
    r->p++;
    ws(r);
    if (r->p < r->end && *r->p == close) {
        r->p++;
        return 0;
    }
    for (;;) {
        if (out->n == cap) {
            cap *= 2;
            out->items = realloc(out->items, cap * sizeof(jv));
            if (obj) out->keys = realloc(out->keys, cap * sizeof(char *));
        }
        memset(&out->items[out->n], 0, sizeof(jv));
        if (obj) {
            ws(r);
            out->keys[out->n] = string(r);
            if (r->err) return -1;
            ws(r);
            if (r->p >= r->end || *r->p != ':') return r->err = 1, -1;
            r->p++;
        }
        if (value(r, &out->items[out->n])) return -1;
        out->n++;
        ws(r);
        if (r->p < r->end && *r->p == ',') {
            r->p++;
            continue;
        }
        if (r->p < r->end && *r->p == close) {
            r->p++;
            return 0;
        }
        return r->err = 1, -1;
    }
}

static int value(rd *r, jv *out) {
    ws(r);
    if (r->p >= r->end) return r->err = 1, -1;
    char c = *r->p;
    if (c == '{' || c == '[') {
        /* (the parser recurses per level: deeper than any config or header is refused, never
           a stack overflow) */
        if (r->depth == 64) return r->err = 1, -1;
        r->depth++;
        out->type = c == '{' ? J_OBJ : J_ARR;
        int rc = items(r, out, c == '{' ? '}' : ']', c == '{');
        r->depth--;
        return rc;
    }
    if (c == '"') {
        out->type = J_STR;
        out->str = string(r);
        return r->err ? -1 : 0;
    }
    if (!strncmp(r->p, "true", 4) || !strncmp(r->p, "false", 5)) {
        out->type = J_BOOL;
        out->num = c == 't';
        r->p += c == 't' ? 4 : 5;
        return 0;
    }
    if (!strncmp(r->p, "null", 4)) {
        out->type = J_NULL;
        r->p += 4;
        return 0;
    }
    char *e;
    out->type = J_NUM;
    out->num = strtod(r->p, &e);
    if (e == r->p) return r->err = 1, -1;
    r->p = e;
    return 0;
}

static void clear(jv *v) {
    if (v->type == J_STR) free(v->str);
    if (v->type == J_ARR || v->type == J_OBJ) {
        for (int i = 0; i < v->n; i++) {
            clear(&v->items[i]);
            if (v->keys) free(v->keys[i]);
        }
        free(v->items);
        free(v->keys);
    }
}

jv *json_parse(const char *s, size_t len) {
    /* (s[len] must be NUL: a malformed text's last literal or number is read up to it) */
    rd r = {s, s + len, 0, 0};
    jv *v = calloc(1, sizeof(jv));
    if (value(&r, v) || r.err) {
        clear(v);
        free(v);
        return NULL;
    }
    return v;
}

void json_free(jv *v) {
    if (!v) return;
    clear(v);
    free(v);
}

jv *json_get(const jv *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++)
        if (!strcmp(obj->keys[i], key)) return &obj->items[i];
    return NULL;
}

double json_num(const jv *obj, const char *key, double dflt) {
    jv *v = json_get(obj, key);
    return v && (v->type == J_NUM || v->type == J_BOOL) ? v->num : dflt;
}
