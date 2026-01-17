/*
 * Phonex-C1 (Hardened): Pure C Transformer
 * ----------------------------------------
 * Compile: gcc -O3 c1p_harden.c -lm -o c1p
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

// --- Hyperparameters ---
#define MAX_LAYERS 4
#define D_MODEL 64
#define N_HEADS 4
#define HEAD_DIM (D_MODEL / N_HEADS)
#define D_FF (D_MODEL * 4)
#define SEQ_LEN 64
#define VOCAB_SIZE 128
#define BATCH_SIZE 1

// Optimization
#define LR 0.001f
#define WEIGHT_DECAY 0.01f
#define BETA1 0.9f
#define BETA2 0.99f
#define EPSILON 1e-8f
#define TRAIN_STEPS 5000

// Sampling
#define TEMP 0.8f
#define TOP_P 0.9f

// --- Data Structures ---

typedef struct {
    float* data;
    float* grad;
    float* m; // Adam First Moment
    float* v; // Adam Second Moment
    int n, d; // Shape [n, d]
    char name[32];
} Tensor;

typedef struct {
    Tensor ln1_g, ln1_b;
    Tensor w_q, w_k, w_v;
    Tensor w_o;          
    Tensor ln2_g, ln2_b;
    Tensor w_ff1, w_ff2; 
} Block;

typedef struct {
    Tensor token_emb;
    // RoPE Precomputed Tables
    float* rope_cos; // [SEQ_LEN, HEAD_DIM/2]
    float* rope_sin; // [SEQ_LEN, HEAD_DIM/2]
    
    Block layers[MAX_LAYERS];
    Tensor ln_f_g, ln_f_b;
    Tensor w_head;
    int n_layers;
} GPT;

typedef struct {
    Tensor q, k, v;             // [SEQ, D_MODEL]
    Tensor att_scores;          // [N_HEADS, SEQ, SEQ]
    Tensor att_probs;           // [N_HEADS, SEQ, SEQ]
    Tensor att_out;             // [SEQ, D_MODEL]
    Tensor att_proj;            // [SEQ, D_MODEL]
    
    Tensor ln1_out, ln1_mean, ln1_var;
    Tensor ln2_out, ln2_mean, ln2_var;
    
    Tensor ffn_in;              // [SEQ, D_FF]
    Tensor ffn_act;             // [SEQ, D_FF]
    Tensor ffn_out;             // [SEQ, D_MODEL]
    
    Tensor res1, res2;          // Residuals
} LayerCache;

typedef struct {
    Tensor emb_out;
    LayerCache layers[MAX_LAYERS];
    Tensor ln_f_out, ln_f_mean, ln_f_var;
    Tensor logits;
    Tensor probs;               
    int* inputs;
} GPTCache;

// --- Helper Functions ---

Tensor tensor_create(int n, int d, const char* name) {
    Tensor t;
    t.n = n; t.d = d;
    snprintf(t.name, 32, "%s", name);
    // Calloc ensures zero initialization (critical for accumulators)
    t.data = (float*)calloc(n * d, sizeof(float));
    t.grad = (float*)calloc(n * d, sizeof(float));
    t.m = (float*)calloc(n * d, sizeof(float));
    t.v = (float*)calloc(n * d, sizeof(float));
    if(!t.data || !t.grad || !t.m || !t.v) { 
        fprintf(stderr, "OOM: %s\n", name); exit(1); 
    }
    return t;
}

void tensor_zero_grad(Tensor* t) {
    memset(t->grad, 0, t->n * t->d * sizeof(float));
}

void tensor_init_xavier(Tensor* t) {
    float scale = sqrtf(6.0f / (float)(t->n + t->d));
    for(int i=0; i < t->n * t->d; i++) {
        t->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// C = A * B. accumulate=1 adds to existing C data
void matmul(Tensor* A, Tensor* B, Tensor* C, int accumulate) {
    // Audit: Shape Safety
    assert(A->d == B->n && "MatMul Shape Mismatch: Inner dim");
    assert(C->n == A->n && "MatMul Shape Mismatch: Output rows");
    assert(C->d == B->d && "MatMul Shape Mismatch: Output cols");

    for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < B->d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->d; k++) {
                sum += A->data[i * A->d + k] * B->data[k * B->d + j];
            }
            if(accumulate) C->data[i * C->d + j] += sum;
            else C->data[i * C->d + j] = sum;
        }
    }
}

// --- Math Kernels ---

float gelu(float x) {
    const float s = 0.7978845608f; 
    return 0.5f * x * (1.0f + tanhf(s * (x + 0.044715f * x * x * x)));
}

float dgelu(float x) {
    const float s = 0.7978845608f;
    float x3 = x * x * x;
    float inner = s * (x + 0.044715f * x3);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * s * (1.0f + 0.134145f * x * x);
}

// --- RoPE (Precomputed & Stable) ---

void precompute_rope(GPT* m) {
    m->rope_cos = (float*)malloc(SEQ_LEN * (HEAD_DIM/2) * sizeof(float));
    m->rope_sin = (float*)malloc(SEQ_LEN * (HEAD_DIM/2) * sizeof(float));
    
    for(int t=0; t<SEQ_LEN; t++) {
        for(int i=0; i<HEAD_DIM/2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / HEAD_DIM);
            float theta = t * freq;
            m->rope_cos[t*(HEAD_DIM/2) + i] = cosf(theta);
            m->rope_sin[t*(HEAD_DIM/2) + i] = sinf(theta);
        }
    }
}

void apply_rope_forward(Tensor* Q, Tensor* K, GPT* m, int t) {
    int row_offset = t * D_MODEL;
    int half_head = HEAD_DIM / 2;
    
    for (int h = 0; h < N_HEADS; h++) {
        int head_offset = h * HEAD_DIM;
        for (int i = 0; i < half_head; i++) {
            float cos_t = m->rope_cos[t*half_head + i];
            float sin_t = m->rope_sin[t*half_head + i];

            int idx1 = row_offset + head_offset + 2*i;
            int idx2 = row_offset + head_offset + 2*i + 1;

            float q1 = Q->data[idx1]; float q2 = Q->data[idx2];
            Q->data[idx1] = q1 * cos_t - q2 * sin_t;
            Q->data[idx2] = q1 * sin_t + q2 * cos_t;
            
            float k1 = K->data[idx1]; float k2 = K->data[idx2];
            K->data[idx1] = k1 * cos_t - k2 * sin_t;
            K->data[idx2] = k1 * sin_t + k2 * cos_t;
        }
    }
}

void apply_rope_backward(Tensor* Q, Tensor* K, GPT* m, int t) {
    int row_offset = t * D_MODEL;
    int half_head = HEAD_DIM / 2;
    
    for (int h = 0; h < N_HEADS; h++) {
        int head_offset = h * HEAD_DIM;
        for (int i = 0; i < half_head; i++) {
            float cos_t = m->rope_cos[t*half_head + i];
            float sin_t = m->rope_sin[t*half_head + i];

            int idx1 = row_offset + head_offset + 2*i;
            int idx2 = row_offset + head_offset + 2*i + 1;

            // Gradient rotation by -theta
            float dq1 = Q->grad[idx1]; float dq2 = Q->grad[idx2];
            Q->grad[idx1] = dq1 * cos_t + dq2 * sin_t;
            Q->grad[idx2] = -dq1 * sin_t + dq2 * cos_t;

            float dk1 = K->grad[idx1]; float dk2 = K->grad[idx2];
            K->grad[idx1] = dk1 * cos_t + dk2 * sin_t;
            K->grad[idx2] = -dk1 * sin_t + dk2 * cos_t;
        }
    }
}

// --- Layer Norm (Exact) ---

void layer_norm_forward(Tensor* x, Tensor* g, Tensor* b, Tensor* out, Tensor* mean, Tensor* var) {
    for (int i = 0; i < x->n; i++) {
        float m = 0, v = 0;
        for(int j=0; j<x->d; j++) m += x->data[i*x->d + j];
        m /= x->d;
        mean->data[i] = m;
        
        for(int j=0; j<x->d; j++) {
            float d = x->data[i*x->d + j] - m;
            v += d*d;
        }
        v /= x->d;
        var->data[i] = v;
        
        float inv_std = 1.0f / sqrtf(v + EPSILON);
        for(int j=0; j<x->d; j++) {
            float x_hat = (x->data[i*x->d + j] - m) * inv_std;
            out->data[i*x->d + j] = x_hat * g->data[j] + b->data[j];
        }
    }
}

void backward_layer_norm(Tensor* x, Tensor* g, Tensor* b, Tensor* y, Tensor* mean, Tensor* var) {
    int D = x->d;
    for (int i=0; i<x->n; i++) {
        float inv_std = 1.0f / sqrtf(var->data[i] + EPSILON);
        float m = mean->data[i];
        
        float sum_dy_xhat = 0.0f;
        float sum_dy = 0.0f;
        int offset = i*D;
        
        // 1. Accumulate statistics
        for(int j=0; j<D; j++) {
            float dy = y->grad[offset + j];
            float x_hat = (x->data[offset + j] - m) * inv_std;
            
            g->grad[j] += dy * x_hat;
            b->grad[j] += dy;
            
            float dx_norm = dy * g->data[j]; 
            sum_dy += dx_norm;
            sum_dy_xhat += dx_norm * x_hat;
        }
        
        // 2. Distribute gradient
        for(int j=0; j<D; j++) {
            float dy = y->grad[offset + j];
            float dx_norm = dy * g->data[j];
            float x_hat = (x->data[offset + j] - m) * inv_std;
            
            float term = (D * dx_norm) - sum_dy - (x_hat * sum_dy_xhat);
            x->grad[offset + j] += (1.0f / D) * inv_std * term;
        }
    }
}

// --- Linear Backward ---

void backward_linear(Tensor* x, Tensor* w, Tensor* b, Tensor* y) {
    // Audit: Shape Safety
    assert(x->d == w->n && "Backward Linear Shape Mismatch");
    
    // dW = x^T * dy
    for(int i=0; i<w->n; i++) {
        for(int j=0; j<w->d; j++) {
            float sum = 0;
            for(int k=0; k<x->n; k++) sum += x->data[k*x->d + i] * y->grad[k*y->d + j];
            w->grad[i*w->d + j] += sum;
        }
    }
    // dx = dy * W^T
    for(int i=0; i<x->n; i++) {
        for(int j=0; j<x->d; j++) {
            float sum = 0;
            for(int k=0; k<w->d; k++) sum += y->grad[i*y->d + k] * w->data[j*w->d + k];
            x->grad[i*x->d + j] += sum;
        }
    }
    // db
    if(b) {
        for(int i=0; i<y->n; i++) {
            for(int j=0; j<y->d; j++) b->grad[j] += y->grad[i*y->d + j];
        }
    }
}

// --- Attention ---

void attention_forward(LayerCache* c, GPT* m, int len) {
    // RoPE
    for(int t=0; t<len; t++) apply_rope_forward(&c->q, &c->k, m, t);
    
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    for(int h=0; h<N_HEADS; h++) {
        for(int t=0; t<len; t++) {
            // Score
            for(int k=0; k<len; k++) {
                float score = -1e9f; // Mask
                if (k <= t) {
                    score = 0.0f;
                    for(int d=0; d<HEAD_DIM; d++) {
                        int q_idx = t*D_MODEL + h*HEAD_DIM + d;
                        int k_idx = k*D_MODEL + h*HEAD_DIM + d;
                        score += c->q.data[q_idx] * c->k.data[k_idx];
                    }
                    score *= scale;
                }
                c->att_scores.data[h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN + k] = score;
            }
            
            // Softmax
            float max_val = -1e9f;
            int row_off = h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN;
            for(int k=0; k<len; k++) {
                if(k<=t && c->att_scores.data[row_off+k] > max_val) 
                    max_val = c->att_scores.data[row_off+k];
            }
            
            float sum = 0.0f;
            for(int k=0; k<len; k++) {
                float e = (k<=t) ? expf(c->att_scores.data[row_off+k] - max_val) : 0.0f;
                c->att_probs.data[row_off+k] = e;
                sum += e;
            }
            for(int k=0; k<len; k++) c->att_probs.data[row_off+k] /= sum;
        }
        
        // Weighted Sum
        for(int t=0; t<len; t++) {
            int row_off = h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN;
            for(int d=0; d<HEAD_DIM; d++) {
                float val = 0.0f;
                for(int k=0; k<len; k++) {
                    if (k > t) continue;
                    val += c->att_probs.data[row_off + k] * c->v.data[k*D_MODEL + h*HEAD_DIM + d];
                }
                c->att_out.data[t*D_MODEL + h*HEAD_DIM + d] = val;
            }
        }
    }
}

void attention_backward(LayerCache* c, GPT* m, int len) {
    for(int h=0; h<N_HEADS; h++) {
        // 1. Weighted Sum Backward
        for(int t=0; t<len; t++) {
            int row_off = h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN;
            for(int d=0; d<HEAD_DIM; d++) {
                float d_out = c->att_out.grad[t*D_MODEL + h*HEAD_DIM + d];
                for(int k=0; k<len; k++) {
                    if (k > t) continue;
                    int v_idx = k*D_MODEL + h*HEAD_DIM + d;
                    // dV
                    c->v.grad[v_idx] += d_out * c->att_probs.data[row_off+k];
                    // dProb
                    c->att_probs.grad[row_off+k] += d_out * c->v.data[v_idx];
                }
            }
        }
        
        // 2. Softmax Backward
        for(int t=0; t<len; t++) {
            int row_off = h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN;
            float sum_p_dp = 0;
            for(int k=0; k<len; k++) sum_p_dp += c->att_probs.data[row_off+k] * c->att_probs.grad[row_off+k];
            for(int k=0; k<len; k++) {
                if (k > t) continue;
                float p = c->att_probs.data[row_off+k];
                float dp = c->att_probs.grad[row_off+k];
                c->att_scores.grad[row_off+k] += p * (dp - sum_p_dp);
            }
        }
        
        // 3. Score Backward
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for(int t=0; t<len; t++) {
            for(int k=0; k<len; k++) {
                if (k > t) continue;
                float dscore = c->att_scores.grad[h*(SEQ_LEN*SEQ_LEN) + t*SEQ_LEN + k];
                dscore *= scale;
                for(int d=0; d<HEAD_DIM; d++) {
                    int q_idx = t*D_MODEL + h*HEAD_DIM + d;
                    int k_idx = k*D_MODEL + h*HEAD_DIM + d;
                    c->q.grad[q_idx] += dscore * c->k.data[k_idx];
                    c->k.grad[k_idx] += dscore * c->q.data[q_idx];
                }
            }
        }
    }
    
    // 4. RoPE Backward
    for(int t=0; t<len; t++) apply_rope_backward(&c->q, &c->k, m, t);
}

// --- Main Flow ---

void forward(GPT* m, GPTCache* c, int* inputs, int len) {
    c->emb_out.n = len;
    for(int i=0; i<len; i++) {
        int tid = inputs[i];
        // Audit: Input Bounds
        if(tid < 0 || tid >= VOCAB_SIZE) tid = 0; 
        memcpy(&c->emb_out.data[i*D_MODEL], &m->token_emb.data[tid*D_MODEL], D_MODEL * sizeof(float));
    }
    
    Tensor* x = &c->emb_out;
    
    for(int i=0; i<m->n_layers; i++) {
        LayerCache* lc = &c->layers[i];
        Block* b = &m->layers[i];
        
        // Setup dims
        lc->q.n=len; lc->k.n=len; lc->v.n=len; lc->att_out.n=len; lc->att_proj.n=len;
        lc->ln1_out.n=len; lc->res1.n=len;
        
        // Norm 1
        layer_norm_forward(x, &b->ln1_g, &b->ln1_b, &lc->ln1_out, &lc->ln1_mean, &lc->ln1_var);
        
        // QKV
        matmul(&lc->ln1_out, &b->w_q, &lc->q, 0);
        matmul(&lc->ln1_out, &b->w_k, &lc->k, 0);
        matmul(&lc->ln1_out, &b->w_v, &lc->v, 0);
        
        // Attention
        attention_forward(lc, m, len);
        
        // Projection
        matmul(&lc->att_out, &b->w_o, &lc->att_proj, 0);
        
        // Residual 1
        for(int j=0; j<len*D_MODEL; j++) lc->res1.data[j] = x->data[j] + lc->att_proj.data[j];
        
        // FFN
        layer_norm_forward(&lc->res1, &b->ln2_g, &b->ln2_b, &lc->ln2_out, &lc->ln2_mean, &lc->ln2_var);
        matmul(&lc->ln2_out, &b->w_ff1, &lc->ffn_in, 0);
        for(int j=0; j<len*D_FF; j++) lc->ffn_act.data[j] = gelu(lc->ffn_in.data[j]);
        matmul(&lc->ffn_act, &b->w_ff2, &lc->ffn_out, 0);
        
        // Residual 2
        for(int j=0; j<len*D_MODEL; j++) lc->res2.data[j] = lc->res1.data[j] + lc->ffn_out.data[j];
        
        x = &lc->res2;
    }
    
    // Final Head
    layer_norm_forward(x, &m->ln_f_g, &m->ln_f_b, &c->ln_f_out, &c->ln_f_mean, &c->ln_f_var);
    matmul(&c->ln_f_out, &m->w_head, &c->logits, 0);
}

void backward(GPT* m, GPTCache* c, int* inputs, int len) {
    // Loss gradients are in c->logits.grad
    
    // Final Head
    backward_linear(&c->ln_f_out, &m->w_head, NULL, &c->logits);
    backward_layer_norm(&c->layers[m->n_layers-1].res2, &m->ln_f_g, &m->ln_f_b, &c->ln_f_out, &c->ln_f_mean, &c->ln_f_var);
    
    for(int i=m->n_layers-1; i>=0; i--) {
        LayerCache* lc = &c->layers[i];
        Block* b = &m->layers[i];
        
        // --- FFN Block ---
        // dRes2 flows to dRes1 and dFFN_Out
        for(int j=0; j<len*D_MODEL; j++) {
            lc->res1.grad[j] += lc->res2.grad[j];    
            lc->ffn_out.grad[j] += lc->res2.grad[j]; 
        }
        
        backward_linear(&lc->ffn_act, &b->w_ff2, NULL, &lc->ffn_out);
        for(int j=0; j<len*D_FF; j++) lc->ffn_in.grad[j] += lc->ffn_act.grad[j] * dgelu(lc->ffn_in.data[j]);
        backward_linear(&lc->ln2_out, &b->w_ff1, NULL, &lc->ffn_in);
        backward_layer_norm(&lc->res1, &b->ln2_g, &b->ln2_b, &lc->ln2_out, &lc->ln2_mean, &lc->ln2_var);
        
        // --- Attention Block ---
        Tensor* prev_out = (i > 0) ? &c->layers[i-1].res2 : &c->emb_out;
        
        // dRes1 flows to Input and Att_Proj
        for(int j=0; j<len*D_MODEL; j++) {
            prev_out->grad[j] += lc->res1.grad[j];     
            lc->att_proj.grad[j] += lc->res1.grad[j];  
        }
        
        backward_linear(&lc->att_out, &b->w_o, NULL, &lc->att_proj);
        attention_backward(lc, m, len);
        
        backward_linear(&lc->ln1_out, &b->w_q, NULL, &lc->q);
        backward_linear(&lc->ln1_out, &b->w_k, NULL, &lc->k);
        backward_linear(&lc->ln1_out, &b->w_v, NULL, &lc->v);
        
        backward_layer_norm(prev_out, &b->ln1_g, &b->ln1_b, &lc->ln1_out, &lc->ln1_mean, &lc->ln1_var);
    }
    
    // Embeddings
    for(int i=0; i<len; i++) {
        int tid = inputs[i]; if(tid>=VOCAB_SIZE) tid=0;
        for(int j=0; j<D_MODEL; j++) {
            m->token_emb.grad[tid*D_MODEL + j] += c->emb_out.grad[i*D_MODEL + j];
        }
    }
}

// --- Optimization (Correct AdamW) ---

void adamw_step(Tensor* t, int step, int decay_allowed) {
    float bias_corr1 = 1.0f - powf(BETA1, step+1);
    float bias_corr2 = 1.0f - powf(BETA2, step+1);
    
    for(int i=0; i < t->n * t->d; i++) {
        float g = t->grad[i];
        
        // Moments
        t->m[i] = BETA1 * t->m[i] + (1.0f - BETA1) * g;
        t->v[i] = BETA2 * t->v[i] + (1.0f - BETA2) * g * g;
        
        float m_hat = t->m[i] / bias_corr1;
        float v_hat = t->v[i] / bias_corr2;
        float update = m_hat / (sqrtf(v_hat) + EPSILON);
        
        // Apply Weight Decay only if allowed (Weights yes, Biases no)
        if (decay_allowed) {
            t->data[i] -= LR * (update + WEIGHT_DECAY * t->data[i]);
        } else {
            t->data[i] -= LR * update;
        }
    }
    // Audit: Critical Fix for Infinite Gradient Accumulation
    memset(t->grad, 0, t->n * t->d * sizeof(float));
}

void optimizer_step(GPT* m, int step) {
    adamw_step(&m->token_emb, step, 1);
    for(int i=0; i<m->n_layers; i++) {
        // Biases and Norms = 0, Weights = 1
        adamw_step(&m->layers[i].ln1_g, step, 0); adamw_step(&m->layers[i].ln1_b, step, 0);
        adamw_step(&m->layers[i].w_q, step, 1); adamw_step(&m->layers[i].w_k, step, 1);
        adamw_step(&m->layers[i].w_v, step, 1); adamw_step(&m->layers[i].w_o, step, 1);
        adamw_step(&m->layers[i].ln2_g, step, 0); adamw_step(&m->layers[i].ln2_b, step, 0);
        adamw_step(&m->layers[i].w_ff1, step, 1); adamw_step(&m->layers[i].w_ff2, step, 1);
    }
    adamw_step(&m->ln_f_g, step, 0); adamw_step(&m->ln_f_b, step, 0);
    adamw_step(&m->w_head, step, 1);
}

// --- Efficient Top-P ---

typedef struct {
    float p;
    int id;
} TokenProb;

int compare_probs(const void* a, const void* b) {
    float pa = ((TokenProb*)a)->p;
    float pb = ((TokenProb*)b)->p;
    return (pa > pb) ? -1 : (pa < pb) ? 1 : 0; // Descending
}

int sample_top_p(float* probs, int vocab_size, float p_val) {
    TokenProb* tps = malloc(vocab_size * sizeof(TokenProb));
    for(int i=0; i<vocab_size; i++) { tps[i].p = probs[i]; tps[i].id = i; }
    
    qsort(tps, vocab_size, sizeof(TokenProb), compare_probs);
    
    float cum_prob = 0.0f;
    int cutoff = 0;
    for(int i=0; i<vocab_size; i++) {
        cum_prob += tps[i].p;
        cutoff = i;
        if(cum_prob >= p_val) break;
    }
    
    float r = (float)rand() / RAND_MAX * cum_prob;
    float cdf = 0.0f;
    int res = tps[0].id;
    for(int i=0; i<=cutoff; i++) {
        cdf += tps[i].p;
        if(r <= cdf) { res = tps[i].id; break; }
    }
    free(tps);
    return res;
}

// --- Setup ---

void init_gpt(GPT* m) {
    m->n_layers = MAX_LAYERS;
    m->token_emb = tensor_create(VOCAB_SIZE, D_MODEL, "TokEmb"); tensor_init_xavier(&m->token_emb);
    precompute_rope(m);
    
    for(int i=0; i<MAX_LAYERS; i++) {
        m->layers[i].ln1_g = tensor_create(D_MODEL, 1, "L1G"); for(int j=0; j<D_MODEL; j++) m->layers[i].ln1_g.data[j]=1;
        m->layers[i].ln1_b = tensor_create(D_MODEL, 1, "L1B");
        m->layers[i].w_q = tensor_create(D_MODEL, D_MODEL, "WQ"); tensor_init_xavier(&m->layers[i].w_q);
        m->layers[i].w_k = tensor_create(D_MODEL, D_MODEL, "WK"); tensor_init_xavier(&m->layers[i].w_k);
        m->layers[i].w_v = tensor_create(D_MODEL, D_MODEL, "WV"); tensor_init_xavier(&m->layers[i].w_v);
        m->layers[i].w_o = tensor_create(D_MODEL, D_MODEL, "WO"); tensor_init_xavier(&m->layers[i].w_o);
        m->layers[i].ln2_g = tensor_create(D_MODEL, 1, "L2G"); for(int j=0; j<D_MODEL; j++) m->layers[i].ln2_g.data[j]=1;
        m->layers[i].ln2_b = tensor_create(D_MODEL, 1, "L2B");
        m->layers[i].w_ff1 = tensor_create(D_MODEL, D_FF, "WFF1"); tensor_init_xavier(&m->layers[i].w_ff1);
        m->layers[i].w_ff2 = tensor_create(D_FF, D_MODEL, "WFF2"); tensor_init_xavier(&m->layers[i].w_ff2);
    }
    m->ln_f_g = tensor_create(D_MODEL, 1, "LFG"); for(int j=0; j<D_MODEL; j++) m->ln_f_g.data[j]=1;
    m->ln_f_b = tensor_create(D_MODEL, 1, "LFB");
    m->w_head = tensor_create(D_MODEL, VOCAB_SIZE, "HEAD"); tensor_init_xavier(&m->w_head);
}

GPTCache init_cache() {
    GPTCache c;
    c.emb_out = tensor_create(SEQ_LEN, D_MODEL, "EmbOut");
    for(int i=0; i<MAX_LAYERS; i++) {
        c.layers[i].q = tensor_create(SEQ_LEN, D_MODEL, "Q");
        c.layers[i].k = tensor_create(SEQ_LEN, D_MODEL, "K");
        c.layers[i].v = tensor_create(SEQ_LEN, D_MODEL, "V");
        c.layers[i].att_scores = tensor_create(N_HEADS * SEQ_LEN, SEQ_LEN, "AttS");
        c.layers[i].att_probs = tensor_create(N_HEADS * SEQ_LEN, SEQ_LEN, "AttP");
        c.layers[i].att_out = tensor_create(SEQ_LEN, D_MODEL, "AttO");
        c.layers[i].att_proj = tensor_create(SEQ_LEN, D_MODEL, "AttProj");
        c.layers[i].ln1_out = tensor_create(SEQ_LEN, D_MODEL, "L1O");
        c.layers[i].ln1_mean = tensor_create(SEQ_LEN, 1, "L1M");
        c.layers[i].ln1_var = tensor_create(SEQ_LEN, 1, "L1V");
        c.layers[i].ln2_out = tensor_create(SEQ_LEN, D_MODEL, "L2O");
        c.layers[i].ln2_mean = tensor_create(SEQ_LEN, 1, "L2M");
        c.layers[i].ln2_var = tensor_create(SEQ_LEN, 1, "L2V");
        c.layers[i].ffn_in = tensor_create(SEQ_LEN, D_FF, "FFIn");
        c.layers[i].ffn_act = tensor_create(SEQ_LEN, D_FF, "FFAct");
        c.layers[i].ffn_out = tensor_create(SEQ_LEN, D_MODEL, "FFOut");
        c.layers[i].res1 = tensor_create(SEQ_LEN, D_MODEL, "R1");
        c.layers[i].res2 = tensor_create(SEQ_LEN, D_MODEL, "R2");
    }
    c.ln_f_out = tensor_create(SEQ_LEN, D_MODEL, "FinalOut");
    c.ln_f_mean = tensor_create(SEQ_LEN, 1, "FM"); c.ln_f_var = tensor_create(SEQ_LEN, 1, "FV");
    c.logits = tensor_create(SEQ_LEN, VOCAB_SIZE, "Logits");
    c.probs = tensor_create(SEQ_LEN, VOCAB_SIZE, "Probs");
    c.inputs = malloc(SEQ_LEN * sizeof(int));
    return c;
}

void zero_cache_grads(GPTCache* c) {
    tensor_zero_grad(&c->emb_out);
    for(int i=0; i<MAX_LAYERS; i++) {
        tensor_zero_grad(&c->layers[i].q); tensor_zero_grad(&c->layers[i].k); tensor_zero_grad(&c->layers[i].v);
        tensor_zero_grad(&c->layers[i].att_scores); tensor_zero_grad(&c->layers[i].att_probs);
        tensor_zero_grad(&c->layers[i].att_out); tensor_zero_grad(&c->layers[i].att_proj);
        tensor_zero_grad(&c->layers[i].ln1_out);
        tensor_zero_grad(&c->layers[i].ln2_out);
        tensor_zero_grad(&c->layers[i].ffn_in); tensor_zero_grad(&c->layers[i].ffn_act);
        tensor_zero_grad(&c->layers[i].ffn_out);
        tensor_zero_grad(&c->layers[i].res1); tensor_zero_grad(&c->layers[i].res2);
    }
    tensor_zero_grad(&c->ln_f_out); tensor_zero_grad(&c->logits);
}

void save_tensor(FILE* f, Tensor* t) {
    fwrite(&t->n, sizeof(int), 1, f);
    fwrite(&t->d, sizeof(int), 1, f);
    fwrite(t->data, sizeof(float), t->n * t->d, f);
}

void load_tensor(FILE* f, Tensor* t) {
    int n, d;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    if(n != t->n || d != t->d) { printf("Shape Error\n"); exit(1); }
    fread(t->data, sizeof(float), n * d, f);
}

void save_model(GPT* m, const char* path) {
    FILE* f = fopen(path, "wb");
    if(!f) return;
    save_tensor(f, &m->token_emb);
    for(int i=0; i<m->n_layers; i++) {
        save_tensor(f, &m->layers[i].ln1_g); save_tensor(f, &m->layers[i].ln1_b);
        save_tensor(f, &m->layers[i].w_q); save_tensor(f, &m->layers[i].w_k);
        save_tensor(f, &m->layers[i].w_v); save_tensor(f, &m->layers[i].w_o);
        save_tensor(f, &m->layers[i].ln2_g); save_tensor(f, &m->layers[i].ln2_b);
        save_tensor(f, &m->layers[i].w_ff1); save_tensor(f, &m->layers[i].w_ff2);
    }
    save_tensor(f, &m->ln_f_g); save_tensor(f, &m->ln_f_b);
    save_tensor(f, &m->w_head);
    fclose(f);
}

void load_model(GPT* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if(!f) return;
    load_tensor(f, &m->token_emb);
    for(int i=0; i<m->n_layers; i++) {
        load_tensor(f, &m->layers[i].ln1_g); load_tensor(f, &m->layers[i].ln1_b);
        load_tensor(f, &m->layers[i].w_q); load_tensor(f, &m->layers[i].w_k);
        load_tensor(f, &m->layers[i].w_v); load_tensor(f, &m->layers[i].w_o);
        load_tensor(f, &m->layers[i].ln2_g); load_tensor(f, &m->layers[i].ln2_b);
        load_tensor(f, &m->layers[i].w_ff1); load_tensor(f, &m->layers[i].w_ff2);
    }
    load_tensor(f, &m->ln_f_g); load_tensor(f, &m->ln_f_b);
    load_tensor(f, &m->w_head);
    fclose(f);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    GPT model;
    init_gpt(&model);
    GPTCache cache = init_cache();
    
    if(argc < 2) { printf("./c1p train | infer\n"); return 1; }
    
    if(strcmp(argv[1], "train") == 0) {
        char* text = "Hello World! Phonex-C1 Refactored. ";
        int len = strlen(text);
        int input[SEQ_LEN], target[SEQ_LEN];
        
        load_model(&model, "model.bin");
        printf("Training...\n");
        
        for(int step=0; step<TRAIN_STEPS; step++) {
            int start = rand() % (len - SEQ_LEN - 1);
            for(int i=0; i<SEQ_LEN; i++) {
                input[i] = (int)text[start+i];
                target[i] = (int)text[start+i+1];
            }
            
            zero_cache_grads(&cache);
            forward(&model, &cache, input, SEQ_LEN);
            
            // FIX: Stable Cross Entropy (LogSumExp)
            float loss = 0;
            for(int i=0; i<SEQ_LEN; i++) {
                float* logits = &cache.logits.data[i*VOCAB_SIZE];
                float max_l = -1e9f;
                for(int j=0; j<VOCAB_SIZE; j++) if(logits[j] > max_l) max_l = logits[j];
                
                float sum_exp = 0.0f;
                for(int j=0; j<VOCAB_SIZE; j++) sum_exp += expf(logits[j] - max_l);
                float log_sum_exp = max_l + logf(sum_exp);
                
                // Loss = log(sum(exp(x))) - x[target]
                loss += (log_sum_exp - logits[target[i]]);
                
                // Grad: p - y
                for(int j=0; j<VOCAB_SIZE; j++) {
                    float p = expf(logits[j] - log_sum_exp);
                    float y = (j == target[i]) ? 1.0f : 0.0f;
                    cache.logits.grad[i*VOCAB_SIZE + j] = (p - y) / SEQ_LEN;
                }
            }
            
            if(step%100==0) printf("Loss: %.4f\n", loss/SEQ_LEN);
            
            backward(&model, &cache, input, SEQ_LEN);
            optimizer_step(&model, step);
        }
        save_model(&model, "model.bin");
    } else {
        load_model(&model, "model.bin");
        char* p = "Hello";
        int ctx[SEQ_LEN] = {0};
        for(int i=0; i<strlen(p); i++) ctx[i] = p[i];
        int cur = strlen(p);
        
        printf("%s", p);
        while(cur < SEQ_LEN) {
            forward(&model, &cache, ctx, cur);
            
            float* logits = &cache.logits.data[(cur-1)*VOCAB_SIZE];
            float max_l = -1e9f;
            for(int j=0; j<VOCAB_SIZE; j++) if(logits[j] > max_l) max_l = logits[j];
            
            float sum=0;
            for(int j=0; j<VOCAB_SIZE; j++) {
                cache.probs.data[j] = expf((logits[j] - max_l)/TEMP);
                sum += cache.probs.data[j];
            }
            for(int j=0; j<VOCAB_SIZE; j++) cache.probs.data[j] /= sum;
            
            int next = sample_top_p(cache.probs.data, VOCAB_SIZE, TOP_P);
            printf("%c", (char)next);
            ctx[cur++] = next;
        }
        printf("\n");
    }
    return 0;
}