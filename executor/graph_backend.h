// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Version 1: one independent device-launchable CUDA graph per queue slot.
// Graphs, device input/output buffers and their allocations remain caller-owned
// until serve returns. Graphs must contain no host callbacks or host transfers.
// Inputs/outputs are opaque 32-bit words; the existing TCP frame carries them
// in its token fields. Every request supplies exactly input_words and asks for
// output_words. No tokenizer, vocabulary, model metadata or math kernel is used.
// The transport currently limits both counts to max_words(), and preserves its
// FIFO slot retirement and TCP/HTTP framing. It is not an autoregressive API.
unsigned bgs_graph_abi_version(void);
unsigned bgs_graph_max_slots(void);
unsigned bgs_graph_max_words(void);
int bgs_graph_serve(const uint64_t *graphs, const uint64_t *inputs,
                   const uint64_t *outputs, unsigned input_words,
                   unsigned output_words, unsigned seconds);
#ifdef __cplusplus
}
#endif
