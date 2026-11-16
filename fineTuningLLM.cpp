#include <iostream>
#include <string>
#include <vector>
#include "llama.h"

using namespace std;

// Silent log callback to prevent background thread logs from colliding with std::cout
void silent_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

// Helper function that explicitly maps data into the Structure of Arrays (SoA)
void add_to_batch(llama_batch& batch, llama_token token_id, int pos, bool needs_logits) {
    batch.token[batch.n_tokens] = token_id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = needs_logits;
    batch.n_tokens++;
}

int main() {
    // Register the silent log callback before any llama calls
    ggml_log_set(silent_log_callback, NULL);

    cout << "\nStarting C++ GPT-2 Interactive Autocomplete Engine...\n" << endl;

    llama_backend_init();

    // Load the base GPT-2 model once globally
    llama_model_params model_params = llama_model_default_params();
    llama_model* base_model = llama_model_load_from_file("gpt2-q4_k_m.gguf", model_params);

    if (base_model == nullptr) {
        cout << "Error loading base model." << endl;
        return 1;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(base_model);
    llama_context_params ctx_params = llama_context_default_params();

    // Setup sampler with temperature for creative text completion
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    llama_batch batch = llama_batch_init(512, 0, 1);

    // Set max tokens to generate for epic-length responses
    int max_tokens_to_generate = 150;

    cout << "Engine ready! Type a starting phrase and GPT-2 will finish it." << endl;
    cout << "Type 'exit' or 'quit' to close the program.\n";
    cout << "--------------------------------------------------------\n";

    while (true) {
        cout << "\nEnter starting words: ";
        string prompt;
        getline(cin, prompt);

        if (prompt == "exit" || prompt == "quit") {
            break;
        }

        if (prompt.empty()) {
            continue;
        }

        // Fresh context per prompt guarantees a pristine KV cache state
        llama_context* ctx = llama_init_from_model(base_model, ctx_params);
        if (ctx == nullptr) {
            cout << "Error creating llama context." << endl;
            break;
        }

        cout << "\nPrompt: " << prompt;
        cout.flush();

        // Tokenize user prompt
        vector<llama_token> tokens_list(prompt.size() + 1);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);

        if (n_tokens < 0) {
            tokens_list.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
        }
        tokens_list.resize(n_tokens);

        // Populate batch with prompt tokens
        batch.n_tokens = 0;
        for (int i = 0; i < n_tokens; i++) {
            add_to_batch(batch, tokens_list[i], i, false);
        }

        // Force logits calculation on the final token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0) {
            cout << "\nError during prompt decoding" << endl;
            llama_free(ctx);
            break;
        }

        int n_cur = batch.n_tokens;

        // Generation loop for the autocomplete response
        for (int i = 0; i < max_tokens_to_generate; i++) {
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n_chars >= 0) {
                string piece(buf, n_chars);
                cout << piece;
                cout.flush();
            }

            // Clear batch and feed single new token back in
            batch.n_tokens = 0;
            add_to_batch(batch, new_token_id, n_cur, true);

            if (llama_decode(ctx, batch) != 0) {
                cout << "\nError during decoding" << endl;
                break;
            }

            n_cur++;
        }

        cout << "\n\n[End of Generation]\n" << endl;

        // Clean up context for this prompt iteration silently
        llama_free(ctx);
    }

    // Cleanup global resources on exit
    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_model_free(base_model);
    llama_backend_free();

    return 0;
}