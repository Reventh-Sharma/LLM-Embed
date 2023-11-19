```gen_pipe = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
            device='cuda:0'
        )```

Provides same embeddings as:

```full_model(seq_ids).last_hidden_state[0].detach().numpy()```
Where:
```full_model = AutoModel.from_pretrained(model
                                       , output_hidden_states=True
                                       # , use_auth_token=token
                                      )
tokenizer = AutoTokenizer.from_pretrained(model
                                          # , use_auth_token=token
                                         )```
