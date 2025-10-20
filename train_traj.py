Name: status, Length: 741, dtype: int64
X_train shape: (654, 2), y_train shape: (654, 2, 106)
X_test shape: (73, 2), y_test shape: (73, 2, 106)
TransformerCondDecoder(
  (cond_proj): Linear(in_features=2, out_features=512, bias=True)
  (y_in_proj): Linear(in_features=2, out_features=512, bias=True)
  (pos): PositionalEncoding()
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (out_head): Linear(in_features=512, out_features=2, bias=True)
  (criterion): MSELoss()
)
