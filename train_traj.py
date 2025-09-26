Name: azimuth, dtype: float64
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/imoore/misslemdao/tools/traj_ann/traj_generator/nn_functions.py", line 49, in predicting
[rank0]:     preds = self.model.generate(batch_X_test, L=L).cpu().numpy()
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/misslemdao/tools/traj_ann/traj_generator/nn_models.py", line 164, in generate
[rank0]:     h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/transformer.py", line 602, in forward
[rank0]:     output = mod(
[rank0]:              ^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/transformer.py", line 1091, in forward
[rank0]:     + self._mha_block(
[rank0]:       ^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/transformer.py", line 1127, in _mha_block
[rank0]:     x = self.multihead_attn(
[rank0]:         ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/activation.py", line 1368, in forward
[rank0]:     attn_output, attn_output_weights = F.multi_head_attention_forward(
[rank0]:                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/functional.py", line 6014, in multi_head_attention_forward
[rank0]:     is_batched = _mha_shape_check(
[rank0]:                  ^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/functional.py", line 5786, in _mha_shape_check
[rank0]:     assert key.dim() == 3 and value.dim() == 3, (
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: AssertionError: For batched (3-D) `query`, expected `key` and `value` to be 3-D but found 4-D and 4-D tensors respectively

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/imoore/misslemdao/tools/traj_ann/traj_generator/train_nn.py", line 470, in <module>
[rank0]:     fn.plot_mse_polar(X_df, scaler_x, scaler_y)
[rank0]:   File "/home/imoore/misslemdao/tools/traj_ann/traj_generator/nn_functions.py", line 137, in plot_mse_polar
[rank0]:     predicted_alpha, predicted_bank = self.predicting(azimuth, range_, scaler_x, scaler_y)
[rank0]:                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/misslemdao/tools/traj_ann/traj_generator/nn_functions.py", line 52, in predicting
[rank0]:     out = self.model(batch_X_test)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/imoore/miniforge3/envs/py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: TypeError: TransformerCondDecoder.forward() missing 1 required positional argument: 'y_prev'
E0925 17:40:00.218000 16555 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 16589) of binary: /home/imoore/miniforge3/envs/py311/bin/python3.11
Traceback (most recent call last):
