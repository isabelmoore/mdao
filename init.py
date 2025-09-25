
(py311mdao) C:\Users\N81446> C:/Users/N81446/.conda/envs/py311mdao/python.exe c:/Users/N81446/misslemdao/tools/traj_ann/init_cond_opt/train_init_cond.py
seed alphas: (0.0, 9.0)
built 9 cases with step 0.25
Traceback (most recent call last):
  File "c:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt\train_init_cond.py", line 126, in <module>
    runner.run()
  File "c:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt\train_init_cond.py", line 110, in run
    results = pool.map(worker_fn, cases)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 774, in get
    raise self._value
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 540, in _handle_tasks
    put(task)
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object 'AlphaGridRunner.run.<locals>.<lambda>'

(py311mdao) C:\Users\N81446>
