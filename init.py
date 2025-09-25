seed alphas: (0.0, 9.0)
built 9 cases with step 0.25
multiprocessing.pool.RemoteTraceback:
"""
Traceback (most recent call last):
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "c:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt\train_init_cond.py", line 65, in _worker
    os.chdir(repo_root_str)
TypeError: chdir: path should be string, bytes or os.PathLike, not list
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "c:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt\train_init_cond.py", line 129, in <module>
    runner.run()
  File "c:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt\train_init_cond.py", line 113, in run
    results = pool.map(worker_fn, cases)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\N81446\.conda\envs\py311mdao\Lib\multiprocessing\pool.py", line 774, in get
    raise self._value
TypeError: chdir: path should be string, bytes or os.PathLike, not list
