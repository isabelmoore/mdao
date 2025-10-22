import os, multiprocessing as mp, tempfile, contextlib
from functools import partial
from tqdm import tqdm

def _init_mp_context():
    try:
        mp.set_start_method('spawn', force=False)  # safe no-op if already set
    except RuntimeError:
        pass

def simulate_trajectory(case, input_deck, report_dir):
    casenum = case[0]
    case_dir = (report_dir / 'cases' / f'case_{casenum}').resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    # Isolate *everything* in this directory
    #  - cwd for DATCOM's fort.* files
    #  - TMPDIR in case the EXE uses tmp
    env_backup = dict(os.environ)
    os.environ["OPENMDAO_WORKDIR"] = str(case_dir)
    os.environ["TMPDIR"] = str(case_dir)
    os.environ["TEMP"] = str(case_dir)
    os.environ["TMP"] = str(case_dir)

    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_chdir(case_dir))

            # (Optional) disable extra reports to cut I/O contention
            p = om.Problem(name=str(case_dir), reports=[])

            scenario = dymos_generator(problem=p, input_deck=input_deck)
            scenario.setup()

            # If the EXE wrapper allows a run directory, set it here too.
            # e.g., scenario.p.model.<your_comp>.options['run_dir'] = str(case_dir)

            dm.run_problem(scenario.p, run_driver=True, simulate=False)

            final_range = extract_exit_code(p)
            return {'range': final_range, 'status': 1, 'comments': 'SUCCESS'}
    except Exception as e:
        return {'range': 0, 'status': 0, 'comments': f"Error: {e}"}
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

if __name__ == "__main__":
    _init_mp_context()
    full_input_samples = ((0,), (1,), (2,), (3,))
    with mp.Pool(processes=num_processors, initializer=_init_mp_context) as pool:
        task = partial(simulate_trajectory, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task, full_input_samples, chunksize=1),
                            total=len(full_input_samples),
                            desc="Processing Samples"))
