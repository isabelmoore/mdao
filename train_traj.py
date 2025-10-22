[{'range': 0, 'status': 1, 'comments': 'SUCCESS'}, {'range': 0, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 324 of file misdat.for (unit = 2, file = 'fort.2')\nFortran runtime error: File cannot be deleted\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4a6299\n#4  0x457bac\n#5  0x40186e\n#6  0x4e11ef\n#7  0x401e9d\n3  0x4ad70d\n#4  0x476ecd\n#5  0x459027\n#6  0x4788f0\n#7  0x477d8b\n#8  0x47760e\n#9  0x457aff\n#10  0x40186e\n#11  0x4e11ef\n#12  0x401e9d\n"}, {'range': 0, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 67 of file readcd.for (unit = 2, file = 'fort.2')\nFortran runtime error: Sequential READ or WRITE not allowed after EOF marker, possibly use REWIND or BACKSPACE\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4ad70d\n#4  0x476ecd\n#5  0x459027\n#6  0x4788f0\n#7  0x477d8b\n#8  0x47760e\n#9  0x457aff\n#10  0x40186e\n#11  0x4e11ef\n#12  0x401e9d\n"}, {'range': 0, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 324 of file misdat.for (unit = 2, file = 'fort.2')\nFortran runtime error: File cannot be deleted\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4a6299\n#4  0x457bac\n#5  0x40186e\n#6  0x4e11ef\n#7  0x401e9d\n3  0x4ad70d\n#4  0x476ecd\n#5  0x459027\n#6  0x4788f0\n#7  0x477d8b\n#8  0x47760e\n#9  0x457aff\n#10  0x40186e\n#11  0x4e11ef\n#12  0x401e9d\n"}]

    full_input_samples = ((0,), (1,), (2,), (3,))
    # results = simulate_trajectory(input_deck=input_deck, report_dir=report_dir)
    with multiprocessing.Pool(processes=num_processors) as pool:
        print("Created pool...")
        task_partial = partial(simulate_trajectory, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task_partial, full_input_samples), total=len(full_input_samples), desc="Processing Samples"))



def simulate_trajectory(case, input_deck, report_dir):
    casenum = case[0]
    try:
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
        #     warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
        #     buf = io.StringIO()
        #     with contextlib.redirect_stdout(buf):
        print("report_dir:", report_dir)
        case_dir = report_dir / 'cases' / f'case_{casenum}'
        case_dir.mkdir(parents=True, exist_ok=True)
        problem_name = str(case_dir)                
        os.environ["OPENMDAO_WORKDIR"] = str(case_dir)
        p = om.Problem(name=problem_name, reports=["inputs", "optimizer"])
        scenario = dymos_generator(problem=p, input_deck=input_deck)
        # for i, spec in enumerate(input_params):
        #     value = case[i + 1]
        #     set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)
        scenario.setup()
        dm.run_problem(
            scenario.p,
            run_driver=True,
            simulate=False,
        )
        final_range = extract_exit_code(p)
        result = {
            # 'input_params': case,
            'range': final_range,
            'status': 1,
            'comments': "SUCCESS"
        }
    except Exception as e:
        result = {
            # 'input_params': case,
            'range': 0,
            'status': 0,
            'comments': f"Error: {e}"
        }
    return result
from multiprocessing import Pool
