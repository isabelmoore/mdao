def run_traj_test(case, input_deck)

    for i in range(len(input_params)):
        casenum = case[0]
        input_param[i] = case[i]

    print(f"Running Trajectory Tests for case: {casenum}")
    print(f"Processing case data: \n{case}") 

    problem_name = f'case_{casenum}'
    p = om.Problem(name=problem_name)

    scenario = dymos_generator(problem=p, input_deck=input_deck)
    for i in range(len(input_params)):
        scenario.p.model_options[input_params[i].deck_path]

    scenario.setup()

    
    try:     
        dm.run_problem(scenario.p, run_driver=True, simulate=False, restart=r"/home/imoore/misslemdao/tools/traj_ann/dymos_solution.db")
        # om.n2(scenario.p, outfile="n2_post_run.html")

        with open(p.get_outputs_dir() / "SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
            SNOPT_history = f.read()
