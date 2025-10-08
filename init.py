def run_traj_test(case, input_deck):
    
    casenum= case[0]
    azimuth= case[1]
    range_ = case[2]
    
    print(f"Running Trajectory Tests for case: {casenum}")
    print(f"Processing case data: \n{case}") 

    problem_name = f'case_{casenum}'
    p = om.Problem(name=problem_name)

    scenario = dymos_generator(problem=p, input_deck=input_deck)
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["azimuth"]["equals"] = azimuth
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["range"]["equals"] = range_

    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2

    scenario.setup()

    try:     
        dm.run_problem(scenario.p, run_driver=True, simulate=False, restart=r"/home/imoore/misslemdao/tools/traj_ann/dymos_solution.db")
        # om.n2(scenario.p, outfile="n2_post_run.html")

        with open(p.get_outputs_dir() / "SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
            SNOPT_history = f.read()

        # Define where the exit code information starts
        exit_code_start = SNOPT_history.rfind("SNOPTC EXIT")
        exit_code_end = SNOPT_history.find("\n", exit_code_start)

        # Extract the exit code line
        exit_code = int((SNOPT_history[exit_code_start:exit_code_end]).split()[2])

        # TO DO: fix this tomatch that of major iterattions, not minor
        iter_code_start = SNOPT_history.rfind("No. of iterations")
        iter_code_end = SNOPT_history.find("\n", iter_code_start)

        iterations = int((SNOPT_history[iter_code_start:iter_code_end]).split()[3])


        print("Exit Code", exit_code)
        if exit_code != 0:
            status = 0 # FAILURE
        else:
            status = 1 # SUCCESS
        result = {
            'azimuth': azimuth,
            'range': range,
            'boost_11': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
            },
            'terminal': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
            },
            'status': status,
            'iterations': iterations
        }


# phase 1: finding feasibility region
class TrajectoryEnv: 
    self.input_deck = input_deck
    self.problem = problem
    self.scenario = scenario
    self.ts = "traj_vehicle_0"

    self.init_params = {
        "boost_alpha_0": 
            "bounds": (-30, 30), 
            "timeseries": self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0],            
        "boost_alpha_1": 
            "bounds": (-30, 30), 
            "timeseries": self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1],       
        }

    self.success_log = []
    self.best_params = []
    self.best range = 
