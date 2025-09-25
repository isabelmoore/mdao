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
