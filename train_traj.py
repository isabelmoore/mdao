    cases = generate_design_space(variables, num_samples)
    with mp.Pool(processes=num_processors, initializer=_init_mp_context) as pool:
        task = partial(simulate_trajectory, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task, cases, chunksize=1),
                            total=len(cases),
                            desc="Processing Samples"))
