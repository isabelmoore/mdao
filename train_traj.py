    task = partial(simulate_trajectory_case, input_deck=input_deck, report_dir=report_dir)

    results = []
    with ProcessPoolExecutor(max_workers=num_procs) as ex:
        futures = [ex.submit(task, c) for c in cases]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
            results.append(f.result())
    return results
