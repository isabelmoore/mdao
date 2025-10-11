    else:
        db_filename = "results_2025_10_09_121703.db"  # update to the actual file
        db_path = Path(db_filename)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()
        conn.close()
        results = [dict(row) for row in rows]

    # Filter for successful results (status equal to 1).
    successful_results = [res for res in results if res['status'] == 1]
    sorted_successful = sorted(successful_results, key=lambda r: r['range'], reverse=True)

    print("Top 10 Successful Cases by Range:")
    table = PrettyTable()
    if sorted_successful:
        sample_case = sorted_successful[0]
        excluded_keys = ['status', 'case_id', 'comments']
        field_names = [key for key in sample_case.keys() if key not in excluded_keys]
        table.field_names = field_names

        for i, result in enumerate(sorted_successful[:10]):
            row = []
            for key in field_names:
                value = result[key]
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}" 
                else:
                    formatted_value = value 
                row.append(formatted_value) 
            table.add_row(row)
        print(table)
    # Plot successful results in a single column for each parameter
    n_params = len(field_names)-2  # Get the number of parameters
    fig, axs = plt.subplots(n_params, 1, figsize=(10, 3*n_params), sharex=True)

    # Initialize successful cases for numeric parameters only
    successful_cases = {spec.name: [] for spec in input_params if isinstance(spec, NumericSpec)}  # Track only numeric specs
    successful_targets = [res['range'] for res in successful_results]  # Store corresponding outputs

    # Populate successful cases
    for result in sorted_successful:
        for key in successful_cases.keys():  # Iterate only through numeric specs
            if key in result:  # Ensure the key is present in the result
                successful_cases[key].append(float(result[key]))  # Convert to float safely
 
    for i, key in enumerate(successful_cases.keys()):
        if successful_cases[key]:  # Ensure there are successful cases
            successful_case_values = np.array(successful_cases[key], dtype=float)  # Ensure numeric type
            successful_target_values = np.array(successful_targets, dtype=float)  # Ensure a corresponding target
            
            axs[i].scatter(successful_case_values, successful_target_values, marker='o', label='Successful Cases', alpha=0.7)
            axs[i].set_title(f"Successful Cases for {key}")
            axs[i].set_ylabel("Target Value (Range)")

            # Adjust axes limits smartly based on the successful values
            # Set dynamic axis limits based on the data
            min_x = min(successful_case_values)
            max_x = max(successful_case_values)
            min_y = min(successful_target_values)
            max_y = max(successful_target_values)

            axs[i].set_xlim(min_x - 0.1 * (max_x - min_x), 
                            max_x + 0.1 * (max_x - min_x))  # X limits
            axs[i].set_ylim(min_y - 0.1 * (max_y - min_y), 
                            max_y + 0.1 * (max_y - min_y))  # Y limits
                
            axs[i].grid(True)  # Enable grid on the subplot
            axs[i].legend()
        else:
            print(f"Mismatch in lengths: x_values ({len(x_values)}) and y_values ({len(y_values)}) for {key}.")

    plt.subplots_adjust(hspace=1)  # Increase vertical space between plots
    # axs[-1].set_xlabel("Parameter Value")
    # plt.tight_layout()
    plt.grid(True)    
    plt.savefig("parameters.png")  # Display all plots
