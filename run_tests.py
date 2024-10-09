import main

csvs = ['data_processed.csv', 'data_processed_scaled.csv', 'merged_data_processed.csv', 'merged_data_processed_scaled.csv']
names = ['results_data_processed.csv', 'results_data_processed_scaled.csv', 'results_merged_data_processed.csv', 'results_merged_data_processed_scaled.csv']

for i in range(len(csvs)):
    main.runTests(csvs[i], names[i])