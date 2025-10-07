from app.preprocess.preprocess_general import (load_races,
                            process_countries,
                            build_driver_country_table,
                            build_circuit_country_table,
                            build_constructor_country_table)
from app.preprocess.preprocess_mainrace import serve_mainrace_df, create_training_datasets

races = load_races()
countries = process_countries()
drivers = build_driver_country_table()

constructor = build_constructor_country_table()
circuits = build_circuit_country_table()

df = serve_mainrace_df()
create_training_datasets(df=df)
#export df to csv
print("Done loading and processing")