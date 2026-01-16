-- Ingest data for tarifverbuende
CREATE OR REPLACE TABLE tarifverbuende AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/tarifverbuende.csv', HEADER=true);
