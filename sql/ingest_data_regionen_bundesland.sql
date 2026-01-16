-- Ingest data for regionen_bundesland
CREATE OR REPLACE TABLE regionen_bundesland AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/regionen_bundesland.csv', HEADER=true);
