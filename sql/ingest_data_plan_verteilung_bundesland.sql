-- Ingest data for plan_verteilung_bundesland
CREATE OR REPLACE TABLE plan_verteilung_bundesland AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/plan_verteilung_bundesland.csv', HEADER=true);
