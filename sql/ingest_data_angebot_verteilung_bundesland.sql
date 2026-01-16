-- Ingest data for angebot_verteilung_bundesland
CREATE OR REPLACE TABLE angebot_verteilung_bundesland AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/angebot_verteilung_bundesland.csv', HEADER=true);
