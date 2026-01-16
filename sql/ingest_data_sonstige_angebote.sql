-- Ingest data for sonstige_angebote
CREATE OR REPLACE TABLE sonstige_angebote AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/sonstige_angebote.csv', HEADER=true);
