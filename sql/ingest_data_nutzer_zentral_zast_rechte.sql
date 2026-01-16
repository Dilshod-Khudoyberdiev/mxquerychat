-- Ingest data for nutzer_zentral_zast_rechte
CREATE OR REPLACE TABLE nutzer_zentral_zast_rechte AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/nutzer_zentral_zast_rechte.csv', HEADER=true);
