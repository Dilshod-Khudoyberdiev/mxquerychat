-- Ingest data for nutzer_tarifverbund_rechte
CREATE OR REPLACE TABLE nutzer_tarifverbund_rechte AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/nutzer_tarifverbund_rechte.csv', HEADER=true);
