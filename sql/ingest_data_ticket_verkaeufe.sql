-- Ingest data for ticket_verkaeufe
CREATE OR REPLACE TABLE ticket_verkaeufe AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/ticket_verkaeufe.csv', HEADER=true);
