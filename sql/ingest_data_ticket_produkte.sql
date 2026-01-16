-- Ingest data for ticket_produkte
CREATE OR REPLACE TABLE ticket_produkte AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/ticket_produkte.csv', HEADER=true);
