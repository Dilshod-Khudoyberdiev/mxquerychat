-- Auto-generated schema DDL (synthetic dataset)

CREATE TABLE angebot_verteilung_bundesland (
  jahr BIGINT,
  monat BIGINT,
  tarifverbund_id BIGINT,
  angebot_gruppe BIGINT,
  bundesland_code2 BIGINT,
  anteil DOUBLE
);

CREATE TABLE meldestellen (
  meldestelle_id BIGINT,
  meldestelle_nummer BIGINT,
  meldestelle_name VARCHAR,
  meldestelle_code VARCHAR,
  organisation_id BIGINT,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);

CREATE TABLE nutzer (
  nutzer_id BIGINT,
  benutzername VARCHAR,
  vorname VARCHAR,
  nachname VARCHAR,
  email VARCHAR
);

CREATE TABLE nutzer_bundesland_rechte (
  rechte_id BIGINT,
  nutzer_id BIGINT,
  bundesland_id BIGINT,
  darf_upload BOOLEAN,
  darf_download BOOLEAN,
  darf_historie_sehen BOOLEAN
);

CREATE TABLE nutzer_tarifverbund_rechte (
  rechte_id BIGINT,
  nutzer_id BIGINT,
  tarifverbund_id BIGINT,
  darf_upload BOOLEAN,
  darf_download BOOLEAN,
  darf_historie_sehen BOOLEAN
);

CREATE TABLE nutzer_zast_rechte (
  rechte_id BIGINT,
  nutzer_id BIGINT,
  zast_id VARCHAR,
  darf_upload BOOLEAN,
  darf_download BOOLEAN,
  darf_ansehen BOOLEAN
);

CREATE TABLE nutzer_zentral_zast_rechte (
  rechte_id BIGINT,
  nutzer_id BIGINT,
  zentral_zast_id VARCHAR,
  darf_upload BOOLEAN,
  darf_download BOOLEAN,
  darf_ansehen BOOLEAN
);

CREATE TABLE plan_umsatz (
  monat BIGINT,
  jahr BIGINT,
  tarifverbund_id BIGINT,
  umsatz_eur DOUBLE
);

CREATE TABLE plan_verteilung_bundesland (
  jahr BIGINT,
  monat BIGINT,
  tarifverbund_id BIGINT,
  bundesland_code2 BIGINT,
  anteil DOUBLE
);

CREATE TABLE postleitzahlen (
  plz_id BIGINT,
  plz VARCHAR,
  ort VARCHAR,
  kreis_code BIGINT,
  bundesland_code2 VARCHAR,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);

CREATE TABLE regionen_bundesland (
  bundesland_id BIGINT,
  bundesland_name VARCHAR,
  iso_code VARCHAR,
  bundesland_code2 VARCHAR,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);

CREATE TABLE sonstige_angebote (
  monat BIGINT,
  jahr BIGINT,
  meldestelle_code VARCHAR,
  tarifverbund_id BIGINT,
  angebot_gruppe BIGINT,
  umsatz_eur DOUBLE
);

CREATE TABLE tarifverbuende (
  tarifverbund_id BIGINT,
  name VARCHAR,
  kuerzel VARCHAR,
  status VARCHAR,
  erstellt_am TIMESTAMP,
  geaendert_am TIMESTAMP
);

CREATE TABLE ticket_produkte (
  ticket_id BIGINT,
  ticket_name VARCHAR,
  ticket_code BIGINT,
  preis_eur DOUBLE,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);

CREATE TABLE ticket_verkaeufe (
  monat BIGINT,
  jahr BIGINT,
  meldestelle_code VARCHAR,
  tarifverbund_id BIGINT,
  ticket_code BIGINT,
  anzahl BIGINT,
  preis_eur DOUBLE,
  umsatz_eur DOUBLE,
  gueltig_ab DATE,
  plz BIGINT
);

CREATE TABLE ticket_verteilung_bundesland (
  jahr BIGINT,
  monat BIGINT,
  tarifverbund_id BIGINT,
  ticket_code BIGINT,
  bundesland_code2 BIGINT,
  anteil DOUBLE
);

CREATE TABLE verteilung_kopf (
  verteilung_id BIGINT,
  organisation_id BIGINT,
  typ_code BIGINT,
  start_datum DATE,
  end_datum DATE,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);

CREATE TABLE verteilung_positionen (
  position_id BIGINT,
  verteilung_id BIGINT,
  bundesland_id BIGINT,
  anteil DOUBLE,
  erstellt_von_user_id BIGINT,
  erstellt_am TIMESTAMP,
  geaendert_von_user_id BIGINT,
  geaendert_am TIMESTAMP
);
