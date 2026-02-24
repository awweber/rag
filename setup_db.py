"""
setup_db.py – Erstellt und befüllt die SQLite-Datenbank für das Industrie-KI-Assistenzsystem.

Tabellen:
    - maschinen_status:  Aktuelle Zustände aller überwachten Maschinen.
    - anomalie_log:      Protokollierte Anomalien mit Schweregrad und Sensorwerten.
    - ki_projekte:       Metadaten zu internen KI-/ML-Projekten.
    - sensor_statistiken: Aggregierte Sensor-Messwerte pro Maschine und Monat.

Nutzung:
    python setup_db.py          # Erstellt data/industrie_ki.db mit Beispieldaten.
"""

import sqlite3
import os


DB_PATH = os.path.join("data", "industrie_ki.db")


def setup_database():
    """Erstellt alle Tabellen und fügt Demo-Daten ein."""

    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 1. Tabelle: Maschinenstatus – Überblick über alle Produktionsanlagen
    # ------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maschinen_status (
        id                INTEGER PRIMARY KEY,
        maschinen_name    TEXT    NOT NULL,
        abteilung         TEXT    NOT NULL,
        llm_optimiert     BOOLEAN DEFAULT 0,
        letzte_anomalie   DATE,
        rag_zugriff_aktiv BOOLEAN DEFAULT 0,
        betriebsstunden   INTEGER DEFAULT 0,
        zustand           TEXT    DEFAULT 'Normal'
    )
    """)

    # ------------------------------------------------------------------
    # 2. Tabelle: Anomalie-Log – Aufzeichnung auffälliger Sensorereignisse
    # ------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomalie_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        maschinen_id    INTEGER NOT NULL,
        zeitstempel     DATETIME NOT NULL,
        schweregrad     TEXT    CHECK(schweregrad IN ('Niedrig','Mittel','Hoch','Kritisch')),
        sensor_typ      TEXT    NOT NULL,
        messwert        REAL    NOT NULL,
        schwellenwert   REAL    NOT NULL,
        beschreibung    TEXT,
        FOREIGN KEY (maschinen_id) REFERENCES maschinen_status(id)
    )
    """)

    # ------------------------------------------------------------------
    # 3. Tabelle: KI-Projekte – Internes Projektportfolio
    # ------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ki_projekte (
        id                          INTEGER PRIMARY KEY,
        projekt_name                TEXT    NOT NULL,
        technologie                 TEXT,
        effizienz_steigerung_prozent REAL,
        status                      TEXT
    )
    """)

    # ------------------------------------------------------------------
    # 4. Tabelle: Sensor-Statistiken – Monatliche Durchschnittswerte
    # ------------------------------------------------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_statistiken (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        maschinen_id    INTEGER NOT NULL,
        monat           TEXT    NOT NULL,
        sensor_typ      TEXT    NOT NULL,
        mittelwert      REAL,
        maximum         REAL,
        minimum         REAL,
        standardabweichung REAL,
        FOREIGN KEY (maschinen_id) REFERENCES maschinen_status(id)
    )
    """)

    # ==================  Beispieldaten  ==================

    maschinen_daten = [
        (1, 'Fräse-01',          'Produktion',      1, '2026-01-15', 1, 14200, 'Normal'),
        (2, 'Roboter-Arm-Alpha', 'Montage',         0, '2026-02-01', 1,  8700, 'Warnung'),
        (3, 'Spritzguss-M7',     'Produktion',      1, '2025-12-20', 0, 22300, 'Normal'),
        (4, 'Bohreinheit-B4',    'Instandhaltung',  0, '2026-02-10', 1,  5100, 'Kritisch'),
        (5, 'Laser-Schneider-L2','Produktion',      1, '2026-01-28', 1, 11500, 'Normal'),
        (6, 'Presswerk-P9',      'Umformung',       0, None,         0, 31000, 'Normal'),
    ]

    anomalie_daten = [
        (1, 1, '2026-01-15 08:23:11', 'Mittel',    'Vibration',    4.7,  3.5,  'Erhöhte Vibration an Spindel'),
        (2, 2, '2026-02-01 14:05:33', 'Hoch',      'Temperatur',  87.3, 75.0,  'Überhitzung Gelenk 3'),
        (3, 4, '2026-02-10 03:12:44', 'Kritisch',  'Druck',        2.1,  5.0,  'Druckabfall Hydraulik – Notabschaltung'),
        (4, 1, '2025-11-22 19:45:00', 'Niedrig',   'Vibration',    3.6,  3.5,  'Leicht erhöhte Vibration nach Werkzeug­wechsel'),
        (5, 3, '2025-12-20 11:30:15', 'Mittel',    'Temperatur',  72.1, 70.0,  'Temperatur leicht über Schwelle'),
        (6, 5, '2026-01-28 22:10:05', 'Hoch',      'Stromstärke', 16.8, 14.0,  'Laser-Überlast erkannt'),
        (7, 2, '2026-01-18 09:55:20', 'Niedrig',   'Vibration',    2.9,  3.5,  'Kurzzeitiger Vibrations-Peak'),
        (8, 4, '2026-01-05 06:00:00', 'Hoch',      'Druck',        3.8,  5.0,  'Druckabfall langsam – Leckage vermutet'),
    ]

    projekt_daten = [
        (1, 'RAG-Einführung',       'LLM & ChromaDB',   25.5, 'Abgeschlossen'),
        (2, 'Anomalie-Detection',   'Prädiktive KI',    18.0, 'In Arbeit'),
        (3, 'SQL-Agent-Pilot',      'LangChain SQL',    30.2, 'Testphase'),
        (4, 'Predictive Maintenance','LSTM Autoencoder', 12.8, 'In Arbeit'),
        (5, 'Digitaler Zwilling',   'Simulation + LLM',  None, 'Geplant'),
    ]

    sensor_stats = [
        (1, 1, '2025-12', 'Vibration',   2.8, 3.6, 1.9, 0.42),
        (2, 1, '2026-01', 'Vibration',   3.1, 4.7, 2.1, 0.68),
        (3, 2, '2025-12', 'Temperatur', 68.2, 74.5, 61.0, 3.10),
        (4, 2, '2026-01', 'Temperatur', 71.5, 87.3, 63.0, 5.20),
        (5, 3, '2025-12', 'Temperatur', 65.0, 72.1, 58.4, 2.80),
        (6, 4, '2025-12', 'Druck',       5.2,  5.5,  4.8, 0.15),
        (7, 4, '2026-01', 'Druck',       4.6,  5.1,  3.8, 0.45),
        (8, 5, '2026-01', 'Stromstärke',12.5, 16.8, 10.1, 1.90),
    ]

    # Daten einfügen (REPLACE bei Konflikten)
    cursor.executemany(
        "INSERT OR REPLACE INTO maschinen_status VALUES (?,?,?,?,?,?,?,?)",
        maschinen_daten,
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO anomalie_log VALUES (?,?,?,?,?,?,?,?)",
        anomalie_daten,
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO ki_projekte VALUES (?,?,?,?,?)",
        projekt_daten,
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO sensor_statistiken VALUES (?,?,?,?,?,?,?,?)",
        sensor_stats,
    )

    conn.commit()
    conn.close()
    print(f"Datenbank '{DB_PATH}' wurde erfolgreich erstellt und befüllt.")
    print("  • maschinen_status   – 6 Einträge")
    print("  • anomalie_log       – 8 Einträge")
    print("  • ki_projekte        – 5 Einträge")
    print("  • sensor_statistiken – 8 Einträge")


if __name__ == "__main__":
    setup_database()