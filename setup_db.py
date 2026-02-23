import sqlite3

def setup_database():
    # Verbindung zur Datenbankdatei (wird erstellt, falls nicht vorhanden)
    conn = sqlite3.connect('data/industrie_ki.db')
    cursor = conn.cursor()

    # Tabelle für Maschinendaten und deren KI-Status
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS maschinen_status (
        id INTEGER PRIMARY KEY,
        maschinen_name TEXT,
        abteilung TEXT,
        llm_optimiert BOOLEAN,
        letzte_anomalie DATE,
        rag_zugriff_aktiv BOOLEAN
    )
    ''')

    # Tabelle für Forschungsstatistiken und KI-Projekte
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ki_projekte (
        id INTEGER PRIMARY KEY,
        projekt_name TEXT,
        technologie TEXT,
        effizienz_steigerung_prozent REAL,
        status TEXT
    )
    ''')

    # Beispieldaten einfügen
    maschinen_daten = [
        (1, 'Fräse-01', 'Produktion', 1, '2026-01-15', 1),
        (2, 'Roboter-Arm-Alpha', 'Montage', 0, '2026-02-01', 1),
        (3, 'Spritzguss-M7', 'Produktion', 1, '2025-12-20', 0),
        (4, 'Bohreinheit-B4', 'Instandhaltung', 0, '2026-02-10', 1)
    ]

    projekt_daten = [
        (1, 'RAG-Einführung', 'LLM & ChromaDB', 25.5, 'Abgeschlossen'),
        (2, 'Anomalie-Detection', 'Prädiktive KI', 18.0, 'In Arbeit'),
        (3, 'SQL-Agent-Pilot', 'LangChain SQL', 30.2, 'Testphase')
    ]

    # Daten in die Tabellen einfügen
    cursor.executemany('INSERT OR REPLACE INTO maschinen_status VALUES (?,?,?,?,?,?)', maschinen_daten)
    cursor.executemany('INSERT OR REPLACE INTO ki_projekte VALUES (?,?,?,?,?)', projekt_daten)

    conn.commit()
    conn.close()
    print("Datenbank 'industrie_ki.db' wurde erfolgreich im Ordner /data erstellt.")

if __name__ == "__main__":
    setup_database()