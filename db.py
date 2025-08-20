import sqlite3
import json

def insert_sample_data():
    conn = sqlite3.connect("sox_agent.db")
    c = conn.cursor()

    # Insert sample vendors
    vendors = [
        ("AcmeCorp", "active", json.dumps(["travel", "meals"]), json.dumps(["US", "EU"])),
        ("OfficeTech", "active", json.dumps(["office_supplies"]), json.dumps(["US"])),
        ("GlobalStay", "active", json.dumps(["accommodation", "travel"]), json.dumps(["EU"])),
        ("BlacklistedVendor", "blacklisted", json.dumps([]), json.dumps([])),
        ("GoodEats", "active", json.dumps(["meals"]), json.dumps(["US"])),
        ("CityTaxi", "active", json.dumps(["travel"]), json.dumps(["US"])),
        ("Scott Howard", "active", json.dumps(["unknown"]), json.dumps(["US"])),
        ("OfficeMart", "active", json.dumps(["office_supplies"]), json.dumps(["US"]))
    ]

    c.executemany("INSERT OR REPLACE INTO vendors (vendor_name, status, allowed_categories, regions) VALUES (?, ?, ?, ?)", vendors)

    # Insert sample exceptions
    exceptions = [
        ("RPT001", "Airport shuttle taxi", "Approved due to lack of public transport", 1),
        ("RPT002", "Conference registration fee", "One-time exception approved", 1),
    ]
    c.executemany("INSERT INTO exceptions (report_id, line_item_description, reason, active) VALUES (?, ?, ?, ?)", exceptions)

    # Insert approval thresholds by category (example realistic values)
    approval_thresholds = [
        ("travel", 2000.00),
        ("meals", 150.00),
        ("office_supplies", 300.00),
        ("accommodation", 1000.00),
        ("unknown", 10.00),
    ]
    c.executemany("INSERT OR REPLACE INTO approval_thresholds (category, max_approval_amount) VALUES (?, ?)", approval_thresholds)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    insert_sample_data()
    print("Sample data inserted into the SQLite database.")
