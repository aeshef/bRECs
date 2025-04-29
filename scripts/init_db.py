import sys
from pathlib import Path

from db.models import create_all_tables

if __name__ == "__main__":
    print("-" * 30)
    print("Initializing database...")
    print("-" * 30)
    try:
        create_all_tables()
        print("-" * 30)
        print("Database initialization finished.")
        print("Run this script on the server as well after deployment.")
        print("-" * 30)
    except Exception as e:
        print("\n" + "=" * 30)
        print("!!! DATABASE INITIALIZATION FAILED !!!")
        print(f"!!! Error: {e}")
        print("=" * 30)
        sys.exit(1)

