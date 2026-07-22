import argparse
import sys
from web.app import start_server

def main():
    parser = argparse.ArgumentParser(description="Advanced Smart Attendance Monitoring System")
    parser.add_argument('--run-web', action='store_true', help='Start the FastAPI Web Dashboard')
    
    args = parser.parse_args()
    
    if args.run_web or len(sys.argv) == 1:
        print("Starting Advanced Attendance Web Server on http://localhost:8000")
        start_server()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

