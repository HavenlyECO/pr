#!/usr/bin/env python3
"""
fix_progress.py - Add progress indicators to main.py to prevent the appearance of hanging
"""
import os
from datetime import datetime


def fix_progress_indicators():
    """Add progress indicators to main.py so it shows activity during processing"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting progress indicator fix...")

    # First, back up the current file
    backup_name = f"main.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.system(f"cp main.py {backup_name}")
    print(f"Created backup: {backup_name}")

    # Read the current file
    with open('main.py', 'r') as f:
        content = f.read()

    # Add a progress counter at the beginning of evaluate_h2h_all_tomorrow function
    start_marker = "    events = fetch_events(sport_key, regions=regions)"
    replacement = """    # Show initial status message
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching events for {sport_key}...")
    events = fetch_events(sport_key, regions=regions)"""

    content = content.replace(start_marker, replacement)

    # Add a counter after events are fetched
    start_marker = "    if verbose or True:"
    replacement = """    # Always show event count, regardless of verbose mode
    event_count = len(events)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {event_count} events to analyze")

    if verbose or True:"""

    content = content.replace(start_marker, replacement)

    # Add progress indicator at the start of the event loop
    start_marker = "    for event in events:"
    replacement = """    # Set up progress tracking
    processed = 0
    total = len(events)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting analysis of {total} events...")

    for event in events:
        processed += 1"""

    content = content.replace(start_marker, replacement)

    # Add progress update inside the main loop
    start_marker = "        if verbose:"
    replacement = """        # Always show minimal progress
        if processed % 3 == 0 or processed == total:  # Show progress every 3 events or at the end
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing event {processed}/{total}: {away} at {home}")

        if verbose:"""

    content = content.replace(start_marker, replacement)

    # Add completion message at the end
    start_marker = "    if verbose:"
    replacement = """    # Show completion status
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis complete: {len(results)} betting opportunities found\\n")

    if verbose:"""

    content = content.replace(start_marker, replacement)

    # Write the updated content
    with open('main.py', 'w') as f:
        f.write(content)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully added progress indicators to main.py")
    print("Run your script with 'python3 main.py' to see progress updates during processing")

    return True


if __name__ == "__main__":
    fix_progress_indicators()
